using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// Paged KV-cache using block-based allocation from a shared <see cref="KvBlockPool"/>.
/// Memory is allocated in fixed-size blocks on demand, reducing waste for variable-length sequences.
/// Attention kernel compatibility is maintained via staging buffers that gather blocks into
/// contiguous views for <see cref="GetKeysRef"/>/<see cref="GetValuesRef"/>.
/// </summary>
public sealed unsafe class PagedKvCache : IKvCache
{
    private readonly KvBlockPool _pool;
    private readonly KvBlockTable _blockTable;
    private readonly int _numLayers;
    private readonly int _kvStride;
    private readonly int _maxSeqLen;

    // Pre-allocated staging buffers for contiguous view (reused across GetKeysRef/GetValuesRef calls)
    private readonly nint _keyStagingPtr;
    private readonly nint _valueStagingPtr;
    private readonly long _stagingBytes;

    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _blockTable.CurrentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>The block table mapping logical positions to physical blocks.</summary>
    internal KvBlockTable BlockTable => _blockTable;

    /// <summary>
    /// Creates a new paged KV-cache backed by the given block pool.
    /// </summary>
    /// <param name="pool">Shared block pool for allocation.</param>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="kvStride">KV stride (numKvHeads * headDim).</param>
    /// <param name="maxSeqLen">Maximum sequence length this cache can hold.</param>
    public PagedKvCache(KvBlockPool pool, int numLayers, int kvStride, int maxSeqLen)
    {
        _pool = pool;
        _numLayers = numLayers;
        _kvStride = kvStride;
        _maxSeqLen = maxSeqLen;
        _blockTable = new KvBlockTable(pool);

        // Pre-allocate staging buffers (one per K and V, sized for maxSeqLen)
        _stagingBytes = (long)maxSeqLen * kvStride * sizeof(float);
        _keyStagingPtr = (nint)NativeMemory.AlignedAlloc((nuint)_stagingBytes, 64);
        _valueStagingPtr = (nint)NativeMemory.AlignedAlloc((nuint)_stagingBytes, 64);
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        int seqLen = positions.Length;
        if (seqLen == 0) return;

        int maxPos = _blockTable.CurrentLength - 1;

        float* kSrc = (float*)keys.DataPointer;
        float* vSrc = (float*)values.DataPointer;

        int rowBytes = _kvStride * sizeof(float);
        int blockSize = _pool.BlockSize;

        // Validate and find max position, ensure capacity upfront
        for (int v = 0; v < seqLen; v++)
        {
            int pos = positions[v];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds max cache length {_maxSeqLen}.");
            if (pos > maxPos) maxPos = pos;
        }
        _blockTable.EnsureCapacity(maxPos + 1);

        // Batch contiguous tokens within the same block into a single memcpy.
        // During prefill, positions are sequential (0,1,2,...N) so most tokens
        // share a block and can be copied together.
        int i = 0;
        while (i < seqLen)
        {
            int pos = positions[i];
            _blockTable.EnsureWritable(pos);
            var (blockId, offset) = _blockTable.Resolve(pos);

            // Count how many consecutive tokens fit in this same block
            int runLen = 1;
            int remaining = blockSize - offset;
            while (runLen < remaining && i + runLen < seqLen &&
                   positions[i + runLen] == pos + runLen)
            {
                runLen++;
            }

            float* kDst = _pool.GetKeyPtr(blockId, layerIndex) + offset * _kvStride;
            float* vDst = _pool.GetValuePtr(blockId, layerIndex) + offset * _kvStride;
            long batchBytes = (long)runLen * rowBytes;

            Buffer.MemoryCopy(kSrc + (long)i * _kvStride, kDst, batchBytes, batchBytes);
            Buffer.MemoryCopy(vSrc + (long)i * _kvStride, vDst, batchBytes, batchBytes);

            i += runLen;
        }

        int newLength = maxPos + 1;
        if (newLength > _blockTable.CurrentLength)
            _blockTable.Advance(newLength);
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        var kRef = new TensorRef(positions.Length, _kvStride, keys.DType, keys.DeviceId, keys.DataPointer);
        var vRef = new TensorRef(positions.Length, _kvStride, values.DType, values.DeviceId, values.DataPointer);
        Update(kRef, vRef, positions, layerIndex);
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetKeysRef(int layerIndex)
    {
        GatherIntoStaging(_keyStagingPtr, layerIndex, isKey: true);
        return new TensorRef(_blockTable.CurrentLength, _kvStride, DType.Float32, -1, _keyStagingPtr);
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetValuesRef(int layerIndex)
    {
        GatherIntoStaging(_valueStagingPtr, layerIndex, isKey: false);
        return new TensorRef(_blockTable.CurrentLength, _kvStride, DType.Float32, -1, _valueStagingPtr);
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
    {
        GatherIntoStaging(_keyStagingPtr, layerIndex, isKey: true);
        var shape = new TensorShape(_blockTable.CurrentLength, _kvStride);
        return new TensorView(shape, DType.Float32, -1, _keyStagingPtr);
    }

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
    {
        GatherIntoStaging(_valueStagingPtr, layerIndex, isKey: false);
        var shape = new TensorShape(_blockTable.CurrentLength, _kvStride);
        return new TensorView(shape, DType.Float32, -1, _valueStagingPtr);
    }

    /// <summary>
    /// Resets the visible length of the cache to the given position.
    /// Used by prompt caching to truncate to the matched prefix length.
    /// </summary>
    internal void SetCurrentLength(int length)
    {
        if ((uint)length > (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length));
        _blockTable.SetCurrentLength(length);
    }

    /// <summary>
    /// Gathers block data into a contiguous staging buffer for attention kernel consumption.
    /// Copies block-by-block in logical order.
    /// </summary>
    private void GatherIntoStaging(nint stagingPtr, int layerIndex, bool isKey)
    {
        int currentLength = _blockTable.CurrentLength;
        if (currentLength == 0) return;

        int blockSize = _pool.BlockSize;
        int rowBytes = _kvStride * sizeof(float);
        float* dst = (float*)stagingPtr;

        int fullBlocks = currentLength / blockSize;
        int tailTokens = currentLength % blockSize;

        // Copy full blocks (contiguous within each block)
        long blockBytes = (long)blockSize * rowBytes;
        for (int b = 0; b < fullBlocks; b++)
        {
            var (blockId, _) = _blockTable.Resolve(b * blockSize);
            float* src = isKey
                ? _pool.GetKeyPtr(blockId, layerIndex)
                : _pool.GetValuePtr(blockId, layerIndex);

            Buffer.MemoryCopy(src, dst + (long)b * blockSize * _kvStride, blockBytes, blockBytes);
        }

        // Copy partial tail block
        if (tailTokens > 0)
        {
            var (blockId, _) = _blockTable.Resolve(fullBlocks * blockSize);
            float* src = isKey
                ? _pool.GetKeyPtr(blockId, layerIndex)
                : _pool.GetValuePtr(blockId, layerIndex);

            long tailBytes = (long)tailTokens * rowBytes;
            Buffer.MemoryCopy(src, dst + (long)fullBlocks * blockSize * _kvStride, tailBytes, tailBytes);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;

        if (disposing)
        {
            // Only touch managed objects during explicit Dispose
            _blockTable.Free();
        }

        // Free unmanaged staging buffers regardless
        if (_keyStagingPtr != 0)
            NativeMemory.AlignedFree((void*)_keyStagingPtr);
        if (_valueStagingPtr != 0)
            NativeMemory.AlignedFree((void*)_valueStagingPtr);
    }

    /// <summary>Releases unmanaged staging buffers if <see cref="Dispose()"/> was not called.</summary>
    ~PagedKvCache() => Dispose(disposing: false);
}
