using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// Simple pre-allocated KV-cache. Stores per-layer K and V projections in contiguous
/// FP32 buffers of shape <c>[maxSeqLen, numKvHeads * headDim]</c>, 64-byte aligned.
/// </summary>
public sealed unsafe class SimpleKvCache : IKvCache
{
    private readonly nint[] _keys;   // [numLayers] pointers to K buffers
    private readonly nint[] _values; // [numLayers] pointers to V buffers
    private readonly int _numLayers;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _maxSeqLen;
    private readonly int _kvStride; // numKvHeads * headDim
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Total bytes allocated for KV-cache buffers.</summary>
    public long AllocatedBytes => (long)_numLayers * 2 * _maxSeqLen * _kvStride * sizeof(float);

    /// <summary>
    /// Creates a new KV-cache with pre-allocated buffers for all layers.
    /// </summary>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numKvHeads">Number of KV attention heads per layer.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="maxSeqLen">Maximum number of positions this cache can hold.</param>
    public SimpleKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen)
    {
        _numLayers = numLayers;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;
        _kvStride = numKvHeads * headDim;

        _keys = new nint[numLayers];
        _values = new nint[numLayers];

        nuint bufferBytes = (nuint)((long)maxSeqLen * _kvStride * sizeof(float));
        for (int i = 0; i < numLayers; i++)
        {
            _keys[i] = (nint)NativeMemory.AlignedAlloc(bufferBytes, 64);
            _values[i] = (nint)NativeMemory.AlignedAlloc(bufferBytes, 64);
        }
    }

    /// <summary>Releases unmanaged buffers if <see cref="Dispose()"/> was not called.</summary>
    ~SimpleKvCache() => Dispose(disposing: false);

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        int seqLen = positions.Length;
        int maxPos = -1;

        float* kSrc = (float*)keys.DataPointer;
        float* vSrc = (float*)values.DataPointer;
        float* kDst = (float*)_keys[layerIndex];
        float* vDst = (float*)_values[layerIndex];

        int rowBytes = _kvStride * sizeof(float);

        for (int i = 0; i < seqLen; i++)
        {
            int pos = positions[i];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {pos} exceeds max cache length {_maxSeqLen}.");

            if (pos > maxPos) maxPos = pos;

            Buffer.MemoryCopy(
                kSrc + i * _kvStride,
                kDst + pos * _kvStride,
                rowBytes, rowBytes);

            Buffer.MemoryCopy(
                vSrc + i * _kvStride,
                vDst + pos * _kvStride,
                rowBytes, rowBytes);
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
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
        => new(_currentLength, _kvStride, DType.Float32, -1, _keys[layerIndex]);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef GetValuesRef(int layerIndex)
        => new(_currentLength, _kvStride, DType.Float32, -1, _values[layerIndex]);

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ITensor GetKeys(int layerIndex)
    {
        var shape = new TensorShape(_currentLength, _kvStride);
        return new TensorView(shape, DType.Float32, -1, _keys[layerIndex]);
    }

    /// <inheritdoc/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ITensor GetValues(int layerIndex)
    {
        var shape = new TensorShape(_currentLength, _kvStride);
        return new TensorView(shape, DType.Float32, -1, _values[layerIndex]);
    }

    /// <summary>
    /// Resets the visible length of the cache to the given position.
    /// Data beyond this position remains in memory but will be overwritten.
    /// Used by prompt caching to truncate to the matched prefix length.
    /// </summary>
    internal void SetCurrentLength(int length)
    {
        if ((uint)length > (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
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

        for (int i = 0; i < _numLayers; i++)
        {
            if (_keys[i] != 0)
            {
                NativeMemory.AlignedFree((void*)_keys[i]);
                _keys[i] = 0;
            }
            if (_values[i] != 0)
            {
                NativeMemory.AlignedFree((void*)_values[i]);
                _values[i] = 0;
            }
        }
    }
}
