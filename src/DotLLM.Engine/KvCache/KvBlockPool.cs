using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// Pool of fixed-size KV blocks. Each block stores <see cref="BlockSize"/> tokens of K and V data
/// for all layers. Thread-safe allocation and deallocation with reference counting.
/// Inspired by OS page frame allocator.
/// </summary>
public sealed unsafe class KvBlockPool : IDisposable
{
    private readonly int _blockSize;
    private readonly int _numLayers;
    private readonly int _kvStride;     // numKvHeads * headDim
    private readonly int _totalBlocks;
    private readonly int _blockFloats;  // blockSize * kvStride (floats per block per layer)

    // Per-layer contiguous storage. Block i for layer j:
    //   keys at _keyBuffers[j] + (long)i * _blockFloats * sizeof(float)
    private readonly nint[] _keyBuffers;
    private readonly nint[] _valueBuffers;

    // Free list (stack-based)
    private readonly int[] _freeStack;
    private int _freeCount;
    private readonly object _lock = new();

    // Reference counting per block
    private readonly int[] _refCounts;

    private bool _disposed;

    /// <summary>Number of tokens per block.</summary>
    public int BlockSize => _blockSize;

    /// <summary>Total number of blocks in the pool.</summary>
    public int TotalBlocks => _totalBlocks;

    /// <summary>Number of currently free blocks.</summary>
    public int FreeBlocks
    {
        get { lock (_lock) return _freeCount; }
    }

    /// <summary>Number of transformer layers.</summary>
    public int NumLayers => _numLayers;

    /// <summary>KV stride (numKvHeads * headDim).</summary>
    public int KvStride => _kvStride;

    /// <summary>
    /// Creates a new block pool with pre-allocated storage for all blocks across all layers.
    /// </summary>
    /// <param name="numLayers">Number of transformer layers.</param>
    /// <param name="numKvHeads">Number of KV attention heads per layer.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="blockSize">Number of tokens per block.</param>
    /// <param name="totalBlocks">Total number of blocks in the pool.</param>
    public KvBlockPool(int numLayers, int numKvHeads, int headDim,
                       int blockSize = 16, int totalBlocks = 4096)
    {
        _blockSize = blockSize;
        _numLayers = numLayers;
        _kvStride = numKvHeads * headDim;
        _totalBlocks = totalBlocks;
        _blockFloats = blockSize * _kvStride;

        _keyBuffers = new nint[numLayers];
        _valueBuffers = new nint[numLayers];

        nuint layerBytes = (nuint)((long)totalBlocks * _blockFloats * sizeof(float));
        for (int i = 0; i < numLayers; i++)
        {
            _keyBuffers[i] = (nint)NativeMemory.AlignedAlloc(layerBytes, 64);
            _valueBuffers[i] = (nint)NativeMemory.AlignedAlloc(layerBytes, 64);
        }

        // Initialize free stack (all blocks free, LIFO order)
        _freeStack = new int[totalBlocks];
        for (int i = 0; i < totalBlocks; i++)
            _freeStack[i] = totalBlocks - 1 - i; // top of stack = block 0
        _freeCount = totalBlocks;

        _refCounts = new int[totalBlocks];
    }

    /// <summary>
    /// Allocates a block from the free pool. Sets ref count to 1.
    /// </summary>
    /// <returns>Block ID.</returns>
    /// <exception cref="InvalidOperationException">No free blocks available.</exception>
    public int Allocate()
    {
        lock (_lock)
        {
            if (_freeCount == 0)
                throw new InvalidOperationException("KvBlockPool exhausted: no free blocks available.");

            int blockId = _freeStack[--_freeCount];
            _refCounts[blockId] = 1;
            return blockId;
        }
    }

    /// <summary>
    /// Increments the reference count for a block (used for shared prefix / beam search).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddRef(int blockId) => Interlocked.Increment(ref _refCounts[blockId]);

    /// <summary>
    /// Gets the current reference count for a block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int RefCount(int blockId) => Volatile.Read(ref _refCounts[blockId]);

    /// <summary>
    /// Decrements the reference count. When it reaches 0, returns the block to the free pool.
    /// </summary>
    public void Release(int blockId)
    {
        int newCount = Interlocked.Decrement(ref _refCounts[blockId]);
        if (newCount == 0)
        {
            lock (_lock)
            {
                _freeStack[_freeCount++] = blockId;
            }
        }
    }

    /// <summary>
    /// Allocates a new block and copies all layer data from the source block (copy-on-write).
    /// </summary>
    /// <returns>New block ID with independent data.</returns>
    public int CopyBlock(int sourceBlockId)
    {
        int newBlockId = Allocate();

        long blockBytes = (long)_blockFloats * sizeof(float);
        for (int layer = 0; layer < _numLayers; layer++)
        {
            Buffer.MemoryCopy(
                (void*)(_keyBuffers[layer] + sourceBlockId * blockBytes),
                (void*)(_keyBuffers[layer] + newBlockId * blockBytes),
                blockBytes, blockBytes);

            Buffer.MemoryCopy(
                (void*)(_valueBuffers[layer] + sourceBlockId * blockBytes),
                (void*)(_valueBuffers[layer] + newBlockId * blockBytes),
                blockBytes, blockBytes);
        }

        return newBlockId;
    }

    /// <summary>
    /// Gets a pointer to the start of key data for a specific block and layer.
    /// Points to <c>blockSize * kvStride</c> floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetKeyPtr(int blockId, int layerIndex)
        => (float*)_keyBuffers[layerIndex] + (long)blockId * _blockFloats;

    /// <summary>
    /// Gets a pointer to the start of value data for a specific block and layer.
    /// Points to <c>blockSize * kvStride</c> floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetValuePtr(int blockId, int layerIndex)
        => (float*)_valueBuffers[layerIndex] + (long)blockId * _blockFloats;

    /// <summary>
    /// Gets a span over key data for a specific block, layer, and token offset within the block.
    /// Returns <see cref="KvStride"/> floats for one token position.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetKeySpan(int blockId, int layerIndex, int offsetInBlock)
        => new(GetKeyPtr(blockId, layerIndex) + offsetInBlock * _kvStride, _kvStride);

    /// <summary>
    /// Gets a span over value data for a specific block, layer, and token offset within the block.
    /// Returns <see cref="KvStride"/> floats for one token position.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetValueSpan(int blockId, int layerIndex, int offsetInBlock)
        => new(GetValuePtr(blockId, layerIndex) + offsetInBlock * _kvStride, _kvStride);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        for (int i = 0; i < _numLayers; i++)
        {
            if (_keyBuffers[i] != 0)
            {
                NativeMemory.AlignedFree((void*)_keyBuffers[i]);
                _keyBuffers[i] = 0;
            }
            if (_valueBuffers[i] != 0)
            {
                NativeMemory.AlignedFree((void*)_valueBuffers[i]);
                _valueBuffers[i] = 0;
            }
        }

        GC.SuppressFinalize(this);
    }

    /// <summary>Releases unmanaged buffers if <see cref="Dispose()"/> was not called.</summary>
    ~KvBlockPool() => Dispose();
}
