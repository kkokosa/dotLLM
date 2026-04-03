namespace DotLLM.Engine.KvCache;

/// <summary>
/// Per-sequence mapping from logical token positions to physical block IDs in a <see cref="KvBlockPool"/>.
/// Grows dynamically as tokens are appended. Supports fork (shared prefix) and copy-on-write.
/// </summary>
public sealed class KvBlockTable
{
    private readonly KvBlockPool _pool;
    private readonly int _blockSize;
    private readonly List<int> _blockIds = []; // blockIds[i] = physical block for logical block i
    private int _currentLength;

    /// <summary>Number of tokens currently stored.</summary>
    public int CurrentLength => _currentLength;

    /// <summary>Number of blocks currently allocated for this sequence.</summary>
    public int BlockCount => _blockIds.Count;

    /// <summary>
    /// Creates a new block table backed by the given pool.
    /// </summary>
    public KvBlockTable(KvBlockPool pool)
    {
        _pool = pool;
        _blockSize = pool.BlockSize;
    }

    /// <summary>
    /// Ensures enough blocks are allocated to hold up to <paramref name="newLength"/> tokens.
    /// Allocates new blocks from the pool as needed.
    /// </summary>
    public void EnsureCapacity(int newLength)
    {
        int blocksNeeded = (newLength + _blockSize - 1) / _blockSize;
        while (_blockIds.Count < blocksNeeded)
            _blockIds.Add(_pool.Allocate());
    }

    /// <summary>
    /// Records that tokens have been appended up to position <paramref name="newLength"/> - 1.
    /// Allocates new blocks if the current capacity is insufficient.
    /// </summary>
    public void Advance(int newLength)
    {
        EnsureCapacity(newLength);
        _currentLength = newLength;
    }

    /// <summary>
    /// Resolves a logical position to its physical block ID and offset within the block.
    /// </summary>
    /// <param name="position">Logical token position (0-based).</param>
    /// <returns>Tuple of (blockId, offsetInBlock).</returns>
    public (int blockId, int offset) Resolve(int position)
    {
        int logicalBlock = position / _blockSize;
        int offset = position % _blockSize;
        return (_blockIds[logicalBlock], offset);
    }

    /// <summary>
    /// Shares all current blocks with <paramref name="target"/> by incrementing ref counts.
    /// Used for beam search branching or prefix sharing.
    /// </summary>
    public void Fork(KvBlockTable target)
    {
        foreach (int blockId in _blockIds)
        {
            _pool.AddRef(blockId);
            target._blockIds.Add(blockId);
        }
        target._currentLength = _currentLength;
    }

    /// <summary>
    /// Ensures the block containing <paramref name="position"/> is writable (ref count = 1).
    /// If the block is shared (ref count > 1), copies it (copy-on-write).
    /// </summary>
    public void EnsureWritable(int position)
    {
        int logicalBlock = position / _blockSize;
        int blockId = _blockIds[logicalBlock];
        if (_pool.RefCount(blockId) > 1)
        {
            int newBlockId = _pool.CopyBlock(blockId);
            _pool.Release(blockId);
            _blockIds[logicalBlock] = newBlockId;
        }
    }

    /// <summary>
    /// Releases all blocks back to the pool and resets the table.
    /// </summary>
    public void Free()
    {
        foreach (int blockId in _blockIds)
            _pool.Release(blockId);
        _blockIds.Clear();
        _currentLength = 0;
    }

    /// <summary>
    /// Truncates the visible length to the given position.
    /// Does NOT free excess blocks — they remain allocated for potential reuse.
    /// </summary>
    internal void SetCurrentLength(int length) => _currentLength = length;
}
