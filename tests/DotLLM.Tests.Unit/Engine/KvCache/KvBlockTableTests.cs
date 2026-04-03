using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

public sealed class KvBlockTableTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 4;
    private const int HeadDim = 8;
    private const int BlockSize = 4;
    private const int TotalBlocks = 16;

    private KvBlockPool CreatePool()
        => new(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

    [Fact]
    public void InitialState_EmptyTable()
    {
        using var pool = CreatePool();
        var table = new KvBlockTable(pool);

        Assert.Equal(0, table.CurrentLength);
        Assert.Equal(0, table.BlockCount);
    }

    [Fact]
    public void Advance_AllocatesBlocksAsNeeded()
    {
        using var pool = CreatePool();
        var table = new KvBlockTable(pool);

        table.Advance(3);  // 3 tokens, fits in 1 block (blockSize=4)
        Assert.Equal(3, table.CurrentLength);
        Assert.Equal(1, table.BlockCount);

        table.Advance(5);  // crosses into 2nd block
        Assert.Equal(5, table.CurrentLength);
        Assert.Equal(2, table.BlockCount);
    }

    [Fact]
    public void Resolve_ReturnsCorrectBlockAndOffset()
    {
        using var pool = CreatePool();
        var table = new KvBlockTable(pool);

        table.Advance(10);  // 3 blocks needed (4+4+2)

        // Position 0 → block 0, offset 0
        var (block0, offset0) = table.Resolve(0);
        Assert.Equal(0, offset0);

        // Position 3 → block 0, offset 3
        var (block0b, offset3) = table.Resolve(3);
        Assert.Equal(block0, block0b);
        Assert.Equal(3, offset3);

        // Position 4 → block 1, offset 0
        var (block1, offset4) = table.Resolve(4);
        Assert.NotEqual(block0, block1);
        Assert.Equal(0, offset4);

        // Position 9 → block 2, offset 1
        var (_, offset9) = table.Resolve(9);
        Assert.Equal(1, offset9);
    }

    [Fact]
    public void EnsureCapacity_DoesNotAllocateExtraBlocks()
    {
        using var pool = CreatePool();
        var table = new KvBlockTable(pool);

        table.EnsureCapacity(4);  // exactly 1 block
        Assert.Equal(1, table.BlockCount);

        table.EnsureCapacity(4);  // same capacity, no new blocks
        Assert.Equal(1, table.BlockCount);

        table.EnsureCapacity(5);  // needs 2 blocks now
        Assert.Equal(2, table.BlockCount);
    }

    [Fact]
    public void Fork_SharesBlocksViaRefCount()
    {
        using var pool = CreatePool();
        var source = new KvBlockTable(pool);
        var target = new KvBlockTable(pool);

        source.Advance(6);  // 2 blocks
        Assert.Equal(TotalBlocks - 2, pool.FreeBlocks);

        source.Fork(target);

        // Both tables have same length and blocks
        Assert.Equal(6, target.CurrentLength);
        Assert.Equal(2, target.BlockCount);

        // Blocks are shared (ref count = 2), no new blocks allocated
        Assert.Equal(TotalBlocks - 2, pool.FreeBlocks);

        var (srcBlock0, _) = source.Resolve(0);
        var (tgtBlock0, _) = target.Resolve(0);
        Assert.Equal(srcBlock0, tgtBlock0);

        Assert.Equal(2, pool.RefCount(srcBlock0));
    }

    [Fact]
    public void EnsureWritable_CopiesSharedBlock()
    {
        using var pool = CreatePool();
        var source = new KvBlockTable(pool);
        var target = new KvBlockTable(pool);

        source.Advance(4);  // 1 block

        var (srcBlockBefore, _) = source.Resolve(0);
        Assert.Equal(1, pool.RefCount(srcBlockBefore));

        source.Fork(target);
        Assert.Equal(2, pool.RefCount(srcBlockBefore));

        // Now make target writable at position 0 — should trigger CoW
        target.EnsureWritable(0);

        var (tgtBlockAfter, _) = target.Resolve(0);
        Assert.NotEqual(srcBlockBefore, tgtBlockAfter);  // New block
        Assert.Equal(1, pool.RefCount(srcBlockBefore));   // Source refcount back to 1
        Assert.Equal(1, pool.RefCount(tgtBlockAfter));    // New block has refcount 1
    }

    [Fact]
    public void EnsureWritable_NoOpWhenNotShared()
    {
        using var pool = CreatePool();
        var table = new KvBlockTable(pool);

        table.Advance(4);
        var (blockBefore, _) = table.Resolve(0);

        table.EnsureWritable(0);  // Already refcount=1, should be a no-op

        var (blockAfter, _) = table.Resolve(0);
        Assert.Equal(blockBefore, blockAfter);
    }

    [Fact]
    public void Free_ReleasesAllBlocksToPool()
    {
        using var pool = CreatePool();
        var table = new KvBlockTable(pool);

        table.Advance(10);  // 3 blocks
        Assert.Equal(TotalBlocks - 3, pool.FreeBlocks);

        table.Free();

        Assert.Equal(0, table.CurrentLength);
        Assert.Equal(0, table.BlockCount);
        Assert.Equal(TotalBlocks, pool.FreeBlocks);
    }

    [Fact]
    public void SetCurrentLength_TruncatesVisibleLength()
    {
        using var pool = CreatePool();
        var table = new KvBlockTable(pool);

        table.Advance(10);
        Assert.Equal(10, table.CurrentLength);

        table.SetCurrentLength(5);
        Assert.Equal(5, table.CurrentLength);
        Assert.Equal(3, table.BlockCount);  // Blocks not freed — just length truncated
    }
}
