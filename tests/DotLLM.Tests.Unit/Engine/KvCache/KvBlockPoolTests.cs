using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

public sealed unsafe class KvBlockPoolTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 4;
    private const int HeadDim = 8;
    private const int BlockSize = 4;
    private const int TotalBlocks = 8;
    private const int KvStride = NumKvHeads * HeadDim; // 32

    [Fact]
    public void Constructor_AllBlocksFree()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        Assert.Equal(BlockSize, pool.BlockSize);
        Assert.Equal(TotalBlocks, pool.TotalBlocks);
        Assert.Equal(TotalBlocks, pool.FreeBlocks);
    }

    [Fact]
    public void Allocate_ReturnsBlockWithRefCount1()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        int blockId = pool.Allocate();

        Assert.Equal(1, pool.RefCount(blockId));
        Assert.Equal(TotalBlocks - 1, pool.FreeBlocks);
    }

    [Fact]
    public void Allocate_ReturnsDistinctBlocks()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        var ids = new HashSet<int>();
        for (int i = 0; i < TotalBlocks; i++)
            ids.Add(pool.Allocate());

        Assert.Equal(TotalBlocks, ids.Count);
        Assert.Equal(0, pool.FreeBlocks);
    }

    [Fact]
    public void Allocate_ThrowsWhenExhausted()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        for (int i = 0; i < TotalBlocks; i++)
            pool.Allocate();

        Assert.Throws<InvalidOperationException>(() => pool.Allocate());
    }

    [Fact]
    public void Release_ReturnsBlockToPool()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        int blockId = pool.Allocate();
        Assert.Equal(TotalBlocks - 1, pool.FreeBlocks);

        pool.Release(blockId);
        Assert.Equal(TotalBlocks, pool.FreeBlocks);
        Assert.Equal(0, pool.RefCount(blockId));
    }

    [Fact]
    public void AddRef_IncrementsRefCount()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        int blockId = pool.Allocate();
        pool.AddRef(blockId);

        Assert.Equal(2, pool.RefCount(blockId));
    }

    [Fact]
    public void Release_DoesNotFreeUntilRefCountZero()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        int blockId = pool.Allocate();
        pool.AddRef(blockId);    // refCount = 2
        pool.Release(blockId);   // refCount = 1

        Assert.Equal(TotalBlocks - 1, pool.FreeBlocks);
        Assert.Equal(1, pool.RefCount(blockId));

        pool.Release(blockId);   // refCount = 0, returned to pool
        Assert.Equal(TotalBlocks, pool.FreeBlocks);
    }

    [Fact]
    public void GetKeyPtr_WritesAndReadsCorrectly()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        int blockId = pool.Allocate();

        // Write to slot 0, layer 0
        float* kPtr = pool.GetKeyPtr(blockId, layerIndex: 0);
        for (int i = 0; i < KvStride; i++)
            kPtr[i] = i + 1.0f;

        // Read back
        float* kRead = pool.GetKeyPtr(blockId, layerIndex: 0);
        for (int i = 0; i < KvStride; i++)
            Assert.Equal(i + 1.0f, kRead[i]);
    }

    [Fact]
    public void GetValuePtr_WritesAndReadsCorrectly()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        int blockId = pool.Allocate();

        float* vPtr = pool.GetValuePtr(blockId, layerIndex: 1);
        for (int i = 0; i < KvStride; i++)
            vPtr[i] = i * 0.5f;

        float* vRead = pool.GetValuePtr(blockId, layerIndex: 1);
        for (int i = 0; i < KvStride; i++)
            Assert.Equal(i * 0.5f, vRead[i]);
    }

    [Fact]
    public void LayersAreIndependent()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        int blockId = pool.Allocate();

        float* k0 = pool.GetKeyPtr(blockId, 0);
        float* k1 = pool.GetKeyPtr(blockId, 1);

        k0[0] = 42.0f;
        k1[0] = 99.0f;

        Assert.Equal(42.0f, pool.GetKeyPtr(blockId, 0)[0]);
        Assert.Equal(99.0f, pool.GetKeyPtr(blockId, 1)[0]);
    }

    [Fact]
    public void CopyBlock_CreatesIndependentCopy()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        int srcId = pool.Allocate();

        // Fill source with data
        for (int layer = 0; layer < NumLayers; layer++)
        {
            float* kPtr = pool.GetKeyPtr(srcId, layer);
            float* vPtr = pool.GetValuePtr(srcId, layer);
            int blockFloats = BlockSize * KvStride;
            for (int i = 0; i < blockFloats; i++)
            {
                kPtr[i] = layer * 100 + i;
                vPtr[i] = layer * 1000 + i;
            }
        }

        // Copy
        int copyId = pool.CopyBlock(srcId);
        Assert.NotEqual(srcId, copyId);

        // Verify data matches
        for (int layer = 0; layer < NumLayers; layer++)
        {
            float* srcK = pool.GetKeyPtr(srcId, layer);
            float* copyK = pool.GetKeyPtr(copyId, layer);
            float* srcV = pool.GetValuePtr(srcId, layer);
            float* copyV = pool.GetValuePtr(copyId, layer);

            int blockFloats = BlockSize * KvStride;
            for (int i = 0; i < blockFloats; i++)
            {
                Assert.Equal(srcK[i], copyK[i]);
                Assert.Equal(srcV[i], copyV[i]);
            }
        }

        // Modify copy — should not affect source
        pool.GetKeyPtr(copyId, 0)[0] = -1.0f;
        Assert.NotEqual(-1.0f, pool.GetKeyPtr(srcId, 0)[0]);
    }

    [Fact]
    public void GetKeySpan_ReturnsCorrectSlice()
    {
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

        int blockId = pool.Allocate();

        // Write to offset 2 in the block
        var span = pool.GetKeySpan(blockId, layerIndex: 0, offsetInBlock: 2);
        Assert.Equal(KvStride, span.Length);
        span.Fill(7.0f);

        // Read back via raw pointer
        float* kPtr = pool.GetKeyPtr(blockId, 0);
        Assert.Equal(7.0f, kPtr[2 * KvStride]); // offset 2
        Assert.Equal(7.0f, kPtr[2 * KvStride + KvStride - 1]);
    }

    [Fact]
    public void Dispose_IsSafeToCallMultipleTimes()
    {
        var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);
        pool.Dispose();
        pool.Dispose(); // Should not throw
    }
}
