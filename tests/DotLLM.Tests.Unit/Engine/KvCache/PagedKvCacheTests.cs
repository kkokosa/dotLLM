using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

public sealed unsafe class PagedKvCacheTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 4;
    private const int HeadDim = 8;
    private const int MaxSeqLen = 16;
    private const int BlockSize = 4;
    private const int TotalBlocks = 32;
    private const int KvStride = NumKvHeads * HeadDim; // 32

    private KvBlockPool CreatePool()
        => new(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

    [Fact]
    public void Constructor_InitializesCorrectly()
    {
        using var pool = CreatePool();
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        Assert.Equal(0, cache.CurrentLength);
        Assert.Equal(MaxSeqLen, cache.MaxLength);
    }

    [Fact]
    public void Update_StoresAndRetrievesData()
    {
        using var pool = CreatePool();
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        int seqLen = 3;
        nint kPtr = AllocAndFillNative(seqLen, 1.0f);
        nint vPtr = AllocAndFillNative(seqLen, 2.0f);
        try
        {
            var kRef = new TensorRef(seqLen, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(seqLen, KvStride, DType.Float32, -1, vPtr);

            cache.Update(kRef, vRef, [0, 1, 2], layerIndex: 0);

            var cachedK = cache.GetKeysRef(0);
            var cachedV = cache.GetValuesRef(0);

            Assert.Equal(3, cachedK.Dim0);

            var kSpan = new ReadOnlySpan<float>((void*)cachedK.DataPointer, 3 * KvStride);
            var vSpan = new ReadOnlySpan<float>((void*)cachedV.DataPointer, 3 * KvStride);

            for (int i = 0; i < 3 * KvStride; i++)
            {
                Assert.Equal(1.0f, kSpan[i]);
                Assert.Equal(2.0f, vSpan[i]);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    [Fact]
    public void Update_AdvancesCurrentLength()
    {
        using var pool = CreatePool();
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        nint kPtr = AllocAndFillNative(5, 1.0f);
        nint vPtr = AllocAndFillNative(5, 2.0f);
        try
        {
            var kRef = new TensorRef(5, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(5, KvStride, DType.Float32, -1, vPtr);

            cache.Update(kRef, vRef, [0, 1, 2, 3, 4], layerIndex: 0);
            Assert.Equal(5, cache.CurrentLength);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    [Fact]
    public void PrefillThenDecode_MaintainsAllData()
    {
        using var pool = CreatePool();
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        // Prefill: 5 tokens with distinct values
        for (int pos = 0; pos < 5; pos++)
        {
            float fillVal = pos + 1.0f;
            nint kPtr = AllocAndFillNative(1, fillVal);
            nint vPtr = AllocAndFillNative(1, fillVal * 10);
            try
            {
                var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
                var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);
                cache.Update(kRef, vRef, [pos], layerIndex: 0);
            }
            finally
            {
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }
        }
        Assert.Equal(5, cache.CurrentLength);

        // Decode: 3 more tokens
        for (int step = 0; step < 3; step++)
        {
            int pos = 5 + step;
            float fillVal = pos + 1.0f;
            nint kPtr = AllocAndFillNative(1, fillVal);
            nint vPtr = AllocAndFillNative(1, fillVal * 10);
            try
            {
                var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
                var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);
                cache.Update(kRef, vRef, [pos], layerIndex: 0);
            }
            finally
            {
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }
        }
        Assert.Equal(8, cache.CurrentLength);

        // Verify all 8 positions via staging buffer
        var cachedK = cache.GetKeysRef(0);
        var cachedV = cache.GetValuesRef(0);
        var kSpan = new ReadOnlySpan<float>((void*)cachedK.DataPointer, 8 * KvStride);
        var vSpan = new ReadOnlySpan<float>((void*)cachedV.DataPointer, 8 * KvStride);

        for (int pos = 0; pos < 8; pos++)
        {
            float expectedK = pos + 1.0f;
            float expectedV = (pos + 1.0f) * 10;
            for (int d = 0; d < KvStride; d++)
            {
                Assert.Equal(expectedK, kSpan[pos * KvStride + d]);
                Assert.Equal(expectedV, vSpan[pos * KvStride + d]);
            }
        }
    }

    [Fact]
    public void MultipleLayersAreIndependent()
    {
        using var pool = CreatePool();
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        nint k0Ptr = AllocAndFillNative(1, 1.0f);
        nint v0Ptr = AllocAndFillNative(1, 10.0f);
        nint k1Ptr = AllocAndFillNative(1, 2.0f);
        nint v1Ptr = AllocAndFillNative(1, 20.0f);
        try
        {
            cache.Update(
                new TensorRef(1, KvStride, DType.Float32, -1, k0Ptr),
                new TensorRef(1, KvStride, DType.Float32, -1, v0Ptr),
                [0], layerIndex: 0);
            cache.Update(
                new TensorRef(1, KvStride, DType.Float32, -1, k1Ptr),
                new TensorRef(1, KvStride, DType.Float32, -1, v1Ptr),
                [0], layerIndex: 1);

            // Note: PagedKvCache uses a shared staging buffer, so GetKeysRef results
            // are only valid until the next GetKeysRef/GetValuesRef call.
            // Read each layer's data immediately before requesting the next.
            var cachedK0 = cache.GetKeysRef(0);
            float layer0Key = *(float*)cachedK0.DataPointer;

            var cachedK1 = cache.GetKeysRef(1);
            float layer1Key = *(float*)cachedK1.DataPointer;

            Assert.Equal(1.0f, layer0Key);
            Assert.Equal(2.0f, layer1Key);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)k0Ptr);
            NativeMemory.AlignedFree((void*)v0Ptr);
            NativeMemory.AlignedFree((void*)k1Ptr);
            NativeMemory.AlignedFree((void*)v1Ptr);
        }
    }

    [Fact]
    public void Update_ThrowsOnExceedMaxLength()
    {
        using var pool = CreatePool();
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        nint kPtr = AllocAndFillNative(1, 1.0f);
        nint vPtr = AllocAndFillNative(1, 2.0f);
        try
        {
            var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                cache.Update(kRef, vRef, [MaxSeqLen], layerIndex: 0));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    /// <summary>
    /// Correctness oracle: PagedKvCache and SimpleKvCache should produce identical
    /// GetKeysRef/GetValuesRef output for the same sequence of updates.
    /// </summary>
    [Fact]
    public void MatchesSimpleKvCache_ForIdenticalOperations()
    {
        using var pool = CreatePool();
        using var paged = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);
        using var simple = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        // Batch prefill: 6 tokens
        int batchLen = 6;
        nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)(batchLen * KvStride * sizeof(float)), 64);
        nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)(batchLen * KvStride * sizeof(float)), 64);
        try
        {
            // Fill with distinct values per position
            for (int t = 0; t < batchLen; t++)
            {
                for (int d = 0; d < KvStride; d++)
                {
                    ((float*)kPtr)[t * KvStride + d] = t * 100 + d;
                    ((float*)vPtr)[t * KvStride + d] = t * 1000 + d;
                }
            }

            int[] positions = [0, 1, 2, 3, 4, 5];
            var kRef = new TensorRef(batchLen, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(batchLen, KvStride, DType.Float32, -1, vPtr);

            for (int layer = 0; layer < NumLayers; layer++)
            {
                paged.Update(kRef, vRef, positions, layer);
                simple.Update(kRef, vRef, positions, layer);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }

        // Decode: 4 more tokens
        for (int step = 0; step < 4; step++)
        {
            int pos = batchLen + step;
            nint kDec = AllocAndFillNative(1, pos * 7.0f);
            nint vDec = AllocAndFillNative(1, pos * 77.0f);
            try
            {
                var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kDec);
                var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vDec);
                for (int layer = 0; layer < NumLayers; layer++)
                {
                    paged.Update(kRef, vRef, [pos], layer);
                    simple.Update(kRef, vRef, [pos], layer);
                }
            }
            finally
            {
                NativeMemory.AlignedFree((void*)kDec);
                NativeMemory.AlignedFree((void*)vDec);
            }
        }

        Assert.Equal(simple.CurrentLength, paged.CurrentLength);

        // Compare GetKeysRef and GetValuesRef for all layers
        int totalLen = simple.CurrentLength;
        for (int layer = 0; layer < NumLayers; layer++)
        {
            var simpleK = simple.GetKeysRef(layer);
            var pagedK = paged.GetKeysRef(layer);
            var simpleV = simple.GetValuesRef(layer);
            var pagedV = paged.GetValuesRef(layer);

            Assert.Equal(simpleK.Dim0, pagedK.Dim0);

            var sK = new ReadOnlySpan<float>((void*)simpleK.DataPointer, totalLen * KvStride);
            var pK = new ReadOnlySpan<float>((void*)pagedK.DataPointer, totalLen * KvStride);
            var sV = new ReadOnlySpan<float>((void*)simpleV.DataPointer, totalLen * KvStride);
            var pV = new ReadOnlySpan<float>((void*)pagedV.DataPointer, totalLen * KvStride);

            for (int i = 0; i < totalLen * KvStride; i++)
            {
                Assert.Equal(sK[i], pK[i]);
                Assert.Equal(sV[i], pV[i]);
            }
        }
    }

    [Fact]
    public void SetCurrentLength_TruncatesAndAllowsOverwrite()
    {
        using var pool = CreatePool();
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        // Write 8 positions
        for (int pos = 0; pos < 8; pos++)
        {
            nint kPtr = AllocAndFillNative(1, pos + 1.0f);
            nint vPtr = AllocAndFillNative(1, (pos + 1.0f) * 10);
            try
            {
                cache.Update(
                    new TensorRef(1, KvStride, DType.Float32, -1, kPtr),
                    new TensorRef(1, KvStride, DType.Float32, -1, vPtr),
                    [pos], layerIndex: 0);
            }
            finally
            {
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }
        }
        Assert.Equal(8, cache.CurrentLength);

        cache.SetCurrentLength(3);
        Assert.Equal(3, cache.CurrentLength);

        // GetKeysRef should return only 3 positions
        var keysRef = cache.GetKeysRef(0);
        Assert.Equal(3, keysRef.Dim0);
    }

    [Fact]
    public void Dispose_ReturnsBlocksToPool()
    {
        var pool = CreatePool();
        var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        // Add some data to allocate blocks
        nint kPtr = AllocAndFillNative(5, 1.0f);
        nint vPtr = AllocAndFillNative(5, 2.0f);
        try
        {
            cache.Update(
                new TensorRef(5, KvStride, DType.Float32, -1, kPtr),
                new TensorRef(5, KvStride, DType.Float32, -1, vPtr),
                [0, 1, 2, 3, 4], layerIndex: 0);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }

        int freeAfterAlloc = pool.FreeBlocks;
        cache.Dispose();

        Assert.True(pool.FreeBlocks > freeAfterAlloc);
        pool.Dispose();
    }

    [Fact]
    public void ITensorPath_DelegatesToTensorRef()
    {
        using var pool = CreatePool();
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        using var kTensor = AllocAndFill(2, 5.0f);
        using var vTensor = AllocAndFill(2, 6.0f);

        cache.Update(kTensor, vTensor, [0, 1], layerIndex: 0);

        Assert.Equal(2, cache.CurrentLength);

        using var cachedK = cache.GetKeys(0);
        Assert.Equal(2, cachedK.Shape[0]);
        Assert.Equal(5.0f, *(float*)cachedK.DataPointer);
    }

    [Fact]
    public void Factory_CreatesWorkingCaches()
    {
        using var factory = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize);

        using var cache1 = factory.Create(MaxSeqLen);
        using var cache2 = factory.Create(MaxSeqLen);

        nint kPtr = AllocAndFillNative(1, 42.0f);
        nint vPtr = AllocAndFillNative(1, 43.0f);
        try
        {
            cache1.Update(
                new TensorRef(1, KvStride, DType.Float32, -1, kPtr),
                new TensorRef(1, KvStride, DType.Float32, -1, vPtr),
                [0], layerIndex: 0);
            cache2.Update(
                new TensorRef(1, KvStride, DType.Float32, -1, kPtr),
                new TensorRef(1, KvStride, DType.Float32, -1, vPtr),
                [0], layerIndex: 0);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }

        Assert.Equal(1, cache1.CurrentLength);
        Assert.Equal(1, cache2.CurrentLength);

        // Both share the same pool (via factory)
        var k1 = cache1.GetKeysRef(0);
        var k2 = cache2.GetKeysRef(0);
        Assert.Equal(42.0f, *(float*)k1.DataPointer);
        Assert.Equal(42.0f, *(float*)k2.DataPointer);
    }

    /// <summary>
    /// Allocates a native buffer of shape [seqLen, KvStride] filled with a constant value.
    /// </summary>
    private static nint AllocAndFillNative(int seqLen, float value)
    {
        nuint bytes = (nuint)(seqLen * KvStride * sizeof(float));
        nint ptr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
        new Span<float>((void*)ptr, seqLen * KvStride).Fill(value);
        return ptr;
    }

    /// <summary>
    /// Allocates an <see cref="UnmanagedTensor"/> of shape [seqLen, KvStride] filled with a constant value.
    /// </summary>
    private static UnmanagedTensor AllocAndFill(int seqLen, float value)
    {
        var shape = new TensorShape(seqLen, KvStride);
        var tensor = UnmanagedTensor.Allocate(shape, DType.Float32, -1);
        new Span<float>((void*)tensor.DataPointer, seqLen * KvStride).Fill(value);
        return tensor;
    }
}
