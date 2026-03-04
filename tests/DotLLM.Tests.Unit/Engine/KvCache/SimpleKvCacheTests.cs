using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

public sealed unsafe class SimpleKvCacheTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 4;
    private const int HeadDim = 8;
    private const int MaxSeqLen = 16;
    private const int KvStride = NumKvHeads * HeadDim; // 32

    [Fact]
    public void Constructor_InitializesCorrectly()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        Assert.Equal(0, cache.CurrentLength);
        Assert.Equal(MaxSeqLen, cache.MaxLength);
    }

    [Fact]
    public void Update_StoresDataAtPositions()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        // Create K/V tensors for 3 tokens
        int seqLen = 3;
        using var kTensor = AllocAndFill(seqLen, 1.0f);
        using var vTensor = AllocAndFill(seqLen, 2.0f);

        int[] positions = [0, 1, 2];
        cache.Update(kTensor, vTensor, positions, layerIndex: 0);

        // Read back and verify
        using var cachedK = cache.GetKeys(0);
        using var cachedV = cache.GetValues(0);

        Assert.Equal(3, cachedK.Shape[0]);

        var kSpan = new ReadOnlySpan<float>((void*)cachedK.DataPointer, 3 * KvStride);
        var vSpan = new ReadOnlySpan<float>((void*)cachedV.DataPointer, 3 * KvStride);

        for (int i = 0; i < 3 * KvStride; i++)
        {
            Assert.Equal(1.0f, kSpan[i]);
            Assert.Equal(2.0f, vSpan[i]);
        }
    }

    [Fact]
    public void Update_AdvancesCurrentLength()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        using var kTensor = AllocAndFill(5, 1.0f);
        using var vTensor = AllocAndFill(5, 2.0f);

        int[] positions = [0, 1, 2, 3, 4];
        cache.Update(kTensor, vTensor, positions, layerIndex: 0);

        Assert.Equal(5, cache.CurrentLength);
    }

    [Fact]
    public void PrefillThenDecode_MaintainsAllData()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        // Prefill: 5 tokens with value = position index + 1
        for (int pos = 0; pos < 5; pos++)
        {
            float fillVal = pos + 1.0f;
            using var kT = AllocAndFill(1, fillVal);
            using var vT = AllocAndFill(1, fillVal * 10);
            cache.Update(kT, vT, [pos], layerIndex: 0);
        }
        Assert.Equal(5, cache.CurrentLength);

        // Decode: 3 more tokens
        for (int step = 0; step < 3; step++)
        {
            int pos = 5 + step;
            float fillVal = pos + 1.0f;
            using var kT = AllocAndFill(1, fillVal);
            using var vT = AllocAndFill(1, fillVal * 10);
            cache.Update(kT, vT, [pos], layerIndex: 0);
        }
        Assert.Equal(8, cache.CurrentLength);

        // Verify all 8 positions
        using var cachedK = cache.GetKeys(0);
        using var cachedV = cache.GetValues(0);
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
    public void Dispose_IsSafeToCallMultipleTimes()
    {
        var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);
        cache.Dispose();
        cache.Dispose(); // Should not throw
    }

    [Fact]
    public void Update_ThrowsOnExceedMaxLength()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        using var kTensor = AllocAndFill(1, 1.0f);
        using var vTensor = AllocAndFill(1, 2.0f);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            cache.Update(kTensor, vTensor, [MaxSeqLen], layerIndex: 0));
    }

    [Fact]
    public void Update_MultipleLayersAreIndependent()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        using var k0 = AllocAndFill(1, 1.0f);
        using var v0 = AllocAndFill(1, 10.0f);
        using var k1 = AllocAndFill(1, 2.0f);
        using var v1 = AllocAndFill(1, 20.0f);

        cache.Update(k0, v0, [0], layerIndex: 0);
        cache.Update(k1, v1, [0], layerIndex: 1);

        using var cachedK0 = cache.GetKeys(0);
        using var cachedK1 = cache.GetKeys(1);

        float firstK0 = *(float*)cachedK0.DataPointer;
        float firstK1 = *(float*)cachedK1.DataPointer;

        Assert.Equal(1.0f, firstK0);
        Assert.Equal(2.0f, firstK1);
    }

    [Fact]
    public void AllocatedBytes_ReturnsExpectedSize()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        // 2 layers × 2 (K+V) × 16 maxSeqLen × 32 kvStride × 4 bytes = 8192
        long expected = (long)NumLayers * 2 * MaxSeqLen * KvStride * sizeof(float);
        Assert.Equal(expected, cache.AllocatedBytes);
    }

    [Fact]
    public void Update_BatchPrefillStoresAllPositions()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        // Prefill 4 tokens at once
        int seqLen = 4;
        nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)(seqLen * KvStride * sizeof(float)), 64);
        nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)(seqLen * KvStride * sizeof(float)), 64);
        try
        {
            // Fill each token's row with its position index
            for (int t = 0; t < seqLen; t++)
            {
                for (int d = 0; d < KvStride; d++)
                {
                    ((float*)kPtr)[t * KvStride + d] = t + 0.5f;
                    ((float*)vPtr)[t * KvStride + d] = (t + 0.5f) * 100;
                }
            }

            var kView = new TensorView(new TensorShape(seqLen, KvStride), DType.Float32, -1, kPtr);
            var vView = new TensorView(new TensorShape(seqLen, KvStride), DType.Float32, -1, vPtr);

            cache.Update(kView, vView, [0, 1, 2, 3], layerIndex: 0);

            Assert.Equal(4, cache.CurrentLength);

            using var cachedK = cache.GetKeys(0);
            var kSpan = new ReadOnlySpan<float>((void*)cachedK.DataPointer, 4 * KvStride);

            for (int t = 0; t < seqLen; t++)
            {
                Assert.Equal(t + 0.5f, kSpan[t * KvStride]);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    /// <summary>
    /// Allocates a tensor of shape [seqLen, KvStride] filled with a constant value.
    /// </summary>
    private static UnmanagedTensor AllocAndFill(int seqLen, float value)
    {
        var shape = new TensorShape(seqLen, KvStride);
        var tensor = UnmanagedTensor.Allocate(shape, DType.Float32, -1);
        new Span<float>((void*)tensor.DataPointer, seqLen * KvStride).Fill(value);
        return tensor;
    }
}
