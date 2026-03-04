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
    public void Update_TensorRef_StoresDataAtPositions()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

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
    public void Update_TensorRef_AdvancesCurrentLength()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

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
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        // Prefill: 5 tokens with value = position index + 1
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

        // Verify all 8 positions
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

    [Fact]
    public void Update_MultipleLayersAreIndependent()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

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

            var cachedK0 = cache.GetKeysRef(0);
            var cachedK1 = cache.GetKeysRef(1);

            Assert.Equal(1.0f, *(float*)cachedK0.DataPointer);
            Assert.Equal(2.0f, *(float*)cachedK1.DataPointer);
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

        int seqLen = 4;
        nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)(seqLen * KvStride * sizeof(float)), 64);
        nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)(seqLen * KvStride * sizeof(float)), 64);
        try
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int d = 0; d < KvStride; d++)
                {
                    ((float*)kPtr)[t * KvStride + d] = t + 0.5f;
                    ((float*)vPtr)[t * KvStride + d] = (t + 0.5f) * 100;
                }
            }

            var kRef = new TensorRef(seqLen, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(seqLen, KvStride, DType.Float32, -1, vPtr);

            cache.Update(kRef, vRef, [0, 1, 2, 3], layerIndex: 0);

            Assert.Equal(4, cache.CurrentLength);

            var cachedK = cache.GetKeysRef(0);
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
    /// Verifies that <see cref="IKvCache.Update(Core.Tensors.ITensor,Core.Tensors.ITensor,ReadOnlySpan{int},int)"/>
    /// correctly delegates to the <see cref="TensorRef"/>-based implementation.
    /// </summary>
    [Fact]
    public void Update_ITensorPath_DelegatesToTensorRef()
    {
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        using var kTensor = AllocAndFill(2, 5.0f);
        using var vTensor = AllocAndFill(2, 6.0f);

        cache.Update(kTensor, vTensor, [0, 1], layerIndex: 0);

        Assert.Equal(2, cache.CurrentLength);

        // Verify via ITensor path
        using var cachedK = cache.GetKeys(0);
        Assert.Equal(2, cachedK.Shape[0]);
        Assert.Equal(5.0f, *(float*)cachedK.DataPointer);
    }

    [Fact]
    public void Finalizer_FreesMemory_WhenDisposeNotCalled()
    {
        // We can't directly test the finalizer frees memory without tooling,
        // but we can verify creating and abandoning a cache doesn't crash.
        var cache = new SimpleKvCache(1, 2, 4, 8);
        // Intentionally NOT calling Dispose — finalizer should handle cleanup.
        // ReSharper disable once RedundantAssignment
        cache = null;
        GC.Collect();
        GC.WaitForPendingFinalizers();
        // If we get here without crash, the finalizer worked.
    }

    /// <summary>
    /// Allocates a native buffer of shape [seqLen, KvStride] filled with a constant value.
    /// Caller must free via <see cref="NativeMemory.AlignedFree"/>.
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
