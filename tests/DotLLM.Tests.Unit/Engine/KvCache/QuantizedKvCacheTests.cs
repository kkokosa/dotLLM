using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

public sealed unsafe class QuantizedKvCacheTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 32; // Must be multiple of 32 (quant block size)
    private const int MaxSeqLen = 16;
    private const int KvStride = NumKvHeads * HeadDim; // 64

    [Fact]
    public void Constructor_AllocatesCorrectly_Q8_0()
    {
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);

        Assert.Equal(0, cache.CurrentLength);
        Assert.Equal(MaxSeqLen, cache.MaxLength);
        Assert.Equal(KvCacheDType.Q8_0, cache.KeyDType);
        Assert.Equal(KvCacheDType.Q8_0, cache.ValueDType);
        Assert.True(cache.AllocatedBytes > 0);
    }

    [Fact]
    public void Update_PureQuantized_QuantizesImmediately()
    {
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);

        nint kPtr = AllocAndFill(1, 1.5f);
        nint vPtr = AllocAndFill(1, 2.5f);
        try
        {
            var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);

            cache.Update(kRef, vRef, [0], layerIndex: 0);

            Assert.Equal(1, cache.CurrentLength);
            Assert.Equal(1, cache.QuantizedLength);
            Assert.Equal(0, cache.WindowLength);

            // Quantized data pointer should be non-zero
            Assert.NotEqual((nint)0, cache.GetQuantizedKeysPtr(0));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    [Fact]
    public void Update_WithWindow_EvictsOnlyWhenWindowFull()
    {
        int windowSize = 4;
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize);

        // Add tokens 0..3 — should all be in window, no eviction
        for (int i = 0; i < windowSize; i++)
        {
            nint kPtr = AllocAndFill(1, 1.0f + i);
            nint vPtr = AllocAndFill(1, 10.0f + i);
            try
            {
                var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
                var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);
                cache.Update(kRef, vRef, [i], layerIndex: 0);
            }
            finally
            {
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }
        }

        Assert.Equal(windowSize, cache.CurrentLength);
        Assert.Equal(0, cache.QuantizedLength);
        Assert.Equal(windowSize, cache.WindowLength);

        // Add token 4 — should evict token 0 to quantized buffer
        nint kPtr5 = AllocAndFill(1, 5.0f);
        nint vPtr5 = AllocAndFill(1, 15.0f);
        try
        {
            var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr5);
            var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr5);
            cache.Update(kRef, vRef, [windowSize], layerIndex: 0);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr5);
            NativeMemory.AlignedFree((void*)vPtr5);
        }

        Assert.Equal(windowSize + 1, cache.CurrentLength);
        Assert.Equal(1, cache.QuantizedLength);
        Assert.Equal(windowSize, cache.WindowLength);
    }

    [Fact]
    public void Update_SeparateKeyValueTypes_Works()
    {
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q4_0, windowSize: 0);

        Assert.Equal(KvCacheDType.Q8_0, cache.KeyDType);
        Assert.Equal(KvCacheDType.Q4_0, cache.ValueDType);

        // Key rows should be larger than value rows
        Assert.True(cache.KeyQuantizedRowBytes > cache.ValueQuantizedRowBytes);

        nint kPtr = AllocAndFill(1, 1.0f);
        nint vPtr = AllocAndFill(1, 2.0f);
        try
        {
            var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);
            cache.Update(kRef, vRef, [0], layerIndex: 0);

            Assert.Equal(1, cache.CurrentLength);
            Assert.Equal(1, cache.QuantizedLength);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }
    }

    [Fact]
    public void AllocatedBytes_ReflectsCompression()
    {
        using var fpCache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);
        using var q8Cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);

        // Quantized should use significantly less memory
        Assert.True(q8Cache.AllocatedBytes < fpCache.AllocatedBytes,
            $"Q8_0 cache ({q8Cache.AllocatedBytes}) should be smaller than FP32 cache ({fpCache.AllocatedBytes})");
    }

    [Fact]
    public void Dispose_IsSafeToCallMultipleTimes()
    {
        var cache = new QuantizedKvCache(1, NumKvHeads, HeadDim, 8,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);
        cache.Dispose();
        cache.Dispose(); // Should not throw
    }

    [Fact]
    public void IQuantizedKvCache_InterfaceMethods_Work()
    {
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q4_0, windowSize: 4);

        IQuantizedKvCache qkv = cache;

        Assert.Equal(KvCacheDType.Q8_0, qkv.KeyDType);
        Assert.Equal(KvCacheDType.Q4_0, qkv.ValueDType);
        Assert.Equal(0, qkv.QuantizedLength);
        Assert.Equal(0, qkv.WindowLength);
    }

    [Fact]
    public void PrefillThenDecode_Q8_0_TracksLengthsCorrectly()
    {
        int windowSize = 4;
        using var cache = new QuantizedKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen,
            KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize);

        // Prefill: 3 tokens at once
        nint kPtr = AllocAndFill(3, 1.0f);
        nint vPtr = AllocAndFill(3, 2.0f);
        try
        {
            var kRef = new TensorRef(3, KvStride, DType.Float32, -1, kPtr);
            var vRef = new TensorRef(3, KvStride, DType.Float32, -1, vPtr);
            cache.Update(kRef, vRef, [0, 1, 2], layerIndex: 0);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kPtr);
            NativeMemory.AlignedFree((void*)vPtr);
        }

        Assert.Equal(3, cache.CurrentLength);
        Assert.Equal(0, cache.QuantizedLength); // all within window
        Assert.Equal(3, cache.WindowLength);

        // Decode: add tokens one by one until we trigger eviction
        for (int i = 3; i < 8; i++)
        {
            kPtr = AllocAndFill(1, 1.0f + i);
            vPtr = AllocAndFill(1, 2.0f + i);
            try
            {
                var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kPtr);
                var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vPtr);
                cache.Update(kRef, vRef, [i], layerIndex: 0);
            }
            finally
            {
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }
        }

        Assert.Equal(8, cache.CurrentLength);
        Assert.Equal(4, cache.QuantizedLength); // tokens 0..3 evicted
        Assert.Equal(4, cache.WindowLength);     // tokens 4..7 in window
    }

    private static nint AllocAndFill(int seqLen, float value)
    {
        nuint bytes = (nuint)(seqLen * KvStride * sizeof(float));
        nint ptr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
        new Span<float>((void*)ptr, seqLen * KvStride).Fill(value);
        return ptr;
    }
}
