using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// Coverage for <see cref="IKvCache.TryReserveSlot"/> + <see cref="IKvCache.CommitSlot"/>.
/// The primitive lets transformer K/V projections write directly into the cache slot,
/// skipping the scratch + <c>Update</c> memcpy. The contract is that the resulting
/// cache state must be byte-identical to the legacy <c>Update</c> path.
/// </summary>
public sealed unsafe class ReserveSlotTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 4;
    private const int HeadDim = 8;
    private const int KvStride = NumKvHeads * HeadDim; // 32

    // ── SimpleKvCache ───────────────────────────────────────────────────

    [Fact]
    public void Simple_TryReserveSlot_Contiguous_ReturnsTrueAndExposesInPlaceSlot()
    {
        const int MaxSeqLen = 16;
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        Span<int> positions = stackalloc int[] { 0, 1, 2 };
        bool ok = cache.TryReserveSlot(layerIndex: 0, positions, out var kDst, out var vDst);

        Assert.True(ok);
        Assert.Equal(3 * KvStride, kDst.Length);
        Assert.Equal(3 * KvStride, vDst.Length);
    }

    [Fact]
    public void Simple_TryReserveSlot_NonContiguous_ReturnsFalse()
    {
        const int MaxSeqLen = 16;
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        Span<int> positions = stackalloc int[] { 0, 2, 3 };
        bool ok = cache.TryReserveSlot(layerIndex: 0, positions, out var kDst, out var vDst);

        Assert.False(ok);
        Assert.True(kDst.IsEmpty);
        Assert.True(vDst.IsEmpty);
    }

    [Fact]
    public void Simple_TryReserveSlot_OutOfRange_ReturnsFalse()
    {
        const int MaxSeqLen = 16;
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        // Run [15..17) exceeds maxSeqLen=16.
        Span<int> positions = stackalloc int[] { 15, 16, 17 };
        bool ok = cache.TryReserveSlot(layerIndex: 0, positions, out _, out _);

        Assert.False(ok);
    }

    [Fact]
    public void Simple_TryReserveSlot_EmptyPositions_ReturnsFalse()
    {
        const int MaxSeqLen = 16;
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        bool ok = cache.TryReserveSlot(layerIndex: 0, ReadOnlySpan<int>.Empty, out _, out _);
        Assert.False(ok);
    }

    [Fact]
    public void Simple_CommitSlot_AdvancesCurrentLength()
    {
        const int MaxSeqLen = 16;
        using var cache = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        Span<int> positions = stackalloc int[] { 0, 1, 2 };
        Assert.True(cache.TryReserveSlot(0, positions, out _, out _));
        cache.CommitSlot(0, positions);

        Assert.Equal(3, cache.CurrentLength);
    }

    /// <summary>
    /// Bit-exact: building the cache via TryReserveSlot+write+CommitSlot produces
    /// byte-identical buffers to the legacy scratch+Update path.
    /// </summary>
    [Fact]
    public void Simple_ReserveSlot_BitExactWithUpdate_Prefill()
    {
        const int MaxSeqLen = 16;
        const int SeqLen = 6;

        using var cacheUpdate = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);
        using var cacheSlot = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        // Deterministic synthetic K/V.
        nint kSrc = (nint)NativeMemory.AlignedAlloc((nuint)(SeqLen * KvStride * sizeof(float)), 64);
        nint vSrc = (nint)NativeMemory.AlignedAlloc((nuint)(SeqLen * KvStride * sizeof(float)), 64);
        try
        {
            for (int t = 0; t < SeqLen; t++)
            for (int d = 0; d < KvStride; d++)
            {
                ((float*)kSrc)[t * KvStride + d] = MathF.Sin(t * 0.37f + d * 0.013f);
                ((float*)vSrc)[t * KvStride + d] = MathF.Cos(t * 0.41f + d * 0.017f);
            }

            int[] positions = [0, 1, 2, 3, 4, 5];

            // Path A: legacy Update.
            for (int layer = 0; layer < NumLayers; layer++)
            {
                var kRef = new TensorRef(SeqLen, KvStride, DType.Float32, -1, kSrc);
                var vRef = new TensorRef(SeqLen, KvStride, DType.Float32, -1, vSrc);
                cacheUpdate.Update(kRef, vRef, positions, layer);
            }

            // Path B: TryReserveSlot + write + CommitSlot.
            for (int layer = 0; layer < NumLayers; layer++)
            {
                Assert.True(cacheSlot.TryReserveSlot(layer, positions, out var kDst, out var vDst));
                new ReadOnlySpan<float>((void*)kSrc, SeqLen * KvStride).CopyTo(kDst);
                new ReadOnlySpan<float>((void*)vSrc, SeqLen * KvStride).CopyTo(vDst);
                cacheSlot.CommitSlot(layer, positions);
            }

            Assert.Equal(cacheUpdate.CurrentLength, cacheSlot.CurrentLength);

            for (int layer = 0; layer < NumLayers; layer++)
            {
                var kA = cacheUpdate.GetKeysRef(layer);
                var kB = cacheSlot.GetKeysRef(layer);
                var vA = cacheUpdate.GetValuesRef(layer);
                var vB = cacheSlot.GetValuesRef(layer);

                int floats = SeqLen * KvStride;
                AssertBytesEqual(kA.DataPointer, kB.DataPointer, floats);
                AssertBytesEqual(vA.DataPointer, vB.DataPointer, floats);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kSrc);
            NativeMemory.AlignedFree((void*)vSrc);
        }
    }

    /// <summary>
    /// Decode pattern: per-step single-token writes via TryReserveSlot must produce
    /// byte-identical state to the legacy Update path.
    /// </summary>
    [Fact]
    public void Simple_ReserveSlot_BitExactWithUpdate_DecodeSequence()
    {
        const int MaxSeqLen = 16;
        const int Steps = 8;

        using var cacheUpdate = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);
        using var cacheSlot = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

        nint kStep = (nint)NativeMemory.AlignedAlloc((nuint)(KvStride * sizeof(float)), 64);
        nint vStep = (nint)NativeMemory.AlignedAlloc((nuint)(KvStride * sizeof(float)), 64);
        try
        {
            for (int step = 0; step < Steps; step++)
            {
                for (int d = 0; d < KvStride; d++)
                {
                    ((float*)kStep)[d] = MathF.Tan((step + 1) * 0.07f + d * 0.003f);
                    ((float*)vStep)[d] = MathF.Sinh((step + 1) * 0.05f + d * 0.011f);
                }

                int[] positions = [step];

                for (int layer = 0; layer < NumLayers; layer++)
                {
                    var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kStep);
                    var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vStep);
                    cacheUpdate.Update(kRef, vRef, positions, layer);

                    Assert.True(cacheSlot.TryReserveSlot(layer, positions, out var kDst, out var vDst));
                    new ReadOnlySpan<float>((void*)kStep, KvStride).CopyTo(kDst);
                    new ReadOnlySpan<float>((void*)vStep, KvStride).CopyTo(vDst);
                    cacheSlot.CommitSlot(layer, positions);
                }
            }

            Assert.Equal(cacheUpdate.CurrentLength, cacheSlot.CurrentLength);

            for (int layer = 0; layer < NumLayers; layer++)
            {
                var kA = cacheUpdate.GetKeysRef(layer);
                var kB = cacheSlot.GetKeysRef(layer);
                var vA = cacheUpdate.GetValuesRef(layer);
                var vB = cacheSlot.GetValuesRef(layer);
                AssertBytesEqual(kA.DataPointer, kB.DataPointer, Steps * KvStride);
                AssertBytesEqual(vA.DataPointer, vB.DataPointer, Steps * KvStride);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kStep);
            NativeMemory.AlignedFree((void*)vStep);
        }
    }

    // ── PagedKvCache ────────────────────────────────────────────────────

    [Fact]
    public void Paged_TryReserveSlot_SingleBlock_ReturnsTrue()
    {
        const int BlockSize = 4;
        const int TotalBlocks = 8;
        const int MaxSeqLen = 16;
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        // Run fits entirely within block 0 (positions 0..2 of blockSize=4).
        Span<int> positions = stackalloc int[] { 0, 1, 2 };
        bool ok = cache.TryReserveSlot(0, positions, out var kDst, out var vDst);

        Assert.True(ok);
        Assert.Equal(3 * KvStride, kDst.Length);
        Assert.Equal(3 * KvStride, vDst.Length);
    }

    [Fact]
    public void Paged_TryReserveSlot_BlockBoundary_ReturnsFalse()
    {
        const int BlockSize = 4;
        const int TotalBlocks = 8;
        const int MaxSeqLen = 16;
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        // Run [3,4,5] crosses block 0 → block 1.
        Span<int> positions = stackalloc int[] { 3, 4, 5 };
        bool ok = cache.TryReserveSlot(0, positions, out var kDst, out var vDst);

        Assert.False(ok);
        Assert.True(kDst.IsEmpty);
        Assert.True(vDst.IsEmpty);
    }

    [Fact]
    public void Paged_TryReserveSlot_SingleTokenDecode_AlwaysFits()
    {
        const int BlockSize = 4;
        const int TotalBlocks = 8;
        const int MaxSeqLen = 16;
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        // seqLen=1 always fits in any block — every decode position is reservable.
        Span<int> positionBuf = stackalloc int[1];
        for (int p = 0; p < MaxSeqLen; p++)
        {
            positionBuf[0] = p;
            Assert.True(cache.TryReserveSlot(0, positionBuf, out var kDst, out var vDst),
                $"position {p} should be reservable as a single-token slot");
            Assert.Equal(KvStride, kDst.Length);
            Assert.Equal(KvStride, vDst.Length);
        }
    }

    [Fact]
    public void Paged_TryReserveSlot_NonContiguous_ReturnsFalse()
    {
        const int BlockSize = 4;
        const int TotalBlocks = 8;
        const int MaxSeqLen = 16;
        using var pool = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);
        using var cache = new PagedKvCache(pool, NumLayers, KvStride, MaxSeqLen);

        Span<int> positions = stackalloc int[] { 0, 2 };
        bool ok = cache.TryReserveSlot(0, positions, out _, out _);
        Assert.False(ok);
    }

    /// <summary>
    /// Bit-exact: paged decode sequence built via TryReserveSlot must match the legacy
    /// Update path on the data the attention kernel reads through GetKeysRef/GetValuesRef
    /// (the staging buffer).
    /// </summary>
    [Fact]
    public void Paged_ReserveSlot_BitExactWithUpdate_DecodeSequence()
    {
        const int BlockSize = 4;
        const int TotalBlocks = 8;
        const int MaxSeqLen = 16;
        const int Steps = 10;

        using var poolA = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);
        using var poolB = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);
        using var cacheUpdate = new PagedKvCache(poolA, NumLayers, KvStride, MaxSeqLen);
        using var cacheSlot = new PagedKvCache(poolB, NumLayers, KvStride, MaxSeqLen);

        nint kStep = (nint)NativeMemory.AlignedAlloc((nuint)(KvStride * sizeof(float)), 64);
        nint vStep = (nint)NativeMemory.AlignedAlloc((nuint)(KvStride * sizeof(float)), 64);
        try
        {
            for (int step = 0; step < Steps; step++)
            {
                for (int d = 0; d < KvStride; d++)
                {
                    ((float*)kStep)[d] = MathF.Sin((step + 1) * 0.13f + d * 0.007f);
                    ((float*)vStep)[d] = MathF.Cos((step + 1) * 0.11f + d * 0.005f);
                }
                int[] positions = [step];

                for (int layer = 0; layer < NumLayers; layer++)
                {
                    var kRef = new TensorRef(1, KvStride, DType.Float32, -1, kStep);
                    var vRef = new TensorRef(1, KvStride, DType.Float32, -1, vStep);
                    cacheUpdate.Update(kRef, vRef, positions, layer);

                    Assert.True(cacheSlot.TryReserveSlot(layer, positions, out var kDst, out var vDst));
                    new ReadOnlySpan<float>((void*)kStep, KvStride).CopyTo(kDst);
                    new ReadOnlySpan<float>((void*)vStep, KvStride).CopyTo(vDst);
                    cacheSlot.CommitSlot(layer, positions);
                }
            }

            Assert.Equal(cacheUpdate.CurrentLength, cacheSlot.CurrentLength);

            // Compare via the staging-gathered contiguous view (what attention sees).
            for (int layer = 0; layer < NumLayers; layer++)
            {
                var kA = cacheUpdate.GetKeysRef(layer);
                var kB = cacheSlot.GetKeysRef(layer);
                var vA = cacheUpdate.GetValuesRef(layer);
                var vB = cacheSlot.GetValuesRef(layer);
                AssertBytesEqual(kA.DataPointer, kB.DataPointer, Steps * KvStride);
                AssertBytesEqual(vA.DataPointer, vB.DataPointer, Steps * KvStride);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kStep);
            NativeMemory.AlignedFree((void*)vStep);
        }
    }

    /// <summary>
    /// Prefill: a single multi-token reservation that fits in one block produces
    /// byte-identical state to Update.
    /// </summary>
    [Fact]
    public void Paged_ReserveSlot_BitExactWithUpdate_SingleBlockPrefill()
    {
        const int BlockSize = 8;
        const int TotalBlocks = 4;
        const int MaxSeqLen = 16;
        const int SeqLen = 5; // fits in block 0 (size 8)

        using var poolA = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);
        using var poolB = new KvBlockPool(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);
        using var cacheUpdate = new PagedKvCache(poolA, NumLayers, KvStride, MaxSeqLen);
        using var cacheSlot = new PagedKvCache(poolB, NumLayers, KvStride, MaxSeqLen);

        nint kSrc = (nint)NativeMemory.AlignedAlloc((nuint)(SeqLen * KvStride * sizeof(float)), 64);
        nint vSrc = (nint)NativeMemory.AlignedAlloc((nuint)(SeqLen * KvStride * sizeof(float)), 64);
        try
        {
            for (int t = 0; t < SeqLen; t++)
            for (int d = 0; d < KvStride; d++)
            {
                ((float*)kSrc)[t * KvStride + d] = MathF.Sin(t * 0.37f + d * 0.013f);
                ((float*)vSrc)[t * KvStride + d] = MathF.Cos(t * 0.41f + d * 0.017f);
            }

            int[] positions = [0, 1, 2, 3, 4];
            for (int layer = 0; layer < NumLayers; layer++)
            {
                var kRef = new TensorRef(SeqLen, KvStride, DType.Float32, -1, kSrc);
                var vRef = new TensorRef(SeqLen, KvStride, DType.Float32, -1, vSrc);
                cacheUpdate.Update(kRef, vRef, positions, layer);

                Assert.True(cacheSlot.TryReserveSlot(layer, positions, out var kDst, out var vDst));
                new ReadOnlySpan<float>((void*)kSrc, SeqLen * KvStride).CopyTo(kDst);
                new ReadOnlySpan<float>((void*)vSrc, SeqLen * KvStride).CopyTo(vDst);
                cacheSlot.CommitSlot(layer, positions);
            }

            Assert.Equal(cacheUpdate.CurrentLength, cacheSlot.CurrentLength);
            for (int layer = 0; layer < NumLayers; layer++)
            {
                var kA = cacheUpdate.GetKeysRef(layer);
                var kB = cacheSlot.GetKeysRef(layer);
                var vA = cacheUpdate.GetValuesRef(layer);
                var vB = cacheSlot.GetValuesRef(layer);
                AssertBytesEqual(kA.DataPointer, kB.DataPointer, SeqLen * KvStride);
                AssertBytesEqual(vA.DataPointer, vB.DataPointer, SeqLen * KvStride);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)kSrc);
            NativeMemory.AlignedFree((void*)vSrc);
        }
    }

    // ── Caches that opt out (default IKvCache fallback) ────────────────

    [Fact]
    public void Quantized_TryReserveSlot_ReturnsFalse_NoSlotExposed()
    {
        // Quantized caches store quantized rows, not F32 — no in-place slot.
        // Default IKvCache implementation returns false.
        using var cache = new QuantizedKvCache(
            NumLayers, NumKvHeads, HeadDim, maxSeqLen: 16,
            keyDType: KvCacheDType.Q8_0, valueDType: KvCacheDType.Q8_0, windowSize: 0);

        IKvCache ikv = cache;
        Span<int> positions = stackalloc int[] { 0, 1, 2 };
        bool ok = ikv.TryReserveSlot(0, positions, out var kDst, out var vDst);

        Assert.False(ok);
        Assert.True(kDst.IsEmpty);
        Assert.True(vDst.IsEmpty);
    }

    // ── Helpers ────────────────────────────────────────────────────────

    private static void AssertBytesEqual(nint a, nint b, int floatCount)
    {
        var sa = new ReadOnlySpan<byte>((void*)a, floatCount * sizeof(float));
        var sb = new ReadOnlySpan<byte>((void*)b, floatCount * sizeof(float));
        Assert.True(sa.SequenceEqual(sb), "KV buffers must be byte-identical between Update and ReserveSlot paths.");
    }
}
