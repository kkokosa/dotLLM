using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class DequantizeTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int Q5_0BlockBytes = 22;
    private const int Q5_0GroupSize = 32;
    private const int Q4_1BlockBytes = 20;
    private const int Q5_1BlockBytes = 24;

    // ──────────────────── FP16 ────────────────────

    [Fact]
    public void Fp16_KnownValues_MatchExpected()
    {
        Half[] input = [Half.Zero, (Half)1.0f, (Half)(-2.5f), Half.MaxValue];
        float[] expected = input.Select(h => (float)h).ToArray();

        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(input.Length * sizeof(Half)), 32);
        try
        {
            input.AsSpan().CopyTo(new Span<Half>((void*)ptr, input.Length));
            float[] dest = new float[input.Length];

            Dequantize.ToFloat32(ptr, input.Length, QuantizationType.F16, dest);

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], dest[i]);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Fp16_AllZeros_ProducesZeros()
    {
        const int count = 64;
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(Half)), 32);
        try
        {
            NativeMemory.Clear((void*)ptr, (nuint)(count * sizeof(Half)));
            float[] dest = new float[count];

            Dequantize.ToFloat32(ptr, count, QuantizationType.F16, dest);

            Assert.All(dest, v => Assert.Equal(0f, v));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Fp16_NegativeValues_Correct()
    {
        Half[] input = [(Half)(-1.0f), (Half)(-0.5f), (Half)(-100.0f)];

        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(input.Length * sizeof(Half)), 32);
        try
        {
            input.AsSpan().CopyTo(new Span<Half>((void*)ptr, input.Length));
            float[] dest = new float[input.Length];

            Dequantize.ToFloat32(ptr, input.Length, QuantizationType.F16, dest);

            for (int i = 0; i < input.Length; i++)
                Assert.Equal((float)input[i], dest[i]);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── Q8_0 ────────────────────

    [Fact]
    public void Q8_0_SingleBlock_HandCalculated()
    {
        // scale = 0.5, qs = [0, 1, 2, ..., 31]
        // expected: [0.0, 0.5, 1.0, 1.5, ..., 15.5]
        nint ptr = AllocQ8_0Block(scale: (Half)0.5f, fillQs: i => (sbyte)i);
        try
        {
            float[] dest = new float[Q8_0GroupSize];
            Dequantize.ToFloat32(ptr, Q8_0GroupSize, QuantizationType.Q8_0, dest);

            for (int i = 0; i < Q8_0GroupSize; i++)
                Assert.Equal(0.5f * i, dest[i], 1e-3f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q8_0_ScaleZero_AllZeros()
    {
        nint ptr = AllocQ8_0Block(scale: Half.Zero, fillQs: i => (sbyte)(i + 1));
        try
        {
            float[] dest = new float[Q8_0GroupSize];
            Dequantize.ToFloat32(ptr, Q8_0GroupSize, QuantizationType.Q8_0, dest);

            Assert.All(dest, v => Assert.Equal(0f, v));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q8_0_MultipleBlocks_DifferentScales()
    {
        const int blockCount = 4;
        const int totalElements = blockCount * Q8_0GroupSize;
        nuint totalBytes = (nuint)(blockCount * Q8_0BlockBytes);

        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 32);
        try
        {
            byte* p = (byte*)ptr;
            for (int b = 0; b < blockCount; b++)
            {
                Half scale = (Half)(b + 1.0f); // scales: 1, 2, 3, 4
                *(Half*)p = scale;
                for (int i = 0; i < Q8_0GroupSize; i++)
                    ((sbyte*)(p + 2))[i] = 1; // all qs = 1
                p += Q8_0BlockBytes;
            }

            float[] dest = new float[totalElements];
            Dequantize.ToFloat32(ptr, totalElements, QuantizationType.Q8_0, dest);

            for (int b = 0; b < blockCount; b++)
            {
                float expectedScale = b + 1.0f;
                for (int i = 0; i < Q8_0GroupSize; i++)
                    Assert.Equal(expectedScale, dest[b * Q8_0GroupSize + i], 1e-3f);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q8_0_MaxValues_NoOverflow()
    {
        // scale = 1.0, qs = 127 (sbyte max) → output = 127.0
        nint ptr = AllocQ8_0Block(scale: (Half)1.0f, fillQs: _ => sbyte.MaxValue);
        try
        {
            float[] dest = new float[Q8_0GroupSize];
            Dequantize.ToFloat32(ptr, Q8_0GroupSize, QuantizationType.Q8_0, dest);

            Assert.All(dest, v => Assert.Equal(127f, v, 1e-3f));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q8_0_NegativeQs_Correct()
    {
        // scale = 2.0, qs = -1 → output = -2.0
        nint ptr = AllocQ8_0Block(scale: (Half)2.0f, fillQs: _ => (sbyte)-1);
        try
        {
            float[] dest = new float[Q8_0GroupSize];
            Dequantize.ToFloat32(ptr, Q8_0GroupSize, QuantizationType.Q8_0, dest);

            Assert.All(dest, v => Assert.Equal(-2.0f, v, 1e-3f));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q8_0_ScalarMatchesSimd_RandomBlocks()
    {
        const int blockCount = 16;
        const int totalElements = blockCount * Q8_0GroupSize;
        nuint totalBytes = (nuint)(blockCount * Q8_0BlockBytes);

        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            // Fill with pseudo-random data.
            var rng = new Random(42);
            byte* p = (byte*)ptr;
            for (int b = 0; b < blockCount; b++)
            {
                *(Half*)p = (Half)(rng.NextSingle() * 2.0f - 1.0f);
                for (int i = 0; i < Q8_0GroupSize; i++)
                    ((sbyte*)(p + 2))[i] = (sbyte)rng.Next(-128, 128);
                p += Q8_0BlockBytes;
            }

            float[] scalarDest = new float[totalElements];
            float[] simdDest = new float[totalElements];

            Dequantize.DequantizeQ8_0Scalar(ptr, totalElements, scalarDest);

            if (Avx2.IsSupported)
            {
                Dequantize.DequantizeQ8_0Avx2(ptr, totalElements, simdDest);

                for (int i = 0; i < totalElements; i++)
                    Assert.Equal(scalarDest[i], simdDest[i], 1e-5f);
            }

            // Also verify dispatch path matches scalar.
            float[] dispatchDest = new float[totalElements];
            Dequantize.ToFloat32(ptr, totalElements, QuantizationType.Q8_0, dispatchDest);

            for (int i = 0; i < totalElements; i++)
                Assert.Equal(scalarDest[i], dispatchDest[i], 1e-5f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── Q5_0 ────────────────────

    [Fact]
    public void Q5_0_Scalar_ZeroPayload_Gives_NegativeSixteenTimesScale()
    {
        // All nibbles + high bits = 0 → every value becomes (0 - 16) * scale = -16 * scale.
        nint ptr = AllocQ5_0Block(scale: (Half)0.5f, qh: 0u, fillQs: _ => 0);
        try
        {
            float[] dest = new float[Q5_0GroupSize];
            Dequantize.ToFloat32(ptr, Q5_0GroupSize, QuantizationType.Q5_0, dest);
            for (int i = 0; i < Q5_0GroupSize; i++)
                Assert.Equal(-8.0f, dest[i]);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q5_0_Scalar_AllBitsSet_Gives_PositiveFifteenTimesScale()
    {
        // All nibbles = 0xFF (lo=0xF, hi=0xF) + all high bits set → each element = (31 - 16) = 15.
        nint ptr = AllocQ5_0Block(scale: (Half)1.0f, qh: 0xFFFFFFFFu, fillQs: _ => 0xFF);
        try
        {
            float[] dest = new float[Q5_0GroupSize];
            Dequantize.ToFloat32(ptr, Q5_0GroupSize, QuantizationType.Q5_0, dest);
            for (int i = 0; i < Q5_0GroupSize; i++)
                Assert.Equal(15.0f, dest[i]);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q5_0_ScalarVsAvx2_MatchOnPseudoRandomBlocks()
    {
        const int blockCount = 32;
        const int totalElements = blockCount * Q5_0GroupSize;
        nuint totalBytes = (nuint)(blockCount * Q5_0BlockBytes);

        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            // Fill with deterministic pseudo-random data.
            var rng = new Random(1337);
            byte* p = (byte*)ptr;
            for (int b = 0; b < blockCount; b++)
            {
                *(Half*)p = (Half)(rng.NextSingle() * 4.0f - 2.0f);
                uint qh = (uint)rng.Next() ^ ((uint)rng.Next() << 16);
                *(uint*)(p + 2) = qh;
                for (int i = 0; i < 16; i++)
                    (p + 6)[i] = (byte)rng.Next(0, 256);
                p += Q5_0BlockBytes;
            }

            float[] scalarDest = new float[totalElements];
            float[] simdDest = new float[totalElements];

            Dequantize.DequantizeQ5_0Scalar(ptr, totalElements, scalarDest);

            if (Avx2.IsSupported)
            {
                Dequantize.DequantizeQ5_0Avx2(ptr, totalElements, simdDest);

                for (int i = 0; i < totalElements; i++)
                    Assert.Equal(scalarDest[i], simdDest[i], 1e-5f);
            }

            // Also verify the public dispatch path matches scalar (exercises the Avx2.IsSupported branch).
            float[] dispatchDest = new float[totalElements];
            Dequantize.ToFloat32(ptr, totalElements, QuantizationType.Q5_0, dispatchDest);

            for (int i = 0; i < totalElements; i++)
                Assert.Equal(scalarDest[i], dispatchDest[i], 1e-5f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── F32 ────────────────────

    [Fact]
    public void F32_CopiesDirectly()
    {
        float[] input = [1.0f, -2.5f, 0f, float.MaxValue];
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(input.Length * sizeof(float)), 32);
        try
        {
            input.AsSpan().CopyTo(new Span<float>((void*)ptr, input.Length));
            float[] dest = new float[input.Length];

            Dequantize.ToFloat32(ptr, input.Length, QuantizationType.F32, dest);

            for (int i = 0; i < input.Length; i++)
                Assert.Equal(input[i], dest[i]);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── Dispatch ────────────────────

    [Fact]
    public void UnsupportedType_Throws()
    {
        float[] dest = new float[32];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Dequantize.ToFloat32(nint.Zero, 32, QuantizationType.Q4_0, dest));
    }

    [Fact]
    public void DestTooSmall_Throws()
    {
        float[] dest = new float[1];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 32, QuantizationType.F32, dest));
    }

    [Fact]
    public void Q8_0_NonAlignedCount_Throws()
    {
        // elementCount = 33 is not a multiple of 32 — must throw before any data access.
        float[] dest = new float[64];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 33, QuantizationType.Q8_0, dest));
    }

    // ──────────────────── Q4_1 ────────────────────

    [Fact]
    public void Q4_1_SingleBlock_HandCalculated()
    {
        // Layout (20 bytes / 32 elements): d(Half@0), m(Half@2), qs[16]@4.
        // value = d * nibble + m, with the low nibble of qs[j] feeding element j and
        // the high nibble feeding element j + 16 (llama.cpp dequantize_row_q4_1).
        // d = 0.5, m = -3. qs[0] = 0x9A → lo = 0xA (10), hi = 0x9 (9).
        //   dest[0]  = 0.5 * 10 - 3 = 2.0
        //   dest[16] = 0.5 *  9 - 3 = 1.5
        // qs[15] = 0x0F → lo = 15, hi = 0.
        //   dest[15] = 0.5 * 15 - 3 = 4.5
        //   dest[31] = 0.5 *  0 - 3 = -3.0
        nint ptr = AllocQ4_1Block((Half)0.5f, (Half)(-3.0f), j => j switch
        {
            0 => (byte)0x9A,
            15 => (byte)0x0F,
            _ => (byte)0x00,
        });
        try
        {
            float[] dest = new float[Q8_0GroupSize];
            Dequantize.ToFloat32(ptr, Q8_0GroupSize, QuantizationType.Q4_1, dest);

            Assert.Equal(2.0f, dest[0], 1e-4f);
            Assert.Equal(1.5f, dest[16], 1e-4f);
            Assert.Equal(4.5f, dest[15], 1e-4f);
            Assert.Equal(-3.0f, dest[31], 1e-4f);
            // Every untouched nibble decodes to the block minimum.
            for (int j = 1; j < 15; j++)
            {
                Assert.Equal(-3.0f, dest[j], 1e-4f);
                Assert.Equal(-3.0f, dest[j + 16], 1e-4f);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q4_1_TwoBlocks_StrideCorrect()
    {
        // Catches a wrong block stride (e.g. 18 bytes — the Q4_0 size — instead of 20).
        const int blockCount = 2;
        nint ptr = (nint)NativeMemory.AlignedAlloc(blockCount * Q4_1BlockBytes, 64);
        try
        {
            NativeMemory.Clear((void*)ptr, blockCount * Q4_1BlockBytes);
            byte* b0 = (byte*)ptr;
            byte* b1 = (byte*)ptr + Q4_1BlockBytes;

            *(Half*)b0 = (Half)1.0f;  *(Half*)(b0 + 2) = (Half)0.0f;  b0[4] = 0x03;  // lo = 3
            *(Half*)b1 = (Half)2.0f;  *(Half*)(b1 + 2) = (Half)1.0f;  b1[4] = 0x05;  // lo = 5

            float[] dest = new float[blockCount * Q8_0GroupSize];
            Dequantize.ToFloat32(ptr, blockCount * Q8_0GroupSize, QuantizationType.Q4_1, dest);

            Assert.Equal(3.0f, dest[0], 1e-4f);   // 1.0 * 3 + 0
            Assert.Equal(11.0f, dest[32], 1e-4f); // 2.0 * 5 + 1
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q4_1_RowByteSize_Matches()
    {
        Assert.Equal(20L, Dequantize.RowByteSize(32, QuantizationType.Q4_1));
        Assert.Equal(640L, Dequantize.RowByteSize(1024, QuantizationType.Q4_1));
    }

    [Fact]
    public void Q4_1_NonAlignedCount_Throws()
    {
        float[] dest = new float[40];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 40, QuantizationType.Q4_1, dest));
    }

    // ──────────────────── Q5_1 ────────────────────

    [Fact]
    public void Q5_1_SingleBlock_HandCalculated()
    {
        // Layout (24 bytes / 32 elements): d(Half@0), m(Half@2), qh[4]@4, qs[16]@8.
        // value = d * ((qh_bit << 4) | nibble) + m. Element j takes qh bit j,
        // element j + 16 takes qh bit j + 16 (llama.cpp dequantize_row_q5_1).
        // d = 0.25, m = 1. qs[0] = 0x21 → lo = 1, hi = 2. qh bit 0 set, bit 16 set.
        //   dest[0]  = 0.25 * (16 | 1) + 1 = 0.25 * 17 + 1 = 5.25
        //   dest[16] = 0.25 * (16 | 2) + 1 = 0.25 * 18 + 1 = 5.5
        // qs[1] = 0x21 with no qh bits set:
        //   dest[1]  = 0.25 * 1 + 1 = 1.25
        //   dest[17] = 0.25 * 2 + 1 = 1.5
        uint qh = (1u << 0) | (1u << 16);
        nint ptr = AllocQ5_1Block((Half)0.25f, (Half)1.0f, qh, j => j <= 1 ? (byte)0x21 : (byte)0x00);
        try
        {
            float[] dest = new float[Q5_0GroupSize];
            Dequantize.ToFloat32(ptr, Q5_0GroupSize, QuantizationType.Q5_1, dest);

            Assert.Equal(5.25f, dest[0], 1e-4f);
            Assert.Equal(5.5f, dest[16], 1e-4f);
            Assert.Equal(1.25f, dest[1], 1e-4f);
            Assert.Equal(1.5f, dest[17], 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q5_1_AllBitsSet_GivesMaxCode()
    {
        // Every nibble 0xF and every high bit set → code 31 everywhere.
        nint ptr = AllocQ5_1Block((Half)1.0f, (Half)0.0f, 0xFFFFFFFFu, _ => 0xFF);
        try
        {
            float[] dest = new float[Q5_0GroupSize];
            Dequantize.ToFloat32(ptr, Q5_0GroupSize, QuantizationType.Q5_1, dest);
            for (int i = 0; i < Q5_0GroupSize; i++)
                Assert.Equal(31.0f, dest[i], 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q5_1_TwoBlocks_StrideCorrect()
    {
        // Catches a wrong block stride (e.g. 22 bytes — the Q5_0 size — instead of 24).
        const int blockCount = 2;
        nint ptr = (nint)NativeMemory.AlignedAlloc(blockCount * Q5_1BlockBytes, 64);
        try
        {
            NativeMemory.Clear((void*)ptr, blockCount * Q5_1BlockBytes);
            byte* b0 = (byte*)ptr;
            byte* b1 = (byte*)ptr + Q5_1BlockBytes;

            *(Half*)b0 = (Half)1.0f;  *(Half*)(b0 + 2) = (Half)0.0f;  *(uint*)(b0 + 4) = 0u;  b0[8] = 0x07;
            *(Half*)b1 = (Half)2.0f;  *(Half*)(b1 + 2) = (Half)1.0f;  *(uint*)(b1 + 4) = 1u;  b1[8] = 0x02;

            float[] dest = new float[blockCount * Q5_0GroupSize];
            Dequantize.ToFloat32(ptr, blockCount * Q5_0GroupSize, QuantizationType.Q5_1, dest);

            Assert.Equal(7.0f, dest[0], 1e-4f);    // 1.0 * 7 + 0
            Assert.Equal(37.0f, dest[32], 1e-4f);  // 2.0 * (16 | 2) + 1
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q5_1_RowByteSize_Matches()
    {
        Assert.Equal(24L, Dequantize.RowByteSize(32, QuantizationType.Q5_1));
        Assert.Equal(768L, Dequantize.RowByteSize(1024, QuantizationType.Q5_1));
    }

    [Fact]
    public void Q5_1_NonAlignedCount_Throws()
    {
        float[] dest = new float[40];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 40, QuantizationType.Q5_1, dest));
    }

    // ──────────────────── Helpers ────────────────────

    private static nint AllocQ4_1Block(Half d, Half m, Func<int, byte> fillQs)
    {
        nint ptr = (nint)NativeMemory.AlignedAlloc(Q4_1BlockBytes, 32);
        byte* p = (byte*)ptr;
        *(Half*)p = d;
        *(Half*)(p + 2) = m;
        for (int i = 0; i < 16; i++)
            (p + 4)[i] = fillQs(i);
        return ptr;
    }

    private static nint AllocQ5_1Block(Half d, Half m, uint qh, Func<int, byte> fillQs)
    {
        nint ptr = (nint)NativeMemory.AlignedAlloc(Q5_1BlockBytes, 32);
        byte* p = (byte*)ptr;
        *(Half*)p = d;
        *(Half*)(p + 2) = m;
        *(uint*)(p + 4) = qh;
        for (int i = 0; i < 16; i++)
            (p + 8)[i] = fillQs(i);
        return ptr;
    }

    private static nint AllocQ8_0Block(Half scale, Func<int, sbyte> fillQs)
    {
        nint ptr = (nint)NativeMemory.AlignedAlloc(Q8_0BlockBytes, 32);
        byte* p = (byte*)ptr;
        *(Half*)p = scale;
        for (int i = 0; i < Q8_0GroupSize; i++)
            ((sbyte*)(p + 2))[i] = fillQs(i);
        return ptr;
    }

    private static nint AllocQ5_0Block(Half scale, uint qh, Func<int, byte> fillQs)
    {
        nint ptr = (nint)NativeMemory.AlignedAlloc(Q5_0BlockBytes, 32);
        byte* p = (byte*)ptr;
        *(Half*)p = scale;
        *(uint*)(p + 2) = qh;
        for (int i = 0; i < 16; i++)
            (p + 6)[i] = fillQs(i);
        return ptr;
    }
}
