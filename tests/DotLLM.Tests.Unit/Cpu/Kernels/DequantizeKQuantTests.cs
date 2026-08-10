using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class DequantizeKQuantTests
{
    private const int Q4_K_BlockBytes = 144;
    private const int Q5_K_BlockBytes = 176;
    private const int Q6_K_BlockBytes = 210;
    private const int KQuantGroupSize = 256;

    // ──────────────────── Scale unpacking ────────────────────

    [Fact]
    public void UnpackQ4Q5Scales_KnownPattern()
    {
        // Build a known 12-byte scale pack and verify unpacking.
        // Sub-blocks 0-3: scales12[0..3] low 6 bits are scales, scales12[4..7] low 6 bits are mins.
        // Sub-blocks 4-7: complex packing with bytes 8-11 and high bits of 0-7.
        byte* scales12 = stackalloc byte[12];

        // Simple case: all values in [0..63], no high-bit overlap
        // scales12[0] = 10 (scale 0), scales12[1] = 20, scales12[2] = 30, scales12[3] = 40
        // scales12[4] = 5 (min 0), scales12[5] = 15, scales12[6] = 25, scales12[7] = 35
        // scales12[8..11] = 0 (no high bits for sub-blocks 4-7)
        scales12[0] = 10; scales12[1] = 20; scales12[2] = 30; scales12[3] = 40;
        scales12[4] = 5;  scales12[5] = 15; scales12[6] = 25; scales12[7] = 35;
        scales12[8] = 0;  scales12[9] = 0;  scales12[10] = 0; scales12[11] = 0;

        byte* sc = stackalloc byte[8];
        byte* mn = stackalloc byte[8];
        Dequantize.UnpackQ4Q5Scales(scales12, sc, mn);

        // Sub-blocks 0-3
        Assert.Equal(10, sc[0]);
        Assert.Equal(20, sc[1]);
        Assert.Equal(30, sc[2]);
        Assert.Equal(40, sc[3]);

        Assert.Equal(5, mn[0]);
        Assert.Equal(15, mn[1]);
        Assert.Equal(25, mn[2]);
        Assert.Equal(35, mn[3]);

        // Sub-blocks 4-7: low nibble of scales12[8..11] = 0, high 2 bits of scales12[0..3] = 0
        // So scales[4..7] = 0, mins[4..7] = 0
        for (int j = 4; j < 8; j++)
        {
            Assert.Equal(0, sc[j]);
            Assert.Equal(0, mn[j]);
        }
    }

    [Fact]
    public void UnpackQ4Q5Scales_AllMax()
    {
        // All bytes = 0xFF → max 6-bit values = 63
        byte* scales12 = stackalloc byte[12];
        for (int i = 0; i < 12; i++) scales12[i] = 0xFF;

        byte* sc = stackalloc byte[8];
        byte* mn = stackalloc byte[8];
        Dequantize.UnpackQ4Q5Scales(scales12, sc, mn);

        // Sub-blocks 0-3: 0xFF & 63 = 63
        for (int j = 0; j < 4; j++)
        {
            Assert.Equal(63, sc[j]);
            Assert.Equal(63, mn[j]);
        }

        // Sub-blocks 4-7: (0xFF & 0xF) | ((0xFF >> 6) << 4) = 15 | (3 << 4) = 15 | 48 = 63
        for (int j = 4; j < 8; j++)
        {
            Assert.Equal(63, sc[j]);
            Assert.Equal(63, mn[j]);
        }
    }

    // ──────────────────── Q6_K dequant ────────────────────

    [Fact]
    public void Q6_K_SingleBlock_AllZeroScales_ProducesZeros()
    {
        nuint totalBytes = Q6_K_BlockBytes;
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            NativeMemory.Clear((void*)ptr, totalBytes);
            // d = 1.0 but all scales = 0 → all outputs = 0
            Unsafe.WriteUnaligned((byte*)ptr + 208, (Half)1.0f);

            float[] dest = new float[KQuantGroupSize];
            Dequantize.ToFloat32(ptr, KQuantGroupSize, QuantizationType.Q6_K, dest);

            Assert.All(dest, v => Assert.Equal(0f, v));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q6_K_SingleBlock_HandCalculated()
    {
        nuint totalBytes = Q6_K_BlockBytes;
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            NativeMemory.Clear((void*)ptr, totalBytes);
            byte* block = (byte*)ptr;

            // Set d = 0.5
            Unsafe.WriteUnaligned(block + 208, (Half)0.5f);

            // Set scale[0] = 2 (first sub-block of 16)
            ((sbyte*)(block + 192))[0] = 2;

            // Set value[0]: ql[0] low nibble = 5, qh[0] bits [0:1] = 1 → q = 5 | (1<<4) = 21
            // result = 0.5 * 2 * (21 - 32) = 1.0 * (-11) = -11.0
            block[0] = 5; // ql[0] = 0x05 (low nibble = 5, high nibble = 0)
            block[128] = 1; // qh[0] low 2 bits = 1

            float[] dest = new float[KQuantGroupSize];
            Dequantize.ToFloat32(ptr, KQuantGroupSize, QuantizationType.Q6_K, dest);

            Assert.Equal(-11.0f, dest[0], 0.1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q6_K_ScalarMatchesAvx2_RandomBlocks()
    {
        if (!Avx2.IsSupported) return;

        const int blockCount = 8;
        const int totalElements = blockCount * KQuantGroupSize;
        nuint totalBytes = (nuint)(blockCount * Q6_K_BlockBytes);

        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            FillRandomBytes((byte*)ptr, (int)totalBytes, new Random(42));

            // Fix d values to be reasonable
            for (int b = 0; b < blockCount; b++)
            {
                byte* block = (byte*)ptr + b * Q6_K_BlockBytes;
                Unsafe.WriteUnaligned(block + 208, (Half)(0.01f));
            }

            float[] scalarDest = new float[totalElements];
            float[] avx2Dest = new float[totalElements];

            Dequantize.DequantizeQ6_KScalar(ptr, totalElements, scalarDest);
            Dequantize.DequantizeQ6_KAvx2(ptr, totalElements, avx2Dest);

            for (int i = 0; i < totalElements; i++)
                Assert.Equal(scalarDest[i], avx2Dest[i], 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(16)]
    public void Q6_K_MultipleBlocks(int blockCount)
    {
        int totalElements = blockCount * KQuantGroupSize;
        nuint totalBytes = (nuint)(blockCount * Q6_K_BlockBytes);

        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            FillRandomBytes((byte*)ptr, (int)totalBytes, new Random(42));

            for (int b = 0; b < blockCount; b++)
                Unsafe.WriteUnaligned((byte*)ptr + b * Q6_K_BlockBytes + 208, (Half)0.01f);

            float[] dest = new float[totalElements];
            Dequantize.ToFloat32(ptr, totalElements, QuantizationType.Q6_K, dest);

            Assert.All(dest, v => Assert.True(float.IsFinite(v), $"Non-finite value: {v}"));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── Q3_K dequant ────────────────────

    private const int Q3_K_BlockBytes = 110;

    [Fact]
    public void Q3_K_SingleBlock_HandCalculated()
    {
        // Block layout: hmask[32] + qs[64] + scales[12] + d[2] = 110 bytes.
        nuint totalBytes = Q3_K_BlockBytes;
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            NativeMemory.Clear((void*)ptr, totalBytes);
            byte* block = (byte*)ptr;

            // d = 1.0
            Unsafe.WriteUnaligned(block + 32 + 64 + 12, (Half)1.0f);

            // scales12 (offset 32+64=96):
            //   sub 0 → low nibble in scales12[0] (low 4 bits) + high 2 bits in scales12[8] bits 0-1
            //   We want unsigned scale = 33 (= 32 + 1 → signed scale = +1).
            //   33 = 0b100001 → low nibble 0b0001 (=1), high 2 bits 0b10 (=2).
            block[96 + 0] = 0x01;        // scales12[0] = low nibble
            block[96 + 8] = 0x02;        // scales12[8] bit 0-1 = high 2 bits of scale[0]

            // qs[0] (offset 32): set element 0's 2 low bits to 0b11 (= 3)
            block[32 + 0] = 0x03;

            // hmask[0] (offset 0): set element 0's high bit to 1
            block[0] = 0x01;

            // Element 0: signed_3bit = ((1<<2) | 3) - 4 = 7 - 4 = 3
            // Signed scale = 33 - 32 = 1
            // d × scale × signed_3bit = 1.0 × 1 × 3 = 3.0
            float[] dest = new float[KQuantGroupSize];
            Dequantize.ToFloat32(ptr, KQuantGroupSize, QuantizationType.Q3_K, dest);

            Assert.Equal(3.0f, dest[0], 0.01f);

            // Element 1 (no qs/hmask bits set, scale[0] = 1):
            //   signed_3bit = (0 << 2 | 0) - 4 = -4
            //   value = 1.0 × 1 × -4 = -4
            Assert.Equal(-4.0f, dest[1], 0.01f);

            // Sub-block 1 (elements 16..31) has scale[1] = 0 - 32 = -32 → all values = 1 × -32 × -4 = 128
            // (since qs/hmask are all zero, signed_3bit = -4 for every element).
            Assert.Equal(128.0f, dest[16], 0.01f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    /// <summary>
    /// Discriminating Q3_K oracle test (#311). <see cref="Q3_K_SingleBlock_HandCalculated"/>
    /// only touches element 0/1 of sub-block 0 and an all-zero sub-block 1 — a degenerate
    /// case where the correct and the incorrect bit layouts coincide, which is exactly why
    /// a total scramble of Q3_K could sit in this PR undetected.
    ///
    /// This test drives DENSE pseudorandom super-block bytes through
    /// <see cref="Dequantize.ToFloat32"/> and compares against a LITERAL transcription of
    /// llama.cpp's <c>ggml-quants.c dequantize_row_q3_K</c> — including the 32-bit
    /// <c>aux</c>/<c>kmask</c> scale shuffle and the <c>shift</c>/<c>m</c> loop over
    /// 128-element halves. The reference is written in llama.cpp's own control-flow shape,
    /// structurally unlike the production kernel's closed-form indexing, so agreement is
    /// evidence rather than a shared-mistake tautology.
    ///
    /// Discrimination proof: reverting either half of the fix makes this red —
    /// the scale hi-bits byte/shift transposition (<c>8 + sub%4 @ (sub/4)*2</c> →
    /// <c>8 + sub/4 @ (sub%4)*2</c>) or the element ordering
    /// (<c>qs[(t%32)+32*(t/128)] @ ((t/32)%4)*2</c> → <c>qs[t/4] @ (t%4)*2</c>).
    /// </summary>
    [Fact]
    public void Q3_K_DenseRandomBlocks_MatchLlamaCppReference()
    {
        const int blocks = 5;
        const int elements = blocks * KQuantGroupSize;
        nuint totalBytes = (nuint)(blocks * Q3_K_BlockBytes);
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            var rng = new Random(20260810);
            byte* raw = (byte*)ptr;
            for (int i = 0; i < (int)totalBytes; i++) raw[i] = (byte)rng.Next(256);
            // Keep the fp16 super-block deltas finite and O(1) so the comparison is
            // about bit layout, not about NaN/Inf plumbing.
            for (int b = 0; b < blocks; b++)
                Unsafe.WriteUnaligned(raw + b * Q3_K_BlockBytes + 108, (Half)(0.25f + 0.125f * b));

            float[] actual = new float[elements];
            Dequantize.ToFloat32(ptr, elements, QuantizationType.Q3_K, actual);

            float[] expected = LlamaCppDequantizeRowQ3K(raw, blocks);

            // Both sides compute the identical product in float — require exact equality.
            for (int i = 0; i < elements; i++)
            {
                Assert.True(expected[i] == actual[i],
                    $"Q3_K element {i} (block {i / KQuantGroupSize}, sub {(i % KQuantGroupSize) / 16}, "
                    + $"lane {i % 16}): llama.cpp reference {expected[i]} != dotLLM {actual[i]}");
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    /// <summary>
    /// Literal transcription of llama.cpp <c>ggml-quants.c dequantize_row_q3_K</c>
    /// (the authoritative GGUF Q3_K semantics), kept in its original control-flow shape
    /// on purpose — see <see cref="Q3_K_DenseRandomBlocks_MatchLlamaCppReference"/>.
    /// </summary>
    private static float[] LlamaCppDequantizeRowQ3K(byte* src, int nb)
    {
        const uint kmask1 = 0x03030303u;
        const uint kmask2 = 0x0f0f0f0fu;

        var y = new float[nb * KQuantGroupSize];
        int outIdx = 0;
        uint* aux = stackalloc uint[4];
        sbyte* scales = (sbyte*)aux;

        for (int i = 0; i < nb; i++)
        {
            byte* block = src + i * Q3_K_BlockBytes;
            byte* hm = block;               // hmask[32]
            byte* q = block + 32;           // qs[64]
            float dAll = (float)Unsafe.ReadUnaligned<Half>(block + 108);

            for (int w = 0; w < 3; w++) aux[w] = Unsafe.ReadUnaligned<uint>(block + 96 + w * 4);
            uint tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            byte m = 1;
            int qOff = 0;
            int isIdx = 0;
            for (int n = 0; n < KQuantGroupSize; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; ++j)
                {
                    float dl = dAll * (scales[isIdx++] - 32);
                    for (int l = 0; l < 16; ++l)
                        y[outIdx++] = dl * (((q[qOff + l] >> shift) & 3) - (((hm[l] & m) != 0) ? 0 : 4));

                    dl = dAll * (scales[isIdx++] - 32);
                    for (int l = 0; l < 16; ++l)
                        y[outIdx++] = dl * (((q[qOff + l + 16] >> shift) & 3) - (((hm[l + 16] & m) != 0) ? 0 : 4));

                    shift += 2;
                    m <<= 1;
                }
                qOff += 32;
            }
        }
        return y;
    }

    [Fact]
    public void Q3_K_RowByteSize_Matches()
    {
        // 256 elements = 1 super-block = 110 bytes.
        Assert.Equal(110L, Dequantize.RowByteSize(256, QuantizationType.Q3_K));
        // 1024 elements = 4 super-blocks = 440 bytes.
        Assert.Equal(440L, Dequantize.RowByteSize(1024, QuantizationType.Q3_K));
    }

    [Fact]
    public void Q3_K_NonAlignedCount_Throws()
    {
        float[] dest = new float[100];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 100, QuantizationType.Q3_K, dest));
    }

    // ──────────────────── Q2_K dequant ────────────────────

    private const int Q2_K_BlockBytes = 84;

    [Fact]
    public void Q2_K_SingleBlock_HandCalculated()
    {
        // Block layout: scales[16] + qs[64] + d[2] + dmin[2] = 84 bytes.
        nuint totalBytes = Q2_K_BlockBytes;
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            NativeMemory.Clear((void*)ptr, totalBytes);
            byte* block = (byte*)ptr;

            // d = 1.0, dmin = 0.5
            Unsafe.WriteUnaligned(block + 80, (Half)1.0f);
            Unsafe.WriteUnaligned(block + 82, (Half)0.5f);

            // scales[0]: low nibble = scale (we want scale = 3), high nibble = dmin coef (we want 2).
            // Packed as: (dmin_coef << 4) | scale = (2 << 4) | 3 = 0x23
            block[0] = 0x23;

            // qs[0] (offset 16): set element 0's 2 low bits to 0b10 (= 2).
            // qs encoding: 4 elements per byte, low-to-high.
            //   byte 0, bits 0-1 → element 0
            //   byte 0, bits 2-3 → element 1
            //   byte 0, bits 4-5 → element 2
            //   byte 0, bits 6-7 → element 3
            block[16 + 0] = 0x02;  // element 0 = 2, elements 1-3 = 0

            // Element 0: q2 = 2, scale = 3, dmin_coef = 2
            //   value = d * scale * q2 - dmin * dmin_coef
            //         = 1.0 * 3 * 2 - 0.5 * 2
            //         = 6 - 1 = 5
            float[] dest = new float[KQuantGroupSize];
            Dequantize.ToFloat32(ptr, KQuantGroupSize, QuantizationType.Q2_K, dest);

            Assert.Equal(5.0f, dest[0], 0.01f);

            // Element 1: q2 = 0, scale = 3, dmin_coef = 2
            //   value = 1.0 * 3 * 0 - 0.5 * 2 = -1
            Assert.Equal(-1.0f, dest[1], 0.01f);

            // Sub-block 1 (elements 16..31): scale = 0, dmin_coef = 0 (all-zero scales[1..15])
            //   value = 1.0 * 0 * 0 - 0.5 * 0 = 0
            Assert.Equal(0.0f, dest[16], 0.01f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q2_K_RowByteSize_Matches()
    {
        // 256 elements = 1 super-block = 84 bytes.
        Assert.Equal(84L, Dequantize.RowByteSize(256, QuantizationType.Q2_K));
        // 1024 elements = 4 super-blocks = 336 bytes.
        Assert.Equal(336L, Dequantize.RowByteSize(1024, QuantizationType.Q2_K));
    }

    [Fact]
    public void Q2_K_ComputeByteCount_MatchesRowByteSize()
    {
        // QuantizationTypeExtensions must agree with the CPU block size (84 bytes / 256).
        Assert.Equal(84L, QuantizationType.Q2_K.ComputeByteCount(256));
        Assert.Equal(336L, QuantizationType.Q2_K.ComputeByteCount(1024));
        Assert.Equal(
            Dequantize.RowByteSize(1024, QuantizationType.Q2_K),
            QuantizationType.Q2_K.ComputeByteCount(1024));
    }

    [Fact]
    public void Q2_K_NonAlignedCount_Throws()
    {
        float[] dest = new float[100];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 100, QuantizationType.Q2_K, dest));
    }

    [Fact]
    public void Q2_K_TwoSuperBlocks_StrideCorrect()
    {
        // Two super-blocks of 256 elements each = 168 bytes total.
        // SB0: d=1.0, dmin=0.0, scales[0]=0x03 (scale=3, dmin_coef=0), qs[0]=0x01 (element 0 q2=1)
        // SB1: d=2.0, dmin=0.0, scales[0]=0x05 (scale=5, dmin_coef=0), qs[0]=0x03 (element 0 q2=3)
        // Expect: dest[0]   = 1.0 * 3 * 1 - 0 = 3.0       (SB0, element 0)
        //         dest[256] = 2.0 * 5 * 3 - 0 = 30.0      (SB1, element 0)
        // Catches super-block stride bugs (e.g. sb*80 instead of sb*84).
        nuint totalBytes = 2 * Q2_K_BlockBytes;  // 168
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            NativeMemory.Clear((void*)ptr, totalBytes);
            byte* sb0 = (byte*)ptr;
            byte* sb1 = (byte*)ptr + Q2_K_BlockBytes;

            // SB0
            Unsafe.WriteUnaligned(sb0 + 80, (Half)1.0f);
            Unsafe.WriteUnaligned(sb0 + 82, (Half)0.0f);
            sb0[0] = 0x03;
            sb0[16] = 0x01;

            // SB1
            Unsafe.WriteUnaligned(sb1 + 80, (Half)2.0f);
            Unsafe.WriteUnaligned(sb1 + 82, (Half)0.0f);
            sb1[0] = 0x05;
            sb1[16] = 0x03;

            float[] dest = new float[2 * KQuantGroupSize];
            Dequantize.ToFloat32(ptr, 2 * KQuantGroupSize, QuantizationType.Q2_K, dest);

            Assert.Equal(3.0f,  dest[0],   0.01f);
            Assert.Equal(30.0f, dest[256], 0.01f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── Q4_K dequant ────────────────────

    [Fact]
    public void Q4_K_SingleBlock_HandCalculated()
    {
        nuint totalBytes = Q4_K_BlockBytes;
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            NativeMemory.Clear((void*)ptr, totalBytes);
            byte* block = (byte*)ptr;

            // d = 1.0, dmin = 0.5
            Unsafe.WriteUnaligned(block, (Half)1.0f);
            Unsafe.WriteUnaligned(block + 2, (Half)0.5f);

            // scales12: scale[0] = 3, min[0] = 2 (simple packing)
            block[4] = 3;  // scale[0] low 6 bits
            block[8] = 2;  // min[0] low 6 bits

            // qs[0]: low nibble = 7 (value[0])
            block[16] = 7;

            // value[0] = d * scale[0] * nibble - dmin * min[0] = 1.0 * 3 * 7 - 0.5 * 2 = 21 - 1 = 20
            float[] dest = new float[KQuantGroupSize];
            Dequantize.ToFloat32(ptr, KQuantGroupSize, QuantizationType.Q4_K, dest);

            Assert.Equal(20.0f, dest[0], 0.5f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q4_K_ScalarMatchesAvx2_RandomBlocks()
    {
        if (!Avx2.IsSupported) return;

        const int blockCount = 8;
        const int totalElements = blockCount * KQuantGroupSize;
        nuint totalBytes = (nuint)(blockCount * Q4_K_BlockBytes);

        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            FillRandomBytes((byte*)ptr, (int)totalBytes, new Random(42));

            for (int b = 0; b < blockCount; b++)
            {
                byte* block = (byte*)ptr + b * Q4_K_BlockBytes;
                Unsafe.WriteUnaligned(block, (Half)0.01f);
                Unsafe.WriteUnaligned(block + 2, (Half)0.01f);
            }

            float[] scalarDest = new float[totalElements];
            float[] avx2Dest = new float[totalElements];

            Dequantize.DequantizeQ4_KScalar(ptr, totalElements, scalarDest);
            Dequantize.DequantizeQ4_KAvx2(ptr, totalElements, avx2Dest);

            for (int i = 0; i < totalElements; i++)
                Assert.Equal(scalarDest[i], avx2Dest[i], 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── Q5_K dequant ────────────────────

    [Fact]
    public void Q5_K_SingleBlock_HandCalculated()
    {
        nuint totalBytes = Q5_K_BlockBytes;
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            NativeMemory.Clear((void*)ptr, totalBytes);
            byte* block = (byte*)ptr;

            // d = 1.0, dmin = 0.5
            Unsafe.WriteUnaligned(block, (Half)1.0f);
            Unsafe.WriteUnaligned(block + 2, (Half)0.5f);

            // scale[0] = 2, min[0] = 1
            block[4] = 2; // scale[0]
            block[8] = 1; // min[0]

            // qs[0] low nibble = 3 (lo4 for value 0)
            block[48] = 3;

            // qh: 5th bit for value 0 → qh[0] bit 0 = 1
            block[16] = 1;

            // value[0] = d * scale[0] * (lo4 | bit5<<4) - dmin * min[0]
            //          = 1.0 * 2 * (3 | 16) - 0.5 * 1 = 2 * 19 - 0.5 = 38 - 0.5 = 37.5
            float[] dest = new float[KQuantGroupSize];
            Dequantize.ToFloat32(ptr, KQuantGroupSize, QuantizationType.Q5_K, dest);

            Assert.Equal(37.5f, dest[0], 0.5f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q5_K_ScalarMatchesAvx2_RandomBlocks()
    {
        if (!Avx2.IsSupported) return;

        const int blockCount = 8;
        const int totalElements = blockCount * KQuantGroupSize;
        nuint totalBytes = (nuint)(blockCount * Q5_K_BlockBytes);

        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            FillRandomBytes((byte*)ptr, (int)totalBytes, new Random(42));

            for (int b = 0; b < blockCount; b++)
            {
                byte* block = (byte*)ptr + b * Q5_K_BlockBytes;
                Unsafe.WriteUnaligned(block, (Half)0.01f);
                Unsafe.WriteUnaligned(block + 2, (Half)0.01f);
            }

            float[] scalarDest = new float[totalElements];
            float[] avx2Dest = new float[totalElements];

            Dequantize.DequantizeQ5_KScalar(ptr, totalElements, scalarDest);
            Dequantize.DequantizeQ5_KAvx2(ptr, totalElements, avx2Dest);

            for (int i = 0; i < totalElements; i++)
                Assert.Equal(scalarDest[i], avx2Dest[i], 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── Dispatch ────────────────────

    [Fact]
    public void Q4_K_NonAlignedCount_Throws()
    {
        float[] dest = new float[300];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 100, QuantizationType.Q4_K, dest));
    }

    [Fact]
    public void Q5_K_NonAlignedCount_Throws()
    {
        float[] dest = new float[300];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 100, QuantizationType.Q5_K, dest));
    }

    [Fact]
    public void Q6_K_NonAlignedCount_Throws()
    {
        float[] dest = new float[300];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 100, QuantizationType.Q6_K, dest));
    }

    // ──────────────────── Helpers ────────────────────

    private static void FillRandomBytes(byte* ptr, int count, Random rng)
    {
        byte[] buf = new byte[count];
        rng.NextBytes(buf);
        fixed (byte* src = buf)
            NativeMemory.Copy(src, ptr, (nuint)count);
    }
}
