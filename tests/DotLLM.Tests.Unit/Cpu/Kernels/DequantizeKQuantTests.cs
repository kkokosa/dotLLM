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
