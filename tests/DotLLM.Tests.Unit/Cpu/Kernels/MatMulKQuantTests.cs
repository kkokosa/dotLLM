using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class MatMulKQuantTests
{
    private const int Q8_K_BlockBytes = 292;
    private const int Q8_K_GroupSize = 256;
    private const int Q4_K_BlockBytes = 144;
    private const int Q5_K_BlockBytes = 176;
    private const int Q6_K_BlockBytes = 210;
    private const int KQuantGroupSize = 256;

    // ──────────────────── Q8_K quantization ────────────────────

    [Fact]
    public void QuantizeF32ToQ8_K_Scalar_RoundtripAccuracy()
    {
        const int k = 256;
        var rng = new Random(42);
        float[] src = new float[k];
        for (int i = 0; i < k; i++) src[i] = rng.NextSingle() * 2f - 1f;

        byte[] dest = new byte[Q8_K_BlockBytes];
        fixed (float* sp = src)
        fixed (byte* dp = dest)
        {
            MatMul.QuantizeF32ToQ8_KScalar(sp, dp, k);

            // Verify float32 scale
            float scale = Unsafe.ReadUnaligned<float>(dp);
            Assert.True(scale > 0, "Scale should be positive for non-zero input");

            // Manual dequantize and compare
            sbyte* qs = (sbyte*)(dp + 4);
            for (int i = 0; i < k; i++)
            {
                float dequantized = scale * qs[i];
                Assert.Equal(src[i], dequantized, MathF.Abs(src[i]) * 0.02f + scale);
            }

            // Verify bsums
            short* bsums = (short*)(dp + 260);
            for (int g = 0; g < 16; g++)
            {
                int expected = 0;
                for (int i = 0; i < 16; i++)
                    expected += qs[g * 16 + i];
                Assert.Equal((short)expected, bsums[g]);
            }
        }
    }

    [Fact]
    public void QuantizeF32ToQ8_K_Avx2MatchesScalar()
    {
        if (!Avx2.IsSupported) return;

        const int k = 512; // 2 blocks
        var rng = new Random(42);
        float[] src = new float[k];
        for (int i = 0; i < k; i++) src[i] = rng.NextSingle() * 2f - 1f;

        int destSize = (k / Q8_K_GroupSize) * Q8_K_BlockBytes;
        byte[] destScalar = new byte[destSize];
        byte[] destAvx2 = new byte[destSize];

        fixed (float* sp = src)
        fixed (byte* dScalar = destScalar, dAvx2 = destAvx2)
        {
            MatMul.QuantizeF32ToQ8_KScalar(sp, dScalar, k);
            MatMul.QuantizeF32ToQ8_KAvx2(sp, dAvx2, k);

            // Scales must match exactly
            for (int b = 0; b < k / Q8_K_GroupSize; b++)
            {
                float sScalar = Unsafe.ReadUnaligned<float>(dScalar + b * Q8_K_BlockBytes);
                float sAvx2 = Unsafe.ReadUnaligned<float>(dAvx2 + b * Q8_K_BlockBytes);
                Assert.Equal(sScalar, sAvx2);
            }

            // qs values must match exactly
            for (int i = 0; i < destSize; i++)
                Assert.Equal(destScalar[i], destAvx2[i]);
        }
    }

    [Fact]
    public void QuantizeF32ToQ8_K_AllZeros()
    {
        const int k = 256;
        float[] src = new float[k];
        byte[] dest = new byte[Q8_K_BlockBytes];

        fixed (float* sp = src)
        fixed (byte* dp = dest)
        {
            MatMul.QuantizeF32ToQ8_K(sp, dp, k);

            float scale = Unsafe.ReadUnaligned<float>(dp);
            Assert.Equal(0f, scale);

            sbyte* qs = (sbyte*)(dp + 4);
            for (int i = 0; i < k; i++)
                Assert.Equal(0, qs[i]);

            short* bsums = (short*)(dp + 260);
            for (int i = 0; i < 16; i++)
                Assert.Equal(0, bsums[i]);
        }
    }

    // ──────────────────── Q4_K × Q8_K vec_dot ────────────────────

    [Fact]
    public void VecDotQ4_K_Q8_K_CrossVerifyAgainstDequant()
    {
        const int superBlockCount = 1;
        const int k = KQuantGroupSize;

        var rng = new Random(42);
        nint qkPtr = AllocRandomKQuantBlock(Q4_K_BlockBytes, rng);
        nint q8kPtr = AllocRandomQ8_KBlock(k, rng);
        try
        {
            float vecDotResult = MatMul.VecDotQ4_K_Q8_KScalar((byte*)qkPtr, (byte*)q8kPtr, superBlockCount);

            // Dequantize Q4_K to float, dequant Q8_K to float, compute float dot
            float[] qkFloats = new float[k];
            Dequantize.ToFloat32(qkPtr, k, QuantizationType.Q4_K, qkFloats);

            float d8 = Unsafe.ReadUnaligned<float>((byte*)q8kPtr);
            sbyte* q8qs = (sbyte*)((byte*)q8kPtr + 4);
            float[] q8Floats = new float[k];
            for (int i = 0; i < k; i++)
                q8Floats[i] = d8 * q8qs[i];

            float refDot = 0;
            for (int i = 0; i < k; i++)
                refDot += qkFloats[i] * q8Floats[i];

            Assert.Equal(refDot, vecDotResult, MathF.Abs(refDot) * 0.15f + 1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8kPtr);
        }
    }

    [Fact]
    public void VecDotQ4_K_Q8_K_ScalarMatchesAvx2()
    {
        if (!Avx2.IsSupported) return;

        const int superBlockCount = 4;
        var rng = new Random(42);

        nint qkPtr = AllocRandomKQuantBlocks(Q4_K_BlockBytes, superBlockCount, rng);
        nint q8kPtr = AllocRandomQ8_KBlocks(superBlockCount, rng);
        try
        {
            float scalar = MatMul.VecDotQ4_K_Q8_KScalar((byte*)qkPtr, (byte*)q8kPtr, superBlockCount);
            float avx2 = MatMul.VecDotQ4_K_Q8_KAvx2((byte*)qkPtr, (byte*)q8kPtr, superBlockCount);

            Assert.Equal(scalar, avx2, MathF.Abs(scalar) * 0.01f + 0.1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8kPtr);
        }
    }

    // ──────────────────── Q6_K × Q8_K vec_dot ────────────────────

    [Fact]
    public void VecDotQ6_K_Q8_K_CrossVerifyAgainstDequant()
    {
        const int superBlockCount = 1;
        const int k = KQuantGroupSize;

        var rng = new Random(42);
        nint qkPtr = AllocRandomKQuantBlock(Q6_K_BlockBytes, rng);
        nint q8kPtr = AllocRandomQ8_KBlock(k, rng);
        try
        {
            Unsafe.WriteUnaligned((byte*)qkPtr + 208, (Half)0.01f);

            float vecDotResult = MatMul.VecDotQ6_K_Q8_KScalar((byte*)qkPtr, (byte*)q8kPtr, superBlockCount);

            float[] qkFloats = new float[k];
            Dequantize.ToFloat32(qkPtr, k, QuantizationType.Q6_K, qkFloats);

            float d8 = Unsafe.ReadUnaligned<float>((byte*)q8kPtr);
            sbyte* q8qs = (sbyte*)((byte*)q8kPtr + 4);
            float[] q8Floats = new float[k];
            for (int i = 0; i < k; i++)
                q8Floats[i] = d8 * q8qs[i];

            float refDot = 0;
            for (int i = 0; i < k; i++)
                refDot += qkFloats[i] * q8Floats[i];

            Assert.Equal(refDot, vecDotResult, MathF.Abs(refDot) * 0.15f + 1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8kPtr);
        }
    }

    [Fact]
    public void VecDotQ6_K_Q8_K_ScalarMatchesAvx2()
    {
        if (!Avx2.IsSupported) return;

        const int superBlockCount = 4;
        var rng = new Random(42);

        nint qkPtr = AllocRandomKQuantBlocks(Q6_K_BlockBytes, superBlockCount, rng);
        nint q8kPtr = AllocRandomQ8_KBlocks(superBlockCount, rng);
        try
        {
            for (int b = 0; b < superBlockCount; b++)
                Unsafe.WriteUnaligned((byte*)qkPtr + b * Q6_K_BlockBytes + 208, (Half)0.01f);

            float scalar = MatMul.VecDotQ6_K_Q8_KScalar((byte*)qkPtr, (byte*)q8kPtr, superBlockCount);
            float avx2 = MatMul.VecDotQ6_K_Q8_KAvx2((byte*)qkPtr, (byte*)q8kPtr, superBlockCount);

            Assert.Equal(scalar, avx2, MathF.Abs(scalar) * 0.01f + 0.1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8kPtr);
        }
    }

    // ──────────────────── Q5_K × Q8_K vec_dot ────────────────────

    [Fact]
    public void VecDotQ5_K_Q8_K_CrossVerifyAgainstDequant()
    {
        const int superBlockCount = 1;
        const int k = KQuantGroupSize;

        var rng = new Random(42);
        nint qkPtr = AllocRandomKQuantBlock(Q5_K_BlockBytes, rng);
        nint q8kPtr = AllocRandomQ8_KBlock(k, rng);
        try
        {
            float vecDotResult = MatMul.VecDotQ5_K_Q8_KScalar((byte*)qkPtr, (byte*)q8kPtr, superBlockCount);

            float[] qkFloats = new float[k];
            Dequantize.ToFloat32(qkPtr, k, QuantizationType.Q5_K, qkFloats);

            float d8 = Unsafe.ReadUnaligned<float>((byte*)q8kPtr);
            sbyte* q8qs = (sbyte*)((byte*)q8kPtr + 4);
            float[] q8Floats = new float[k];
            for (int i = 0; i < k; i++)
                q8Floats[i] = d8 * q8qs[i];

            float refDot = 0;
            for (int i = 0; i < k; i++)
                refDot += qkFloats[i] * q8Floats[i];

            Assert.Equal(refDot, vecDotResult, MathF.Abs(refDot) * 0.15f + 1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8kPtr);
        }
    }

    [Fact]
    public void VecDotQ5_K_Q8_K_ScalarMatchesAvx2()
    {
        if (!Avx2.IsSupported) return;

        const int superBlockCount = 4;
        var rng = new Random(42);

        nint qkPtr = AllocRandomKQuantBlocks(Q5_K_BlockBytes, superBlockCount, rng);
        nint q8kPtr = AllocRandomQ8_KBlocks(superBlockCount, rng);
        try
        {
            float scalar = MatMul.VecDotQ5_K_Q8_KScalar((byte*)qkPtr, (byte*)q8kPtr, superBlockCount);
            float avx2 = MatMul.VecDotQ5_K_Q8_KAvx2((byte*)qkPtr, (byte*)q8kPtr, superBlockCount);

            Assert.Equal(scalar, avx2, MathF.Abs(scalar) * 0.01f + 0.1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8kPtr);
        }
    }

    // ──────────────────── GEMV ────────────────────

    [Fact]
    public void GemvQ4_K_ProducesFiniteResults()
    {
        const int m = 4, k = 256;
        int superBlockCount = k / KQuantGroupSize;
        var rng = new Random(42);

        nint weightsPtr = AllocRandomKQuantBlocks(Q4_K_BlockBytes, m * superBlockCount, rng);
        try
        {
            float[] x = new float[k];
            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

            float[] result = new float[m];
            fixed (float* xp = x, rp = result)
                MatMul.GemvQ4_K((byte*)weightsPtr, xp, rp, m, k);

            for (int i = 0; i < m; i++)
                Assert.True(float.IsFinite(result[i]), $"result[{i}] = {result[i]}");
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    [Fact]
    public void GemvQ6_K_ProducesFiniteResults()
    {
        const int m = 4, k = 256;
        int superBlockCount = k / KQuantGroupSize;
        var rng = new Random(42);

        nint weightsPtr = AllocRandomKQuantBlocks(Q6_K_BlockBytes, m * superBlockCount, rng);
        try
        {
            for (int i = 0; i < m * superBlockCount; i++)
                Unsafe.WriteUnaligned((byte*)weightsPtr + i * Q6_K_BlockBytes + 208, (Half)0.01f);

            float[] x = new float[k];
            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

            float[] result = new float[m];
            fixed (float* xp = x, rp = result)
                MatMul.GemvQ6_K((byte*)weightsPtr, xp, rp, m, k);

            for (int i = 0; i < m; i++)
                Assert.True(float.IsFinite(result[i]), $"result[{i}] = {result[i]}");
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    [Fact]
    public void GemvQ4_K_NonAlignedK_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
        {
            fixed (float* dummy = new float[100])
                MatMul.GemvQ4_K(null, dummy, dummy, 1, 100);
        });
    }

    // ──────────────────── GEMM ────────────────────

    [Fact]
    public void GemmQ4_K_N1_MatchesGemv()
    {
        const int m = 4, k = 256;
        int superBlockCount = k / KQuantGroupSize;
        var rng = new Random(42);

        nint weightsPtr = AllocRandomKQuantBlocks(Q4_K_BlockBytes, m * superBlockCount, rng);
        try
        {
            float[] x = new float[k];
            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

            float[] gemvResult = new float[m];
            float[] gemmResult = new float[m];

            fixed (float* xp = x, gr = gemvResult, gm = gemmResult)
            {
                MatMul.GemvQ4_K((byte*)weightsPtr, xp, gr, m, k);
                MatMul.GemmQ4_K((byte*)weightsPtr, xp, gm, m, k, 1);
            }

            for (int i = 0; i < m; i++)
                Assert.Equal(gemvResult[i], gemmResult[i], 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    [Fact]
    public void GemmQ4_K_MatchesSequentialGemv()
    {
        const int m = 4, k = 256, n = 3;
        int superBlockCount = k / KQuantGroupSize;
        var rng = new Random(42);

        nint weightsPtr = AllocRandomKQuantBlocks(Q4_K_BlockBytes, m * superBlockCount, rng);
        try
        {
            float[] b = new float[n * k];
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

            float[] seqResult = new float[n * m];
            fixed (float* bp = b, sp = seqResult)
            {
                for (int t = 0; t < n; t++)
                    MatMul.GemvQ4_K((byte*)weightsPtr, bp + t * k, sp + t * m, m, k);
            }

            float[] gemmResult = new float[n * m];
            fixed (float* bp = b, gp = gemmResult)
                MatMul.GemmQ4_K((byte*)weightsPtr, bp, gp, m, k, n);

            for (int i = 0; i < n * m; i++)
                Assert.Equal(seqResult[i], gemmResult[i], 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    // ──────────────────── 4-row variants ────────────────────

    [Fact]
    public void VecDotQ4_K_4Row_MatchesSingleRow()
    {
        const int superBlockCount = 2;
        var rng = new Random(42);

        nint w0 = AllocRandomKQuantBlocks(Q4_K_BlockBytes, superBlockCount, rng);
        nint w1 = AllocRandomKQuantBlocks(Q4_K_BlockBytes, superBlockCount, rng);
        nint w2 = AllocRandomKQuantBlocks(Q4_K_BlockBytes, superBlockCount, rng);
        nint w3 = AllocRandomKQuantBlocks(Q4_K_BlockBytes, superBlockCount, rng);
        nint q8k = AllocRandomQ8_KBlocks(superBlockCount, rng);
        try
        {
            float r0 = MatMul.VecDotQ4_K_Q8_KScalar((byte*)w0, (byte*)q8k, superBlockCount);
            float r1 = MatMul.VecDotQ4_K_Q8_KScalar((byte*)w1, (byte*)q8k, superBlockCount);
            float r2 = MatMul.VecDotQ4_K_Q8_KScalar((byte*)w2, (byte*)q8k, superBlockCount);
            float r3 = MatMul.VecDotQ4_K_Q8_KScalar((byte*)w3, (byte*)q8k, superBlockCount);

            float* results = stackalloc float[4];
            MatMul.VecDotQ4_K_Q8_K_4Rows((byte*)w0, (byte*)w1, (byte*)w2, (byte*)w3,
                (byte*)q8k, superBlockCount, results);

            float tol = 0.5f;
            Assert.Equal(r0, results[0], tol);
            Assert.Equal(r1, results[1], tol);
            Assert.Equal(r2, results[2], tol);
            Assert.Equal(r3, results[3], tol);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)w0);
            NativeMemory.AlignedFree((void*)w1);
            NativeMemory.AlignedFree((void*)w2);
            NativeMemory.AlignedFree((void*)w3);
            NativeMemory.AlignedFree((void*)q8k);
        }
    }

    [Fact]
    public void VecDotQ6_K_4Row_MatchesSingleRow()
    {
        const int superBlockCount = 2;
        var rng = new Random(42);

        nint w0 = AllocRandomKQuantBlocks(Q6_K_BlockBytes, superBlockCount, rng);
        nint w1 = AllocRandomKQuantBlocks(Q6_K_BlockBytes, superBlockCount, rng);
        nint w2 = AllocRandomKQuantBlocks(Q6_K_BlockBytes, superBlockCount, rng);
        nint w3 = AllocRandomKQuantBlocks(Q6_K_BlockBytes, superBlockCount, rng);
        nint q8k = AllocRandomQ8_KBlocks(superBlockCount, rng);
        try
        {
            float r0 = MatMul.VecDotQ6_K_Q8_KScalar((byte*)w0, (byte*)q8k, superBlockCount);
            float r1 = MatMul.VecDotQ6_K_Q8_KScalar((byte*)w1, (byte*)q8k, superBlockCount);
            float r2 = MatMul.VecDotQ6_K_Q8_KScalar((byte*)w2, (byte*)q8k, superBlockCount);
            float r3 = MatMul.VecDotQ6_K_Q8_KScalar((byte*)w3, (byte*)q8k, superBlockCount);

            float* results = stackalloc float[4];
            MatMul.VecDotQ6_K_Q8_K_4Rows((byte*)w0, (byte*)w1, (byte*)w2, (byte*)w3,
                (byte*)q8k, superBlockCount, results);

            float tol = 0.5f;
            Assert.Equal(r0, results[0], tol);
            Assert.Equal(r1, results[1], tol);
            Assert.Equal(r2, results[2], tol);
            Assert.Equal(r3, results[3], tol);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)w0);
            NativeMemory.AlignedFree((void*)w1);
            NativeMemory.AlignedFree((void*)w2);
            NativeMemory.AlignedFree((void*)w3);
            NativeMemory.AlignedFree((void*)q8k);
        }
    }

    [Fact]
    public void VecDotQ5_K_4Row_MatchesSingleRow()
    {
        const int superBlockCount = 2;
        var rng = new Random(42);

        nint w0 = AllocRandomKQuantBlocks(Q5_K_BlockBytes, superBlockCount, rng);
        nint w1 = AllocRandomKQuantBlocks(Q5_K_BlockBytes, superBlockCount, rng);
        nint w2 = AllocRandomKQuantBlocks(Q5_K_BlockBytes, superBlockCount, rng);
        nint w3 = AllocRandomKQuantBlocks(Q5_K_BlockBytes, superBlockCount, rng);
        nint q8k = AllocRandomQ8_KBlocks(superBlockCount, rng);
        try
        {
            float r0 = MatMul.VecDotQ5_K_Q8_KScalar((byte*)w0, (byte*)q8k, superBlockCount);
            float r1 = MatMul.VecDotQ5_K_Q8_KScalar((byte*)w1, (byte*)q8k, superBlockCount);
            float r2 = MatMul.VecDotQ5_K_Q8_KScalar((byte*)w2, (byte*)q8k, superBlockCount);
            float r3 = MatMul.VecDotQ5_K_Q8_KScalar((byte*)w3, (byte*)q8k, superBlockCount);

            float* results = stackalloc float[4];
            MatMul.VecDotQ5_K_Q8_K_4Rows((byte*)w0, (byte*)w1, (byte*)w2, (byte*)w3,
                (byte*)q8k, superBlockCount, results);

            float tol = 0.5f;
            Assert.Equal(r0, results[0], tol);
            Assert.Equal(r1, results[1], tol);
            Assert.Equal(r2, results[2], tol);
            Assert.Equal(r3, results[3], tol);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)w0);
            NativeMemory.AlignedFree((void*)w1);
            NativeMemory.AlignedFree((void*)w2);
            NativeMemory.AlignedFree((void*)w3);
            NativeMemory.AlignedFree((void*)q8k);
        }
    }

    // ──────────────────── Helpers ────────────────────

    private static nint AllocRandomKQuantBlock(int blockBytes, Random rng)
    {
        return AllocRandomKQuantBlocks(blockBytes, 1, rng);
    }

    private static nint AllocRandomKQuantBlocks(int blockBytes, int blockCount, Random rng)
    {
        nuint totalBytes = (nuint)(blockCount * blockBytes);
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        byte[] buf = new byte[(int)totalBytes];
        rng.NextBytes(buf);
        fixed (byte* src = buf)
            NativeMemory.Copy(src, (void*)ptr, totalBytes);

        for (int b = 0; b < blockCount; b++)
        {
            byte* block = (byte*)ptr + b * blockBytes;
            if (blockBytes == Q6_K_BlockBytes)
            {
                Unsafe.WriteUnaligned(block + 208, (Half)(rng.NextSingle() * 0.02f));
            }
            else
            {
                Unsafe.WriteUnaligned(block, (Half)(rng.NextSingle() * 0.1f));
                Unsafe.WriteUnaligned(block + 2, (Half)(rng.NextSingle() * 0.1f));
            }
        }

        return ptr;
    }

    /// <summary>
    /// Allocates a Q8_K block by quantizing random f32 data.
    /// This produces structurally valid Q8_K blocks with correct bsums.
    /// </summary>
    private static nint AllocRandomQ8_KBlock(int elementCount, Random rng)
    {
        int blockCount = elementCount / Q8_K_GroupSize;
        return AllocRandomQ8_KBlocks(blockCount, rng);
    }

    private static nint AllocRandomQ8_KBlocks(int blockCount, Random rng)
    {
        int elementCount = blockCount * Q8_K_GroupSize;
        float[] src = new float[elementCount];
        for (int i = 0; i < elementCount; i++)
            src[i] = rng.NextSingle() * 2f - 1f;

        nuint totalBytes = (nuint)(blockCount * Q8_K_BlockBytes);
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);

        fixed (float* sp = src)
            MatMul.QuantizeF32ToQ8_K(sp, (byte*)ptr, elementCount);

        return ptr;
    }
}
