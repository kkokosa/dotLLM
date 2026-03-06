using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class MatMulKQuantTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int Q4_K_BlockBytes = 144;
    private const int Q5_K_BlockBytes = 176;
    private const int Q6_K_BlockBytes = 210;
    private const int KQuantGroupSize = 256;

    // ──────────────────── Q4_K vec_dot scalar ────────────────────

    [Fact]
    public void VecDotQ4_K_Q8_0Scalar_CrossVerifyAgainstDequant()
    {
        // Cross-verify: dequantize Q4_K to float, quantize to Q8_0, then float dot
        // should approximately match the fused vec_dot.
        const int superBlockCount = 1;
        const int k = KQuantGroupSize;

        var rng = new Random(42);
        nint qkPtr = AllocRandomKQuantBlock(Q4_K_BlockBytes, rng);
        nint q8Ptr = AllocRandomQ8_0Blocks(8, rng); // 8 Q8_0 blocks = 256 values
        try
        {
            float vecDotResult = MatMul.VecDotQ4_K_Q8_0Scalar((byte*)qkPtr, (byte*)q8Ptr, superBlockCount);

            // Dequantize both sides to float and compute dot product
            float[] qkFloats = new float[k];
            float[] q8Floats = new float[k];
            Dequantize.ToFloat32(qkPtr, k, QuantizationType.Q4_K, qkFloats);
            Dequantize.ToFloat32(q8Ptr, k, QuantizationType.Q8_0, q8Floats);

            float refDot = 0;
            for (int i = 0; i < k; i++)
                refDot += qkFloats[i] * q8Floats[i];

            // Allow generous tolerance due to double quantization error
            Assert.Equal(refDot, vecDotResult, MathF.Abs(refDot) * 0.15f + 1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    [Fact]
    public void VecDotQ4_K_Q8_0_ScalarMatchesAvx2()
    {
        if (!Avx2.IsSupported) return;

        const int superBlockCount = 4;
        var rng = new Random(42);

        nint qkPtr = AllocRandomKQuantBlocks(Q4_K_BlockBytes, superBlockCount, rng);
        nint q8Ptr = AllocRandomQ8_0Blocks(superBlockCount * 8, rng);
        try
        {
            float scalar = MatMul.VecDotQ4_K_Q8_0Scalar((byte*)qkPtr, (byte*)q8Ptr, superBlockCount);
            float avx2 = MatMul.VecDotQ4_K_Q8_0Avx2((byte*)qkPtr, (byte*)q8Ptr, superBlockCount);

            Assert.Equal(scalar, avx2, MathF.Abs(scalar) * 0.01f + 0.1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    // ──────────────────── Q6_K vec_dot ────────────────────

    [Fact]
    public void VecDotQ6_K_Q8_0Scalar_CrossVerifyAgainstDequant()
    {
        const int superBlockCount = 1;
        const int k = KQuantGroupSize;

        var rng = new Random(42);
        nint qkPtr = AllocRandomKQuantBlock(Q6_K_BlockBytes, rng);
        nint q8Ptr = AllocRandomQ8_0Blocks(8, rng);
        try
        {
            // Fix d to reasonable value
            Unsafe.WriteUnaligned((byte*)qkPtr + 208, (Half)0.01f);

            float vecDotResult = MatMul.VecDotQ6_K_Q8_0Scalar((byte*)qkPtr, (byte*)q8Ptr, superBlockCount);

            float[] qkFloats = new float[k];
            float[] q8Floats = new float[k];
            Dequantize.ToFloat32(qkPtr, k, QuantizationType.Q6_K, qkFloats);
            Dequantize.ToFloat32(q8Ptr, k, QuantizationType.Q8_0, q8Floats);

            float refDot = 0;
            for (int i = 0; i < k; i++)
                refDot += qkFloats[i] * q8Floats[i];

            Assert.Equal(refDot, vecDotResult, MathF.Abs(refDot) * 0.15f + 1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    [Fact]
    public void VecDotQ6_K_Q8_0_ScalarMatchesAvx2()
    {
        if (!Avx2.IsSupported) return;

        const int superBlockCount = 4;
        var rng = new Random(42);

        nint qkPtr = AllocRandomKQuantBlocks(Q6_K_BlockBytes, superBlockCount, rng);
        nint q8Ptr = AllocRandomQ8_0Blocks(superBlockCount * 8, rng);
        try
        {
            for (int b = 0; b < superBlockCount; b++)
                Unsafe.WriteUnaligned((byte*)qkPtr + b * Q6_K_BlockBytes + 208, (Half)0.01f);

            float scalar = MatMul.VecDotQ6_K_Q8_0Scalar((byte*)qkPtr, (byte*)q8Ptr, superBlockCount);
            float avx2 = MatMul.VecDotQ6_K_Q8_0Avx2((byte*)qkPtr, (byte*)q8Ptr, superBlockCount);

            Assert.Equal(scalar, avx2, MathF.Abs(scalar) * 0.01f + 0.1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    // ──────────────────── Q5_K vec_dot ────────────────────

    [Fact]
    public void VecDotQ5_K_Q8_0Scalar_CrossVerifyAgainstDequant()
    {
        const int superBlockCount = 1;
        const int k = KQuantGroupSize;

        var rng = new Random(42);
        nint qkPtr = AllocRandomKQuantBlock(Q5_K_BlockBytes, rng);
        nint q8Ptr = AllocRandomQ8_0Blocks(8, rng);
        try
        {
            float vecDotResult = MatMul.VecDotQ5_K_Q8_0Scalar((byte*)qkPtr, (byte*)q8Ptr, superBlockCount);

            float[] qkFloats = new float[k];
            float[] q8Floats = new float[k];
            Dequantize.ToFloat32(qkPtr, k, QuantizationType.Q5_K, qkFloats);
            Dequantize.ToFloat32(q8Ptr, k, QuantizationType.Q8_0, q8Floats);

            float refDot = 0;
            for (int i = 0; i < k; i++)
                refDot += qkFloats[i] * q8Floats[i];

            Assert.Equal(refDot, vecDotResult, MathF.Abs(refDot) * 0.15f + 1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    [Fact]
    public void VecDotQ5_K_Q8_0_ScalarMatchesAvx2()
    {
        if (!Avx2.IsSupported) return;

        const int superBlockCount = 4;
        var rng = new Random(42);

        nint qkPtr = AllocRandomKQuantBlocks(Q5_K_BlockBytes, superBlockCount, rng);
        nint q8Ptr = AllocRandomQ8_0Blocks(superBlockCount * 8, rng);
        try
        {
            float scalar = MatMul.VecDotQ5_K_Q8_0Scalar((byte*)qkPtr, (byte*)q8Ptr, superBlockCount);
            float avx2 = MatMul.VecDotQ5_K_Q8_0Avx2((byte*)qkPtr, (byte*)q8Ptr, superBlockCount);

            Assert.Equal(scalar, avx2, MathF.Abs(scalar) * 0.01f + 0.1f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)qkPtr);
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    // ──────────────────── GEMV ────────────────────

    [Fact]
    public void GemvQ4_K_ProducesFiniteResults()
    {
        const int m = 4, k = 256;
        int superBlockCount = k / KQuantGroupSize;
        int rowBytes = superBlockCount * Q4_K_BlockBytes;
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
            // Set reasonable d values
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

            // Sequential GEMV
            float[] seqResult = new float[n * m];
            fixed (float* bp = b, sp = seqResult)
            {
                for (int t = 0; t < n; t++)
                    MatMul.GemvQ4_K((byte*)weightsPtr, bp + t * k, sp + t * m, m, k);
            }

            // Batched GEMM
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
        nint q8 = AllocRandomQ8_0Blocks(superBlockCount * 8, rng);
        try
        {
            float r0 = MatMul.VecDotQ4_K_Q8_0Scalar((byte*)w0, (byte*)q8, superBlockCount);
            float r1 = MatMul.VecDotQ4_K_Q8_0Scalar((byte*)w1, (byte*)q8, superBlockCount);
            float r2 = MatMul.VecDotQ4_K_Q8_0Scalar((byte*)w2, (byte*)q8, superBlockCount);
            float r3 = MatMul.VecDotQ4_K_Q8_0Scalar((byte*)w3, (byte*)q8, superBlockCount);

            float* results = stackalloc float[4];
            MatMul.VecDotQ4_K_Q8_0Avx2_4Rows((byte*)w0, (byte*)w1, (byte*)w2, (byte*)w3,
                (byte*)q8, superBlockCount, results);

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
            NativeMemory.AlignedFree((void*)q8);
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

        // Fix Half scale values to be reasonable (not NaN/Inf)
        for (int b = 0; b < blockCount; b++)
        {
            byte* block = (byte*)ptr + b * blockBytes;
            if (blockBytes == Q6_K_BlockBytes)
            {
                // d is at offset 208
                Unsafe.WriteUnaligned(block + 208, (Half)(rng.NextSingle() * 0.02f));
            }
            else
            {
                // d at offset 0, dmin at offset 2
                Unsafe.WriteUnaligned(block, (Half)(rng.NextSingle() * 0.1f));
                Unsafe.WriteUnaligned(block + 2, (Half)(rng.NextSingle() * 0.1f));
            }
        }

        return ptr;
    }

    private static nint AllocRandomQ8_0Blocks(int blockCount, Random rng)
    {
        nuint totalBytes = (nuint)(blockCount * Q8_0BlockBytes);
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);

        for (int b = 0; b < blockCount; b++)
        {
            byte* block = (byte*)ptr + b * Q8_0BlockBytes;
            Unsafe.WriteUnaligned(block, (Half)(rng.NextSingle() * 0.1f));
            for (int i = 0; i < Q8_0GroupSize; i++)
                ((sbyte*)(block + 2))[i] = (sbyte)rng.Next(-127, 128);
        }

        return ptr;
    }
}
