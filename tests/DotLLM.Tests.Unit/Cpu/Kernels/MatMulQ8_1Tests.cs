using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Tests for Q8_1 quantization and Q5_0 × Q8_1 vec_dot kernels.
/// Q8_1 extends Q8_0 with a precomputed block sum <c>s = d * sum(qs)</c>,
/// eliminating 4 SIMD ops per block in Q5_0 dot products.
/// </summary>
public sealed unsafe class MatMulQ8_1Tests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_1BlockBytes = 36;
    private const int Q5_0BlockBytes = 22;
    private const int Q8_0GroupSize = 32;
    private const int Q8_1GroupSize = 32;
    private const int Q5_0GroupSize = 32;

    // ──────────────────── Q8_1 quantization: precomputed sum is correct ────────────────────

    [Theory]
    [InlineData(32)]
    [InlineData(576)]   // SmolLM hidden dim
    [InlineData(1536)]  // SmolLM intermediate dim
    public void QuantizeF32ToQ8_1_PrecomputedSumIsCorrect(int elementCount)
    {
        var rng = new Random(42);
        float* src = (float*)NativeMemory.AlignedAlloc((nuint)(elementCount * sizeof(float)), 64);
        int blockCount = elementCount / Q8_1GroupSize;
        int destBytes = blockCount * Q8_1BlockBytes;
        byte* dest = (byte*)NativeMemory.AlignedAlloc((nuint)destBytes, 64);

        try
        {
            for (int i = 0; i < elementCount; i++)
                src[i] = (rng.NextSingle() - 0.5f) * 10f;

            MatMul.QuantizeF32ToQ8_1(src, dest, elementCount);

            for (int b = 0; b < blockCount; b++)
            {
                byte* block = dest + b * Q8_1BlockBytes;
                float d = (float)Unsafe.ReadUnaligned<Half>(block);
                float s = (float)Unsafe.ReadUnaligned<Half>(block + 2);
                sbyte* qs = (sbyte*)(block + 4);

                int sum = 0;
                for (int i = 0; i < Q8_1GroupSize; i++)
                    sum += qs[i];

                float expectedS = d * sum;
                // Half precision: allow ULP-level tolerance
                Assert.Equal(expectedS, s, MathF.Abs(expectedS) * 0.02f + 0.01f);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(src);
            NativeMemory.AlignedFree(dest);
        }
    }

    // ──────────────────── Q8_1 quantization: scalar matches AVX2 ────────────────────

    [Theory]
    [InlineData(32)]
    [InlineData(576)]
    [InlineData(1536)]
    public void QuantizeF32ToQ8_1_ScalarMatchesAvx2(int elementCount)
    {
        if (!Avx2.IsSupported) return;

        var rng = new Random(42);
        float* src = (float*)NativeMemory.AlignedAlloc((nuint)(elementCount * sizeof(float)), 64);
        int blockCount = elementCount / Q8_1GroupSize;
        int destBytes = blockCount * Q8_1BlockBytes;
        byte* destScalar = (byte*)NativeMemory.AlignedAlloc((nuint)destBytes, 64);
        byte* destAvx2 = (byte*)NativeMemory.AlignedAlloc((nuint)destBytes, 64);

        try
        {
            for (int i = 0; i < elementCount; i++)
                src[i] = (rng.NextSingle() - 0.5f) * 10f;

            MatMul.QuantizeF32ToQ8_1Scalar(src, destScalar, elementCount);
            MatMul.QuantizeF32ToQ8_1Avx2(src, destAvx2, elementCount);

            for (int b = 0; b < blockCount; b++)
            {
                byte* blockS = destScalar + b * Q8_1BlockBytes;
                byte* blockA = destAvx2 + b * Q8_1BlockBytes;

                // d should be identical (same max-abs scan → same Half)
                Half dS = Unsafe.ReadUnaligned<Half>(blockS);
                Half dA = Unsafe.ReadUnaligned<Half>(blockA);
                Assert.Equal(dS, dA);

                // qs values should match
                for (int i = 0; i < Q8_1GroupSize; i++)
                {
                    sbyte qsS = ((sbyte*)(blockS + 4))[i];
                    sbyte qsA = ((sbyte*)(blockA + 4))[i];
                    Assert.Equal(qsS, qsA);
                }

                // s should match (computed from same qs and d)
                Half sS = Unsafe.ReadUnaligned<Half>(blockS + 2);
                Half sA = Unsafe.ReadUnaligned<Half>(blockA + 2);
                Assert.Equal(sS, sA);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(src);
            NativeMemory.AlignedFree(destScalar);
            NativeMemory.AlignedFree(destAvx2);
        }
    }

    // ──────────────────── Q5_0 × Q8_1 scalar matches Q8_0 path ────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(18)]  // 576/32 = 18 blocks (SmolLM hidden dim)
    public void VecDotQ5_0Q8_1Scalar_MatchesQ8_0Path(int blockCount)
    {
        int k = blockCount * Q5_0GroupSize;
        var rng = new Random(42);

        // Generate random f32 input and Q5_0 weights
        float* input = (float*)NativeMemory.AlignedAlloc((nuint)(k * sizeof(float)), 64);
        for (int i = 0; i < k; i++)
            input[i] = (rng.NextSingle() - 0.5f) * 2f;

        nint q5Ptr = AllocRandomQ5_0Blocks(blockCount, rng);

        // Quantize input to both Q8_0 and Q8_1
        int q8_0Bytes = blockCount * Q8_0BlockBytes;
        int q8_1Bytes = blockCount * Q8_1BlockBytes;
        byte* q8_0 = (byte*)NativeMemory.AlignedAlloc((nuint)q8_0Bytes, 64);
        byte* q8_1 = (byte*)NativeMemory.AlignedAlloc((nuint)q8_1Bytes, 64);

        try
        {
            MatMul.QuantizeF32ToQ8_0(input, q8_0, k);
            MatMul.QuantizeF32ToQ8_1(input, q8_1, k);

            float resultQ8_0 = MatMul.VecDotQ5_0Q8_0Scalar((byte*)q5Ptr, q8_0, blockCount);
            float resultQ8_1 = MatMul.VecDotQ5_0Q8_1Scalar((byte*)q5Ptr, q8_1, blockCount);

            // Allow tolerance for Half precision of s field (~0.05% error)
            Assert.Equal(resultQ8_0, resultQ8_1, MathF.Abs(resultQ8_0) * 0.01f + 0.5f);
        }
        finally
        {
            NativeMemory.AlignedFree(input);
            NativeMemory.AlignedFree((void*)q5Ptr);
            NativeMemory.AlignedFree(q8_0);
            NativeMemory.AlignedFree(q8_1);
        }
    }

    // ──────────────────── Q5_0 × Q8_1: scalar matches AVX2 ────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(18)]
    public void VecDotQ5_0Q8_1_ScalarMatchesAvx2(int blockCount)
    {
        if (!Avx2.IsSupported) return;

        var rng = new Random(42);
        nint q5Ptr = AllocRandomQ5_0Blocks(blockCount, rng);
        nint q8Ptr = AllocRandomQ8_1Blocks(blockCount, rng);
        try
        {
            float scalar = MatMul.VecDotQ5_0Q8_1Scalar((byte*)q5Ptr, (byte*)q8Ptr, blockCount);
            float avx2 = MatMul.VecDotQ5_0Q8_1Avx2((byte*)q5Ptr, (byte*)q8Ptr, blockCount);

            Assert.Equal(scalar, avx2, MathF.Abs(scalar) * 1e-5f + 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)q5Ptr);
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    // ──────────────────── Q5_0 × Q8_1: 4-row matches single row ────────────────────

    [Fact]
    public void VecDotQ5_0Q8_1_4Row_MatchesSingleRow()
    {
        if (!Avx2.IsSupported) return;

        const int blockCount = 18;
        var rng = new Random(42);

        nint w0 = AllocRandomQ5_0Blocks(blockCount, rng);
        nint w1 = AllocRandomQ5_0Blocks(blockCount, rng);
        nint w2 = AllocRandomQ5_0Blocks(blockCount, rng);
        nint w3 = AllocRandomQ5_0Blocks(blockCount, rng);
        nint q8 = AllocRandomQ8_1Blocks(blockCount, rng);
        try
        {
            float r0 = MatMul.VecDotQ5_0Q8_1Avx2((byte*)w0, (byte*)q8, blockCount);
            float r1 = MatMul.VecDotQ5_0Q8_1Avx2((byte*)w1, (byte*)q8, blockCount);
            float r2 = MatMul.VecDotQ5_0Q8_1Avx2((byte*)w2, (byte*)q8, blockCount);
            float r3 = MatMul.VecDotQ5_0Q8_1Avx2((byte*)w3, (byte*)q8, blockCount);

            float* results = stackalloc float[4];
            MatMul.VecDotQ5_0Q8_1Avx2_4Rows((byte*)w0, (byte*)w1, (byte*)w2, (byte*)w3,
                (byte*)q8, blockCount, results);

            // Allow float rounding tolerance: single-row uses 2-block unrolling, 4-row does not
            Assert.Equal(r0, results[0], MathF.Abs(r0) * 1e-5f + 1e-4f);
            Assert.Equal(r1, results[1], MathF.Abs(r1) * 1e-5f + 1e-4f);
            Assert.Equal(r2, results[2], MathF.Abs(r2) * 1e-5f + 1e-4f);
            Assert.Equal(r3, results[3], MathF.Abs(r3) * 1e-5f + 1e-4f);
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

    // ──────────────────── Q5_0 × Q8_1 AVX2: 2-block unroll odd/even block counts ────────────────────

    [Theory]
    [InlineData(1)]   // odd, single-block tail only
    [InlineData(3)]   // odd, 1 unrolled pair + 1 tail
    [InlineData(18)]  // even, 9 unrolled pairs, no tail
    [InlineData(19)]  // odd, 9 unrolled pairs + 1 tail
    public void VecDotQ5_0Q8_1_2BlockUnroll_OddAndEvenCounts(int blockCount)
    {
        if (!Avx2.IsSupported) return;

        var rng = new Random(42);
        nint q5Ptr = AllocRandomQ5_0Blocks(blockCount, rng);
        nint q8Ptr = AllocRandomQ8_1Blocks(blockCount, rng);
        try
        {
            float scalar = MatMul.VecDotQ5_0Q8_1Scalar((byte*)q5Ptr, (byte*)q8Ptr, blockCount);
            float avx2 = MatMul.VecDotQ5_0Q8_1Avx2((byte*)q5Ptr, (byte*)q8Ptr, blockCount);

            Assert.Equal(scalar, avx2, MathF.Abs(scalar) * 1e-5f + 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)q5Ptr);
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    // ──────────────────── GemvQ5_0 cross-verify with dequant reference ────────────────────

    [Fact]
    public void GemvQ5_0_CrossVerifyWithDequant()
    {
        const int k = 576; // SmolLM hidden dim
        const int m = 64;
        var rng = new Random(42);

        int blockCount = k / Q5_0GroupSize;
        int totalQ5Bytes = m * blockCount * Q5_0BlockBytes;
        nint weightsPtr = (nint)NativeMemory.AlignedAlloc((nuint)totalQ5Bytes, 64);

        // Fill with random Q5_0 blocks
        for (int row = 0; row < m; row++)
        {
            byte* rowPtr = (byte*)weightsPtr + (long)row * blockCount * Q5_0BlockBytes;
            for (int b = 0; b < blockCount; b++)
            {
                byte* block = rowPtr + b * Q5_0BlockBytes;
                Unsafe.WriteUnaligned(block, (Half)(rng.NextSingle() * 0.1f));
                uint qh = (uint)rng.Next();
                Unsafe.WriteUnaligned(block + 2, qh);
                for (int i = 0; i < 16; i++)
                    block[6 + i] = (byte)rng.Next(256);
            }
        }

        // Random f32 input
        float[] input = new float[k];
        for (int i = 0; i < k; i++)
            input[i] = (rng.NextSingle() - 0.5f) * 2f;

        float[] gemvResult = new float[m];
        float[] refResult = new float[m];

        try
        {
            fixed (float* inputPtr = input, resultPtr = gemvResult)
                MatMul.GemvQ5_0((byte*)weightsPtr, inputPtr, resultPtr, m, k);

            // Reference: dequantize each row, float dot
            float[] rowBuf = new float[k];
            for (int row = 0; row < m; row++)
            {
                nint rowPtr = weightsPtr + (nint)((long)row * blockCount * Q5_0BlockBytes);
                Dequantize.ToFloat32(rowPtr, k, QuantizationType.Q5_0, rowBuf);
                float dot = 0;
                for (int i = 0; i < k; i++)
                    dot += rowBuf[i] * input[i];
                refResult[row] = dot;
            }

            for (int row = 0; row < m; row++)
            {
                Assert.Equal(refResult[row], gemvResult[row],
                    MathF.Abs(refResult[row]) * 0.1f + 0.5f);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    // ──────────────────── Helpers ────────────────────

    private static nint AllocRandomQ5_0Blocks(int blockCount, Random rng)
    {
        nuint totalBytes = (nuint)(blockCount * Q5_0BlockBytes);
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);

        for (int b = 0; b < blockCount; b++)
        {
            byte* block = (byte*)ptr + b * Q5_0BlockBytes;
            Unsafe.WriteUnaligned(block, (Half)(rng.NextSingle() * 0.1f));
            uint qh = (uint)rng.Next();
            Unsafe.WriteUnaligned(block + 2, qh);
            for (int i = 0; i < 16; i++)
                block[6 + i] = (byte)rng.Next(256);
        }

        return ptr;
    }

    private static nint AllocRandomQ8_1Blocks(int blockCount, Random rng)
    {
        nuint totalBytes = (nuint)(blockCount * Q8_1BlockBytes);
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);

        for (int b = 0; b < blockCount; b++)
        {
            byte* block = (byte*)ptr + b * Q8_1BlockBytes;
            float scale = rng.NextSingle() * 0.1f;
            Unsafe.WriteUnaligned(block, (Half)scale);
            sbyte* qs = (sbyte*)(block + 4);
            int sum = 0;
            for (int i = 0; i < Q8_1GroupSize; i++)
            {
                sbyte v = (sbyte)rng.Next(-127, 128);
                qs[i] = v;
                sum += v;
            }
            Unsafe.WriteUnaligned(block + 2, (Half)(scale * sum));
        }

        return ptr;
    }
}
