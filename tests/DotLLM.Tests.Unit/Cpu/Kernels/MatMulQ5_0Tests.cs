using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class MatMulQ5_0Tests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q5_0BlockBytes = 22;
    private const int Q8_0GroupSize = 32;
    private const int Q5_0GroupSize = 32;

    // ──────────────────── Q5_0 × Q8_0 scalar cross-verify ────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(18)]  // 576/32 = 18 blocks (SmolLM hidden dim)
    public void VecDotQ5_0Q8_0Scalar_CrossVerifyAgainstDequant(int blockCount)
    {
        int k = blockCount * Q5_0GroupSize;
        var rng = new Random(42);
        nint q5Ptr = AllocRandomQ5_0Blocks(blockCount, rng);
        nint q8Ptr = AllocRandomQ8_0Blocks(blockCount, rng);
        try
        {
            float vecDotResult = MatMul.VecDotQ5_0Q8_0Scalar((byte*)q5Ptr, (byte*)q8Ptr, blockCount);

            // Dequantize both sides to float and compute dot product
            float[] q5Floats = new float[k];
            float[] q8Floats = new float[k];
            Dequantize.ToFloat32(q5Ptr, k, QuantizationType.Q5_0, q5Floats);
            Dequantize.ToFloat32(q8Ptr, k, QuantizationType.Q8_0, q8Floats);

            float refDot = 0;
            for (int i = 0; i < k; i++)
                refDot += q5Floats[i] * q8Floats[i];

            // Allow tolerance due to integer vs float arithmetic differences
            Assert.Equal(refDot, vecDotResult, MathF.Abs(refDot) * 0.05f + 0.5f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)q5Ptr);
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    // ──────────────────── Q5_0 × Q8_0 scalar matches AVX2 ────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(18)]
    public void VecDotQ5_0Q8_0_ScalarMatchesAvx2(int blockCount)
    {
        if (!Avx2.IsSupported) return;

        var rng = new Random(42);
        nint q5Ptr = AllocRandomQ5_0Blocks(blockCount, rng);
        nint q8Ptr = AllocRandomQ8_0Blocks(blockCount, rng);
        try
        {
            float scalar = MatMul.VecDotQ5_0Q8_0Scalar((byte*)q5Ptr, (byte*)q8Ptr, blockCount);
            float avx2 = MatMul.VecDotQ5_0Q8_0Avx2((byte*)q5Ptr, (byte*)q8Ptr, blockCount);

            Assert.Equal(scalar, avx2, MathF.Abs(scalar) * 1e-5f + 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)q5Ptr);
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(18)]
    [InlineData(128)]
    public void VecDotQ5_0Q8_1_Neon_BitIdenticalToScalarPure(int blockCount)
    {
        if (!Dp.IsSupported || !AdvSimd.Arm64.IsSupported)
            return;

        var rng = new Random(42);
        int k = blockCount * Q5_0GroupSize;
        nint q5Ptr = AllocRandomQ5_0Blocks(blockCount, rng);
        nint q8_1Ptr = (nint)NativeMemory.AlignedAlloc((nuint)(blockCount * MatMul.Q8_1BlockBytes), 64);
        try
        {
            float[] input = new float[k];
            for (int i = 0; i < k; i++)
                input[i] = rng.NextSingle() * 2f - 1f;
            fixed (float* ip = input)
                MatMul.QuantizeF32ToQ8_1(ip, (byte*)q8_1Ptr, k);

            float scalarPure = MatMul.VecDotQ5_0Q8_1ScalarPure((byte*)q5Ptr, (byte*)q8_1Ptr, blockCount);
            float neon = MatMul.VecDotQ5_0Q8_1Neon((byte*)q5Ptr, (byte*)q8_1Ptr, blockCount);
            float dispatched = MatMul.VecDotQ5_0Q8_1Scalar((byte*)q5Ptr, (byte*)q8_1Ptr, blockCount);

            Assert.Equal(
                BitConverter.SingleToInt32Bits(scalarPure),
                BitConverter.SingleToInt32Bits(neon));
            Assert.Equal(
                BitConverter.SingleToInt32Bits(scalarPure),
                BitConverter.SingleToInt32Bits(dispatched));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)q5Ptr);
            NativeMemory.AlignedFree((void*)q8_1Ptr);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(18)]
    [InlineData(128)]
    public void VecDotQ5_0Q8_1R4_Neon_BitIdenticalToScalarR4Pure(int blockCount)
    {
        if (!Dp.IsSupported || !AdvSimd.Arm64.IsSupported)
            return;

        var rng = new Random(42);
        int k = blockCount * Q5_0GroupSize;
        nint q5Ptr = AllocRandomQ5_0Blocks(4 * blockCount, rng);
        nint q8_1Ptr = (nint)NativeMemory.AlignedAlloc((nuint)(blockCount * MatMul.Q8_1BlockBytes), 64);
        try
        {
            float[] input = new float[k];
            for (int i = 0; i < k; i++)
                input[i] = rng.NextSingle() * 2f - 1f;
            fixed (float* ip = input)
                MatMul.QuantizeF32ToQ8_1(ip, (byte*)q8_1Ptr, k);

            using var rw = WeightRepacking.RepackR4(q5Ptr, QuantizationType.Q5_0, 4, k);

            for (int r = 0; r < 4; r++)
            {
                float scalarPure = MatMul.VecDotQ5_0Q8_1ScalarR4Pure((byte*)rw.Ptr, r, (byte*)q8_1Ptr, blockCount);
                float neon = MatMul.VecDotQ5_0Q8_1NeonR4((byte*)rw.Ptr, r, (byte*)q8_1Ptr, blockCount);
                float dispatched = MatMul.VecDotQ5_0Q8_1ScalarR4((byte*)rw.Ptr, r, (byte*)q8_1Ptr, blockCount);

                Assert.Equal(
                    BitConverter.SingleToInt32Bits(scalarPure),
                    BitConverter.SingleToInt32Bits(neon));
                Assert.Equal(
                    BitConverter.SingleToInt32Bits(scalarPure),
                    BitConverter.SingleToInt32Bits(dispatched));
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)q5Ptr);
            NativeMemory.AlignedFree((void*)q8_1Ptr);
        }
    }

    [Fact]
    public void VecDotQ5_0Q8_1_Neon_All32HighBitPatterns_BitIdenticalToScalarPure()
    {
        if (!Dp.IsSupported || !AdvSimd.Arm64.IsSupported)
            return;

        var rng = new Random(123);
        nint q5Ptr = AllocRandomQ5_0Blocks(1, rng);
        nint q8_1Ptr = (nint)NativeMemory.AlignedAlloc((nuint)MatMul.Q8_1BlockBytes, 64);
        try
        {
            float[] input = new float[Q5_0GroupSize];
            for (int i = 0; i < Q5_0GroupSize; i++)
                input[i] = rng.NextSingle() * 2f - 1f;
            fixed (float* ip = input)
                MatMul.QuantizeF32ToQ8_1(ip, (byte*)q8_1Ptr, Q5_0GroupSize);

            for (int bit = 0; bit < 32; bit++)
            {
                uint qh = 1u << bit;
                Unsafe.WriteUnaligned((byte*)q5Ptr + 2, qh);

                float scalarPure = MatMul.VecDotQ5_0Q8_1ScalarPure((byte*)q5Ptr, (byte*)q8_1Ptr, 1);
                float neon = MatMul.VecDotQ5_0Q8_1Neon((byte*)q5Ptr, (byte*)q8_1Ptr, 1);
                Assert.Equal(
                    BitConverter.SingleToInt32Bits(scalarPure),
                    BitConverter.SingleToInt32Bits(neon));
            }

            Unsafe.WriteUnaligned((byte*)q5Ptr + 2, 0xFFFFFFFFu);
            Assert.Equal(
                BitConverter.SingleToInt32Bits(MatMul.VecDotQ5_0Q8_1ScalarPure((byte*)q5Ptr, (byte*)q8_1Ptr, 1)),
                BitConverter.SingleToInt32Bits(MatMul.VecDotQ5_0Q8_1Neon((byte*)q5Ptr, (byte*)q8_1Ptr, 1)));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)q5Ptr);
            NativeMemory.AlignedFree((void*)q8_1Ptr);
        }
    }

    // ──────────────────── GemvQ5_0 cross-verify ────────────────────

    [Fact]
    public void GemvQ5_0_CrossVerifyAgainstDequant()
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
            uint qh = ((uint)rng.Next() << 1) ^ (uint)rng.Next();
            Unsafe.WriteUnaligned(block + 2, qh);
            for (int i = 0; i < 16; i++)
                block[6 + i] = (byte)rng.Next(256);
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
