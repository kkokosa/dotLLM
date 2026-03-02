using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class MatMulTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    // ──────────────────── f32 GEMV ────────────────────

    [Fact]
    public void GemvF32_2x2_KnownValues()
    {
        // A = [[1, 2], [3, 4]], x = [5, 6]
        // result = [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
        float[] a = [1, 2, 3, 4];
        float[] x = [5, 6];
        float[] result = new float[2];

        fixed (float* ap = a, xp = x, rp = result)
            MatMul.GemvF32(ap, xp, rp, 2, 2);

        Assert.Equal(17f, result[0], 1e-5f);
        Assert.Equal(39f, result[1], 1e-5f);
    }

    [Fact]
    public void GemvF32_Identity_ReturnsInput()
    {
        // 3×3 identity matrix × [1, 2, 3] = [1, 2, 3]
        float[] a = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        float[] x = [1, 2, 3];
        float[] result = new float[3];

        fixed (float* ap = a, xp = x, rp = result)
            MatMul.GemvF32(ap, xp, rp, 3, 3);

        Assert.Equal(1f, result[0], 1e-5f);
        Assert.Equal(2f, result[1], 1e-5f);
        Assert.Equal(3f, result[2], 1e-5f);
    }

    [Fact]
    public void GemvF32_ZeroVector_ReturnsZeros()
    {
        float[] a = [1, 2, 3, 4];
        float[] x = [0, 0];
        float[] result = new float[2];

        fixed (float* ap = a, xp = x, rp = result)
            MatMul.GemvF32(ap, xp, rp, 2, 2);

        Assert.Equal(0f, result[0]);
        Assert.Equal(0f, result[1]);
    }

    [Fact]
    public void GemvF32_ScalarMatchesTensorPrimitives()
    {
        var rng = new Random(42);
        const int m = 16, k = 64;
        float[] a = new float[m * k];
        float[] x = new float[k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        float[] scalarResult = new float[m];
        float[] simdResult = new float[m];

        fixed (float* ap = a, xp = x, sr = scalarResult, ir = simdResult)
        {
            MatMul.GemvF32Scalar(ap, xp, sr, m, k);
            MatMul.GemvF32(ap, xp, ir, m, k);
        }

        for (int i = 0; i < m; i++)
            Assert.Equal(scalarResult[i], simdResult[i], 1e-3f);
    }

    // ──────────────────── Q8_0 VecDot ────────────────────

    [Fact]
    public void VecDotQ8_0Scalar_HandCalculated()
    {
        // 1 block: a.scale=1.0, a.qs=[1..32], b.scale=2.0, b.qs=[1..32]
        // dot = sum(i*i for i in 1..32) = 32*33*65/6 = 11440
        // result = 1.0 * 2.0 * 11440 = 22880
        nint aPtr = AllocQ8_0Block((Half)1.0f, i => (sbyte)(i + 1));
        nint bPtr = AllocQ8_0Block((Half)2.0f, i => (sbyte)(i + 1));
        try
        {
            float result = MatMul.VecDotQ8_0Scalar((byte*)aPtr, (byte*)bPtr, 1);

            // Half precision loses some accuracy on scales, but should be close.
            Assert.Equal(22880f, result, 200f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)aPtr);
            NativeMemory.AlignedFree((void*)bPtr);
        }
    }

    [Fact]
    public void VecDotQ8_0_ScalarMatchesAvx2()
    {
        if (!Avx2.IsSupported)
            return;

        var rng = new Random(42);
        const int blockCount = 16;
        nuint totalBytes = (nuint)(blockCount * Q8_0BlockBytes);

        nint aPtr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        nint bPtr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            FillRandomQ8_0Blocks((byte*)aPtr, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)bPtr, blockCount, rng);

            float scalar = MatMul.VecDotQ8_0Scalar((byte*)aPtr, (byte*)bPtr, blockCount);
            float avx2 = MatMul.VecDotQ8_0Avx2((byte*)aPtr, (byte*)bPtr, blockCount);

            Assert.Equal(scalar, avx2, 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)aPtr);
            NativeMemory.AlignedFree((void*)bPtr);
        }
    }

    // ──────────────────── QuantizeF32ToQ8_0 ────────────────────

    [Fact]
    public void QuantizeF32ToQ8_0_RoundTrip_LowError()
    {
        var rng = new Random(42);
        const int k = 256; // 8 blocks
        float[] src = new float[k];
        for (int i = 0; i < k; i++)
            src[i] = rng.NextSingle() * 2f - 1f;

        int blockCount = k / Q8_0GroupSize;
        int q8Bytes = blockCount * Q8_0BlockBytes;
        nint q8Ptr = (nint)NativeMemory.AlignedAlloc((nuint)q8Bytes, 64);
        try
        {
            fixed (float* srcPtr = src)
                MatMul.QuantizeF32ToQ8_0(srcPtr, (byte*)q8Ptr, k);

            // Dequantize back and check error.
            // Q8_0 quantization error per element is at most maxAbs_in_block / 127.
            // Use absolute error bounded by the block's quantization step.
            float[] roundTrip = new float[k];
            Dequantize.ToFloat32(q8Ptr, k, DotLLM.Core.Configuration.QuantizationType.Q8_0, roundTrip);

            for (int block = 0; block < blockCount; block++)
            {
                float maxAbs = 0;
                for (int i = 0; i < Q8_0GroupSize; i++)
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(src[block * Q8_0GroupSize + i]));

                float maxError = maxAbs / 127.0f * 1.5f; // 1.5× quantization step for rounding + Half precision
                for (int i = 0; i < Q8_0GroupSize; i++)
                {
                    int idx = block * Q8_0GroupSize + i;
                    float absError = MathF.Abs(src[idx] - roundTrip[idx]);
                    Assert.True(absError <= maxError + 1e-6f,
                        $"Element {idx}: src={src[idx]}, roundTrip={roundTrip[idx]}, absError={absError}, maxError={maxError}");
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    [Fact]
    public void QuantizeF32ToQ8_0_AllZeros_ProducesZeroScale()
    {
        const int k = 32;
        float[] src = new float[k]; // all zeros
        byte[] dest = new byte[Q8_0BlockBytes];

        fixed (float* srcPtr = src)
        fixed (byte* destPtr = dest)
        {
            MatMul.QuantizeF32ToQ8_0(srcPtr, destPtr, k);
            float scale = (float)Unsafe.ReadUnaligned<Half>(destPtr);
            Assert.Equal(0f, scale);
        }
    }

    // ──────────────────── Q8_0 GEMV (full pipeline) ────────────────────

    [Fact]
    public void GemvQ8_0_KnownOutput()
    {
        // Create a simple 2×32 weight matrix in Q8_0 and a f32 input vector.
        const int m = 2, k = 32;
        int blockCount = k / Q8_0GroupSize;
        int rowBytes = blockCount * Q8_0BlockBytes;

        nint weightsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        try
        {
            // Row 0: scale=1.0, qs=[1,1,...,1] → dot with x should be sum(x[i])
            byte* row0 = (byte*)weightsPtr;
            *(Half*)row0 = (Half)1.0f;
            for (int i = 0; i < Q8_0GroupSize; i++)
                ((sbyte*)(row0 + 2))[i] = 1;

            // Row 1: scale=0.5, qs=[2,2,...,2] → effective weight = 1.0
            byte* row1 = (byte*)weightsPtr + rowBytes;
            *(Half*)row1 = (Half)0.5f;
            for (int i = 0; i < Q8_0GroupSize; i++)
                ((sbyte*)(row1 + 2))[i] = 2;

            float[] x = new float[k];
            for (int i = 0; i < k; i++)
                x[i] = 1.0f;

            float[] result = new float[m];

            fixed (float* xp = x, rp = result)
                MatMul.GemvQ8_0((byte*)weightsPtr, xp, rp, m, k);

            // Both rows have effective weight 1.0, x is all 1.0 → dot ≈ 32
            // Allow quantization error.
            Assert.Equal(32f, result[0], 2f);
            Assert.Equal(32f, result[1], 2f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    [Fact]
    public void GemvQ8_0_LargeK_UsesArrayPool()
    {
        // K > StackAllocThreshold / Q8_0BlockBytes * Q8_0GroupSize to force ArrayPool path.
        const int k = 4096; // 4096/32 = 128 blocks × 34 = 4352 bytes > threshold for some configs
        const int m = 4;
        int blockCount = k / Q8_0GroupSize;
        int rowBytes = blockCount * Q8_0BlockBytes;

        nint weightsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        try
        {
            var rng = new Random(42);
            for (int row = 0; row < m; row++)
                FillRandomQ8_0Blocks((byte*)weightsPtr + row * rowBytes, blockCount, rng);

            float[] x = new float[k];
            for (int i = 0; i < k; i++)
                x[i] = rng.NextSingle() * 2f - 1f;

            float[] result = new float[m];

            fixed (float* xp = x, rp = result)
                MatMul.GemvQ8_0((byte*)weightsPtr, xp, rp, m, k);

            // Just verify finite results.
            for (int i = 0; i < m; i++)
                Assert.True(float.IsFinite(result[i]), $"result[{i}] = {result[i]}");
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    [Fact]
    public void GemvQ8_0_NonAlignedK_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
        {
            fixed (float* dummy = new float[33])
                MatMul.GemvQ8_0(null, dummy, dummy, 1, 33);
        });
    }

    // ──────────────────── Helpers ────────────────────

    private static nint AllocQ8_0Block(Half scale, Func<int, sbyte> fillQs)
    {
        nint ptr = (nint)NativeMemory.AlignedAlloc(Q8_0BlockBytes, 32);
        byte* p = (byte*)ptr;
        *(Half*)p = scale;
        for (int i = 0; i < Q8_0GroupSize; i++)
            ((sbyte*)(p + 2))[i] = fillQs(i);
        return ptr;
    }

    private static void FillRandomQ8_0Blocks(byte* ptr, int blockCount, Random rng)
    {
        for (int b = 0; b < blockCount; b++)
        {
            byte* block = ptr + b * Q8_0BlockBytes;
            *(Half*)block = (Half)(rng.NextSingle() * 0.1f);
            for (int i = 0; i < Q8_0GroupSize; i++)
                ((sbyte*)(block + 2))[i] = (sbyte)rng.Next(-127, 128);
        }
    }
}
