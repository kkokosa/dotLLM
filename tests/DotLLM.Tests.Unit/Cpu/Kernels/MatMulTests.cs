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

    // ──────────────────── Q8_0 VecDot (scalar) ────────────────────

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

    // ──────────────────── VecDot optimized AVX2 accuracy ────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(128)]
    [InlineData(344)]
    public void VecDotQ8_0_OptimizedAvx2_MatchesScalar(int blockCount)
    {
        if (!Avx2.IsSupported)
            return;

        var rng = new Random(42);
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

    // ──────────────────── VecDot AVX-512 accuracy ────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(128)]
    [InlineData(344)]
    public void VecDotQ8_0_Avx512_MatchesScalar(int blockCount)
    {
        if (!Avx512BW.IsSupported)
            return;

        var rng = new Random(42);
        nuint totalBytes = (nuint)(blockCount * Q8_0BlockBytes);

        nint aPtr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        nint bPtr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            FillRandomQ8_0Blocks((byte*)aPtr, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)bPtr, blockCount, rng);

            float scalar = MatMul.VecDotQ8_0Scalar((byte*)aPtr, (byte*)bPtr, blockCount);
            float avx512 = MatMul.VecDotQ8_0Avx512((byte*)aPtr, (byte*)bPtr, blockCount);

            Assert.Equal(scalar, avx512, 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)aPtr);
            NativeMemory.AlignedFree((void*)bPtr);
        }
    }

    // ──────────────────── Multi-row (4-row) accuracy ────────────────────

    [Theory]
    [InlineData(16)]
    [InlineData(128)]
    [InlineData(344)]
    public void VecDotQ8_0_4Row_MatchesSingleRow(int blockCount)
    {
        if (!Avx2.IsSupported)
            return;

        var rng = new Random(42);
        nuint rowBytes = (nuint)(blockCount * Q8_0BlockBytes);

        nint w0 = (nint)NativeMemory.AlignedAlloc(rowBytes, 64);
        nint w1 = (nint)NativeMemory.AlignedAlloc(rowBytes, 64);
        nint w2 = (nint)NativeMemory.AlignedAlloc(rowBytes, 64);
        nint w3 = (nint)NativeMemory.AlignedAlloc(rowBytes, 64);
        nint x = (nint)NativeMemory.AlignedAlloc(rowBytes, 64);
        try
        {
            FillRandomQ8_0Blocks((byte*)w0, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)w1, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)w2, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)w3, blockCount, rng);
            FillRandomQ8_0Blocks((byte*)x, blockCount, rng);

            // Single-row reference.
            float r0 = MatMul.VecDotQ8_0Avx2((byte*)w0, (byte*)x, blockCount);
            float r1 = MatMul.VecDotQ8_0Avx2((byte*)w1, (byte*)x, blockCount);
            float r2 = MatMul.VecDotQ8_0Avx2((byte*)w2, (byte*)x, blockCount);
            float r3 = MatMul.VecDotQ8_0Avx2((byte*)w3, (byte*)x, blockCount);

            // Multi-row batched.
            float* results = stackalloc float[4];
            MatMul.VecDotQ8_0Avx2_4Rows((byte*)w0, (byte*)w1, (byte*)w2, (byte*)w3,
                (byte*)x, blockCount, results);

            Assert.Equal(r0, results[0], 1e-2f);
            Assert.Equal(r1, results[1], 1e-2f);
            Assert.Equal(r2, results[2], 1e-2f);
            Assert.Equal(r3, results[3], 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)w0);
            NativeMemory.AlignedFree((void*)w1);
            NativeMemory.AlignedFree((void*)w2);
            NativeMemory.AlignedFree((void*)w3);
            NativeMemory.AlignedFree((void*)x);
        }
    }

    // ──────────────────── VecDot edge cases ────────────────────

    [Fact]
    public void VecDotQ8_0_AllZeroScales()
    {
        if (!Avx2.IsSupported)
            return;

        const int blockCount = 4;
        nuint totalBytes = (nuint)(blockCount * Q8_0BlockBytes);

        nint aPtr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        nint bPtr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            // Fill with non-zero qs but zero scales.
            for (int b = 0; b < blockCount; b++)
            {
                byte* aBlock = (byte*)aPtr + b * Q8_0BlockBytes;
                byte* bBlock = (byte*)bPtr + b * Q8_0BlockBytes;
                *(Half*)aBlock = (Half)0.0f;
                *(Half*)bBlock = (Half)0.0f;
                for (int i = 0; i < Q8_0GroupSize; i++)
                {
                    ((sbyte*)(aBlock + 2))[i] = 127;
                    ((sbyte*)(bBlock + 2))[i] = -127;
                }
            }

            float scalar = MatMul.VecDotQ8_0Scalar((byte*)aPtr, (byte*)bPtr, blockCount);
            float avx2 = MatMul.VecDotQ8_0Avx2((byte*)aPtr, (byte*)bPtr, blockCount);

            Assert.Equal(0f, scalar);
            Assert.Equal(0f, avx2);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)aPtr);
            NativeMemory.AlignedFree((void*)bPtr);
        }
    }

    [Fact]
    public void VecDotQ8_0_MaxValues()
    {
        if (!Avx2.IsSupported)
            return;

        const int blockCount = 4;
        nuint totalBytes = (nuint)(blockCount * Q8_0BlockBytes);

        nint aPtr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        nint bPtr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            for (int b = 0; b < blockCount; b++)
            {
                byte* aBlock = (byte*)aPtr + b * Q8_0BlockBytes;
                byte* bBlock = (byte*)bPtr + b * Q8_0BlockBytes;
                *(Half*)aBlock = (Half)1.0f;
                *(Half*)bBlock = (Half)1.0f;
                for (int i = 0; i < Q8_0GroupSize; i++)
                {
                    ((sbyte*)(aBlock + 2))[i] = 127;
                    ((sbyte*)(bBlock + 2))[i] = (sbyte)(i % 2 == 0 ? 127 : -127);
                }
            }

            float scalar = MatMul.VecDotQ8_0Scalar((byte*)aPtr, (byte*)bPtr, blockCount);
            float avx2 = MatMul.VecDotQ8_0Avx2((byte*)aPtr, (byte*)bPtr, blockCount);

            Assert.True(float.IsFinite(scalar));
            Assert.Equal(scalar, avx2, 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)aPtr);
            NativeMemory.AlignedFree((void*)bPtr);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    public void VecDotQ8_0_OddBlockCount(int blockCount)
    {
        if (!Avx2.IsSupported)
            return;

        var rng = new Random(42);
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

            if (Avx512BW.IsSupported)
            {
                float avx512 = MatMul.VecDotQ8_0Avx512((byte*)aPtr, (byte*)bPtr, blockCount);
                Assert.Equal(scalar, avx512, 1e-2f);
            }
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

    [Fact]
    public void QuantizeF32ToQ8_0_Avx2_MatchesScalar()
    {
        if (!Avx2.IsSupported)
            return;

        var rng = new Random(42);
        const int k = 1024; // 32 blocks
        float[] src = new float[k];
        for (int i = 0; i < k; i++)
            src[i] = rng.NextSingle() * 2f - 1f;

        int q8Bytes = (k / Q8_0GroupSize) * Q8_0BlockBytes;
        nint scalarPtr = (nint)NativeMemory.AlignedAlloc((nuint)q8Bytes, 64);
        nint avx2Ptr = (nint)NativeMemory.AlignedAlloc((nuint)q8Bytes, 64);
        try
        {
            fixed (float* srcPtr = src)
            {
                MatMul.QuantizeF32ToQ8_0Scalar(srcPtr, (byte*)scalarPtr, k);
                MatMul.QuantizeF32ToQ8_0Avx2(srcPtr, (byte*)avx2Ptr, k);
            }

            // Byte-for-byte comparison.
            var scalarSpan = new ReadOnlySpan<byte>((void*)scalarPtr, q8Bytes);
            var avx2Span = new ReadOnlySpan<byte>((void*)avx2Ptr, q8Bytes);

            Assert.True(scalarSpan.SequenceEqual(avx2Span),
                "AVX2 quantization output differs from scalar reference");
        }
        finally
        {
            NativeMemory.AlignedFree((void*)scalarPtr);
            NativeMemory.AlignedFree((void*)avx2Ptr);
        }
    }

    [Fact]
    public void QuantizeF32ToQ8_0_LargeValues()
    {
        const int k = 32;
        float[] src = new float[k];
        // Values at the edge of practical range but within Half precision.
        // Half max is ~65504, scale = max/127 ≈ 515 which fits in Half.
        for (int i = 0; i < k; i++)
            src[i] = (i % 2 == 0 ? 500.0f : -500.0f);

        int q8Bytes = Q8_0BlockBytes;
        nint q8Ptr = (nint)NativeMemory.AlignedAlloc((nuint)q8Bytes, 64);
        try
        {
            fixed (float* srcPtr = src)
                MatMul.QuantizeF32ToQ8_0(srcPtr, (byte*)q8Ptr, k);

            byte* block = (byte*)q8Ptr;
            float scale = (float)Unsafe.ReadUnaligned<Half>(block);
            Assert.True(float.IsFinite(scale), $"Scale should be finite, got {scale}");

            // All values have the same absolute magnitude → all qs should be ±127.
            sbyte* qs = (sbyte*)(block + 2);
            for (int i = 0; i < k; i++)
            {
                sbyte expected = (sbyte)(i % 2 == 0 ? 127 : -127);
                Assert.Equal(expected, qs[i]);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)q8Ptr);
        }
    }

    [Fact]
    public void QuantizeF32ToQ8_0_AllSameSign()
    {
        const int k = 32;
        float[] src = new float[k];
        for (int i = 0; i < k; i++)
            src[i] = (i + 1) * 0.1f; // all positive

        int q8Bytes = Q8_0BlockBytes;
        nint scalarPtr = (nint)NativeMemory.AlignedAlloc((nuint)q8Bytes, 64);
        nint simdPtr = (nint)NativeMemory.AlignedAlloc((nuint)q8Bytes, 64);
        try
        {
            fixed (float* srcPtr = src)
            {
                MatMul.QuantizeF32ToQ8_0Scalar(srcPtr, (byte*)scalarPtr, k);
                MatMul.QuantizeF32ToQ8_0(srcPtr, (byte*)simdPtr, k);
            }

            var scalarSpan = new ReadOnlySpan<byte>((void*)scalarPtr, q8Bytes);
            var simdSpan = new ReadOnlySpan<byte>((void*)simdPtr, q8Bytes);

            Assert.True(scalarSpan.SequenceEqual(simdSpan),
                "SIMD quantization output differs from scalar for all-positive input");
        }
        finally
        {
            NativeMemory.AlignedFree((void*)scalarPtr);
            NativeMemory.AlignedFree((void*)simdPtr);
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

    [Fact]
    public void GemvQ8_0_LargeM_MultiRowPath()
    {
        // M=100, K=4096 exercises 4-row loop (25 iterations) + remainder.
        const int m = 100, k = 4096;
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

            for (int i = 0; i < m; i++)
                Assert.True(float.IsFinite(result[i]), $"result[{i}] = {result[i]}");

            // Verify against scalar VecDot for each row.
            int xQ8Bytes = blockCount * Q8_0BlockBytes;
            nint xQ8Ptr = (nint)NativeMemory.AlignedAlloc((nuint)xQ8Bytes, 64);
            try
            {
                fixed (float* xp = x)
                    MatMul.QuantizeF32ToQ8_0(xp, (byte*)xQ8Ptr, k);

                for (int row = 0; row < m; row++)
                {
                    float scalarResult = MatMul.VecDotQ8_0Scalar(
                        (byte*)weightsPtr + row * rowBytes, (byte*)xQ8Ptr, blockCount);
                    Assert.Equal(scalarResult, result[row], 1e-2f);
                }
            }
            finally
            {
                NativeMemory.AlignedFree((void*)xQ8Ptr);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    [Fact]
    public void GemvQ8_0_OptimizedMatchesOriginalScalar()
    {
        // M=64, K=4096 — compare full GEMV against pure scalar path.
        const int m = 64, k = 4096;
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

            // Quantize x via scalar for reference.
            int xQ8Bytes = blockCount * Q8_0BlockBytes;
            nint xQ8Ptr = (nint)NativeMemory.AlignedAlloc((nuint)xQ8Bytes, 64);
            try
            {
                fixed (float* xp = x)
                    MatMul.QuantizeF32ToQ8_0Scalar(xp, (byte*)xQ8Ptr, k);

                float[] scalarResults = new float[m];
                for (int row = 0; row < m; row++)
                {
                    scalarResults[row] = MatMul.VecDotQ8_0Scalar(
                        (byte*)weightsPtr + row * rowBytes, (byte*)xQ8Ptr, blockCount);
                }

                float[] optimizedResults = new float[m];
                fixed (float* xp = x, rp = optimizedResults)
                    MatMul.GemvQ8_0((byte*)weightsPtr, xp, rp, m, k);

                for (int i = 0; i < m; i++)
                    Assert.Equal(scalarResults[i], optimizedResults[i], 1e-2f);
            }
            finally
            {
                NativeMemory.AlignedFree((void*)xQ8Ptr);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    // ──────────────────── GEMM ────────────────────

    [Fact]
    public void GemmF32Scalar_2x2x2_KnownValues()
    {
        // A (weights) = [[1,2],[3,4]]  (2×2, M=2, K=2)
        // B (inputs)  = [[5,6],[7,8]]  (2×2, N=2, K=2)
        // C[t,r] = dot(A[r,:], B[t,:])
        // C[0,0] = 1*5+2*6 = 17, C[0,1] = 3*5+4*6 = 39
        // C[1,0] = 1*7+2*8 = 23, C[1,1] = 3*7+4*8 = 53
        float[] a = [1, 2, 3, 4];
        float[] b = [5, 6, 7, 8];
        float[] c = new float[4];

        fixed (float* ap = a, bp = b, cp = c)
            MatMul.GemmF32Scalar(ap, bp, cp, 2, 2, 2);

        Assert.Equal(17f, c[0], 1e-5f);
        Assert.Equal(39f, c[1], 1e-5f);
        Assert.Equal(23f, c[2], 1e-5f);
        Assert.Equal(53f, c[3], 1e-5f);
    }

    [Fact]
    public void GemmF32_MatchesSequentialGemv()
    {
        var rng = new Random(42);
        const int m = 16, k = 64, n = 8;
        float[] a = new float[m * k];
        float[] b = new float[n * k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

        // Sequential GEMV per token
        float[] seqResult = new float[n * m];
        fixed (float* ap = a, bp = b, sp = seqResult)
        {
            for (int t = 0; t < n; t++)
                MatMul.GemvF32(ap, bp + t * k, sp + t * m, m, k);
        }

        // Batched GEMM
        float[] gemmResult = new float[n * m];
        fixed (float* ap = a, bp = b, gp = gemmResult)
            MatMul.GemmF32(ap, bp, gp, m, k, n);

        for (int i = 0; i < n * m; i++)
            Assert.Equal(seqResult[i], gemmResult[i], 1e-3f);
    }

    [Fact]
    public void GemmF32_N1_MatchesGemv()
    {
        var rng = new Random(42);
        const int m = 16, k = 64;
        float[] a = new float[m * k];
        float[] x = new float[k];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        float[] gemvResult = new float[m];
        float[] gemmResult = new float[m];

        fixed (float* ap = a, xp = x, gr = gemvResult, gm = gemmResult)
        {
            MatMul.GemvF32(ap, xp, gr, m, k);
            MatMul.GemmF32(ap, xp, gm, m, k, 1);
        }

        for (int i = 0; i < m; i++)
            Assert.Equal(gemvResult[i], gemmResult[i], 1e-5f);
    }

    [Fact]
    public void GemmQ8_0_MatchesSequentialGemvQ8_0()
    {
        var rng = new Random(42);
        const int m = 16, k = 256, n = 8;
        int blockCount = k / Q8_0GroupSize;
        int rowBytes = blockCount * Q8_0BlockBytes;

        nint weightsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        try
        {
            for (int row = 0; row < m; row++)
                FillRandomQ8_0Blocks((byte*)weightsPtr + row * rowBytes, blockCount, rng);

            float[] b = new float[n * k];
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

            // Sequential GEMV per token
            float[] seqResult = new float[n * m];
            fixed (float* bp = b, sp = seqResult)
            {
                for (int t = 0; t < n; t++)
                    MatMul.GemvQ8_0((byte*)weightsPtr, bp + t * k, sp + t * m, m, k);
            }

            // Batched GEMM
            float[] gemmResult = new float[n * m];
            fixed (float* bp = b, gp = gemmResult)
                MatMul.GemmQ8_0((byte*)weightsPtr, bp, gp, m, k, n);

            for (int i = 0; i < n * m; i++)
                Assert.Equal(seqResult[i], gemmResult[i], 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(128)]
    public void GemmQ8_0_VaryingN_MatchesSequentialGemv(int n)
    {
        var rng = new Random(42);
        const int m = 8, k = 256;
        int blockCount = k / Q8_0GroupSize;
        int rowBytes = blockCount * Q8_0BlockBytes;

        nint weightsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        try
        {
            for (int row = 0; row < m; row++)
                FillRandomQ8_0Blocks((byte*)weightsPtr + row * rowBytes, blockCount, rng);

            float[] b = new float[n * k];
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

            // Sequential GEMV
            float[] seqResult = new float[n * m];
            fixed (float* bp = b, sp = seqResult)
            {
                for (int t = 0; t < n; t++)
                    MatMul.GemvQ8_0((byte*)weightsPtr, bp + t * k, sp + t * m, m, k);
            }

            // Batched GEMM
            float[] gemmResult = new float[n * m];
            fixed (float* bp = b, gp = gemmResult)
                MatMul.GemmQ8_0((byte*)weightsPtr, bp, gp, m, k, n);

            for (int i = 0; i < n * m; i++)
                Assert.Equal(seqResult[i], gemmResult[i], 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    [Fact]
    public void GemmQ8_0_WithPreQuantized_MatchesWithout()
    {
        var rng = new Random(42);
        const int m = 8, k = 256, n = 4;
        int blockCount = k / Q8_0GroupSize;
        int rowBytes = blockCount * Q8_0BlockBytes;
        int q8RowBytes = blockCount * Q8_0BlockBytes;

        nint weightsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        nint scratchPtr = (nint)NativeMemory.AlignedAlloc((nuint)(n * q8RowBytes), 64);
        try
        {
            for (int row = 0; row < m; row++)
                FillRandomQ8_0Blocks((byte*)weightsPtr + row * rowBytes, blockCount, rng);

            float[] b = new float[n * k];
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

            // Pre-quantize input
            fixed (float* bp = b)
            {
                for (int t = 0; t < n; t++)
                    MatMul.QuantizeF32ToQ8_0(bp + t * k, (byte*)scratchPtr + t * q8RowBytes, k);
            }

            // GEMM without pre-quantized
            float[] resultNoPq = new float[n * m];
            fixed (float* bp = b, rp = resultNoPq)
                MatMul.GemmQ8_0((byte*)weightsPtr, bp, rp, m, k, n);

            // GEMM with pre-quantized
            float[] resultPq = new float[n * m];
            fixed (float* bp = b, rp = resultPq)
                MatMul.GemmQ8_0((byte*)weightsPtr, bp, rp, m, k, n, (byte*)scratchPtr);

            for (int i = 0; i < n * m; i++)
                Assert.Equal(resultNoPq[i], resultPq[i], 1e-5f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
            NativeMemory.AlignedFree((void*)scratchPtr);
        }
    }

    [Fact]
    public void GemmF16_MatchesSequentialGemvF16()
    {
        var rng = new Random(42);
        const int m = 8, k = 64, n = 4;

        nint weightsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(m * k * sizeof(ushort)), 64);
        try
        {
            Half* wh = (Half*)weightsPtr;
            for (int i = 0; i < m * k; i++)
                wh[i] = (Half)(rng.NextSingle() * 2f - 1f);

            float[] b = new float[n * k];
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;

            // Sequential GEMV
            float[] seqResult = new float[n * m];
            fixed (float* bp = b, sp = seqResult)
            {
                for (int t = 0; t < n; t++)
                    MatMul.GemvF16(weightsPtr, bp + t * k, sp + t * m, m, k);
            }

            // Batched GEMM
            float[] gemmResult = new float[n * m];
            fixed (float* bp = b, gp = gemmResult)
                MatMul.GemmF16(weightsPtr, bp, gp, m, k, n);

            for (int i = 0; i < n * m; i++)
                Assert.Equal(seqResult[i], gemmResult[i], 1e-3f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)weightsPtr);
        }
    }

    [Fact]
    public void GemmQ8_0_NonAlignedK_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
        {
            fixed (float* dummy = new float[33])
                MatMul.GemmQ8_0(null, dummy, dummy, 1, 33, 1);
        });
    }

    // ──────────────────── Q4_K stub ────────────────────

    [Fact]
    public void VecDotQ4_K_Q8_0Scalar_ThrowsNotImplemented()
    {
        Assert.Throws<NotImplementedException>(() =>
            MatMul.VecDotQ4_K_Q8_0Scalar(null, null, 1));
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
