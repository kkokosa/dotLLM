using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Verifies that multi-threaded MatMul overloads produce bit-identical results to single-threaded.
/// </summary>
public sealed unsafe class MatMulParallelTests : IDisposable
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    private readonly ComputeThreadPool _pool = new(4);

    public void Dispose() => _pool.Dispose();

    // ──────────────────── GEMV Q8_0 ────────────────────

    [Theory]
    [InlineData(64, 128)]
    [InlineData(256, 4096)]
    [InlineData(4096, 4096)]
    [InlineData(33, 64)] // non-aligned M (not multiple of 4)
    public void GemvQ8_0_Parallel_MatchesSingleThreaded(int m, int k)
    {
        var rng = new Random(42);
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;

        byte* weights = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(k * sizeof(float)), 64);
        float* resultST = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);
        float* resultMT = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);

        try
        {
            FillQ8Weights(weights, m, blocksPerRow, rng);
            FillFloats(x, k, rng);

            MatMul.GemvQ8_0(weights, x, resultST, m, k);
            MatMul.GemvQ8_0(weights, x, resultMT, m, k, _pool);

            AssertBitIdentical(resultST, resultMT, m);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(resultST);
            NativeMemory.AlignedFree(resultMT);
        }
    }

    // ──────────────────── GEMV F32 ────────────────────

    [Theory]
    [InlineData(64, 128)]
    [InlineData(256, 4096)]
    [InlineData(4096, 4096)]
    public void GemvF32_Parallel_MatchesSingleThreaded(int m, int k)
    {
        var rng = new Random(42);
        float* a = (float*)NativeMemory.AlignedAlloc((nuint)((long)m * k * sizeof(float)), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(k * sizeof(float)), 64);
        float* resultST = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);
        float* resultMT = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);

        try
        {
            FillFloats(a, m * k, rng);
            FillFloats(x, k, rng);

            MatMul.GemvF32(a, x, resultST, m, k);
            MatMul.GemvF32(a, x, resultMT, m, k, _pool);

            AssertBitIdentical(resultST, resultMT, m);
        }
        finally
        {
            NativeMemory.AlignedFree(a);
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(resultST);
            NativeMemory.AlignedFree(resultMT);
        }
    }

    // ──────────────────── GEMV F16 ────────────────────

    [Theory]
    [InlineData(64, 128)]
    [InlineData(256, 4096)]
    public void GemvF16_Parallel_MatchesSingleThreaded(int m, int k)
    {
        var rng = new Random(42);
        Half* weights = (Half*)NativeMemory.AlignedAlloc((nuint)((long)m * k * sizeof(Half)), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(k * sizeof(float)), 64);
        float* resultST = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);
        float* resultMT = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);

        try
        {
            for (long i = 0; i < (long)m * k; i++)
                weights[i] = (Half)(rng.NextSingle() * 2f - 1f);
            FillFloats(x, k, rng);

            MatMul.GemvF16((nint)weights, x, resultST, m, k);
            MatMul.GemvF16((nint)weights, x, resultMT, m, k, _pool);

            AssertBitIdentical(resultST, resultMT, m);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(resultST);
            NativeMemory.AlignedFree(resultMT);
        }
    }

    // ──────────────────── GEMM Q8_0 ────────────────────

    [Theory]
    [InlineData(64, 128, 4)]
    [InlineData(256, 4096, 8)]
    [InlineData(4096, 4096, 16)]
    public void GemmQ8_0_Parallel_MatchesSingleThreaded(int m, int k, int n)
    {
        var rng = new Random(42);
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;

        byte* weights = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * rowBytes), 64);
        float* b = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * k * sizeof(float)), 64);
        float* cST = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * sizeof(float)), 64);
        float* cMT = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * sizeof(float)), 64);

        try
        {
            FillQ8Weights(weights, m, blocksPerRow, rng);
            FillFloats(b, n * k, rng);

            MatMul.GemmQ8_0(weights, b, cST, m, k, n);
            MatMul.GemmQ8_0(weights, b, cMT, m, k, n, _pool);

            AssertBitIdentical(cST, cMT, n * m);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            NativeMemory.AlignedFree(b);
            NativeMemory.AlignedFree(cST);
            NativeMemory.AlignedFree(cMT);
        }
    }

    // ──────────────────── GEMM F32 ────────────────────

    [Theory]
    [InlineData(64, 128, 4)]
    [InlineData(256, 4096, 8)]
    public void GemmF32_Parallel_MatchesSingleThreaded(int m, int k, int n)
    {
        var rng = new Random(42);
        float* a = (float*)NativeMemory.AlignedAlloc((nuint)((long)m * k * sizeof(float)), 64);
        float* b = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * k * sizeof(float)), 64);
        float* cST = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * sizeof(float)), 64);
        float* cMT = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * sizeof(float)), 64);

        try
        {
            FillFloats(a, m * k, rng);
            FillFloats(b, n * k, rng);

            MatMul.GemmF32(a, b, cST, m, k, n);
            MatMul.GemmF32(a, b, cMT, m, k, n, _pool);

            AssertBitIdentical(cST, cMT, n * m);
        }
        finally
        {
            NativeMemory.AlignedFree(a);
            NativeMemory.AlignedFree(b);
            NativeMemory.AlignedFree(cST);
            NativeMemory.AlignedFree(cMT);
        }
    }

    // ──────────────────── GEMM F16 ────────────────────

    [Theory]
    [InlineData(64, 128, 4)]
    [InlineData(256, 4096, 8)]
    public void GemmF16_Parallel_MatchesSingleThreaded(int m, int k, int n)
    {
        var rng = new Random(42);
        Half* weights = (Half*)NativeMemory.AlignedAlloc((nuint)((long)m * k * sizeof(Half)), 64);
        float* b = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * k * sizeof(float)), 64);
        float* cST = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * sizeof(float)), 64);
        float* cMT = (float*)NativeMemory.AlignedAlloc((nuint)((long)n * m * sizeof(float)), 64);

        try
        {
            for (long i = 0; i < (long)m * k; i++)
                weights[i] = (Half)(rng.NextSingle() * 2f - 1f);
            FillFloats(b, n * k, rng);

            MatMul.GemmF16((nint)weights, b, cST, m, k, n);
            MatMul.GemmF16((nint)weights, b, cMT, m, k, n, _pool);

            AssertBitIdentical(cST, cMT, n * m);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            NativeMemory.AlignedFree(b);
            NativeMemory.AlignedFree(cST);
            NativeMemory.AlignedFree(cMT);
        }
    }

    // ──────────────────── ComputeRows parallel ────────────────────

    [Theory]
    [InlineData(64, 128)]
    [InlineData(4096, 4096)]
    public void ComputeRows_Parallel_MatchesSingleThreaded(int m, int k)
    {
        var rng = new Random(42);
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;

        byte* weights = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        byte* xQ8 = (byte*)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
        float* resultST = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);
        float* resultMT = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);

        try
        {
            FillQ8Weights(weights, m, blocksPerRow, rng);
            FillQ8Row(xQ8, blocksPerRow, rng);

            MatMul.ComputeRows(weights, xQ8, resultST, m, blocksPerRow);
            MatMul.ComputeRows(weights, xQ8, resultMT, m, blocksPerRow, _pool);

            AssertBitIdentical(resultST, resultMT, m);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            NativeMemory.AlignedFree(xQ8);
            NativeMemory.AlignedFree(resultST);
            NativeMemory.AlignedFree(resultMT);
        }
    }

    // ──────────────────── Below threshold: no dispatch overhead ────────────────────

    [Fact]
    public void GemvQ8_0_BelowThreshold_FallsBackToSingleThreaded()
    {
        // m=16 < ParallelMinRows=32, should produce correct results via single-threaded fallback
        int m = 16, k = 64;
        var rng = new Random(42);
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;

        byte* weights = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(k * sizeof(float)), 64);
        float* resultST = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);
        float* resultMT = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);

        try
        {
            FillQ8Weights(weights, m, blocksPerRow, rng);
            FillFloats(x, k, rng);

            MatMul.GemvQ8_0(weights, x, resultST, m, k);
            MatMul.GemvQ8_0(weights, x, resultMT, m, k, _pool);

            AssertBitIdentical(resultST, resultMT, m);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(resultST);
            NativeMemory.AlignedFree(resultMT);
        }
    }

    [Fact]
    public void NullPool_ProducesSameResultAsSingleThreaded()
    {
        int m = 256, k = 128;
        var rng = new Random(42);
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;

        byte* weights = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        float* x = (float*)NativeMemory.AlignedAlloc((nuint)(k * sizeof(float)), 64);
        float* resultST = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);
        float* resultNull = (float*)NativeMemory.AlignedAlloc((nuint)(m * sizeof(float)), 64);

        try
        {
            FillQ8Weights(weights, m, blocksPerRow, rng);
            FillFloats(x, k, rng);

            MatMul.GemvQ8_0(weights, x, resultST, m, k);
            MatMul.GemvQ8_0(weights, x, resultNull, m, k, null);

            AssertBitIdentical(resultST, resultNull, m);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(resultST);
            NativeMemory.AlignedFree(resultNull);
        }
    }

    // ──────────────────── Helpers ────────────────────

    private static void FillFloats(float* ptr, int count, Random rng)
    {
        for (int i = 0; i < count; i++)
            ptr[i] = rng.NextSingle() * 2f - 1f;
    }

    private static void FillQ8Weights(byte* ptr, int rows, int blocksPerRow, Random rng)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* block = ptr + (row * blocksPerRow + b) * Q8_0BlockBytes;
                *(Half*)block = (Half)(rng.NextSingle() * 0.1f);
                for (int i = 0; i < Q8_0GroupSize; i++)
                    ((sbyte*)(block + 2))[i] = (sbyte)rng.Next(-127, 128);
            }
        }
    }

    private static void FillQ8Row(byte* ptr, int blocksPerRow, Random rng)
    {
        for (int b = 0; b < blocksPerRow; b++)
        {
            byte* block = ptr + b * Q8_0BlockBytes;
            *(Half*)block = (Half)(rng.NextSingle() * 0.1f);
            for (int i = 0; i < Q8_0GroupSize; i++)
                ((sbyte*)(block + 2))[i] = (sbyte)rng.Next(-127, 128);
        }
    }

    private static void AssertBitIdentical(float* expected, float* actual, int count)
    {
        for (int i = 0; i < count; i++)
        {
            Assert.True(
                BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"Mismatch at index {i}: expected {expected[i]:G9}, actual {actual[i]:G9}");
        }
    }
}
