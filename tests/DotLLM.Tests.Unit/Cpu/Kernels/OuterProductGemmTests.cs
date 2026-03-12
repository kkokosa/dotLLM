using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class OuterProductGemmTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    // ──────────────────── Scalar microkernel ────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(16)]
    [InlineData(18)]   // SmolLM-135M: 576/32 = 18
    public void OuterProductScalar_4x3_MatchesPerTokenComputeRows(int blockCount)
    {
        var rng = new Random(42);
        int m = 4;
        int n = 3;
        int rowBytes = blockCount * Q8_0BlockBytes;
        int groupBytes = 4 * rowBytes;

        // Allocate row-major weights (4 rows)
        byte* weights = (byte*)NativeMemory.AlignedAlloc((nuint)groupBytes, 64);
        byte*[] xPtrs = new byte*[n];
        for (int t = 0; t < n; t++)
            xPtrs[t] = (byte*)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);

        float* cOuter = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cRef = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            // Fill weights in row-major form, then repack to R4
            byte* rowMajor = (byte*)NativeMemory.AlignedAlloc((nuint)(4 * rowBytes), 64);
            for (int r = 0; r < 4; r++)
                FillRandomQ8_0Blocks(rowMajor + r * rowBytes, blockCount, rng);

            // Repack to R4: interleave blocks from 4 rows
            for (int b = 0; b < blockCount; b++)
            {
                for (int r = 0; r < 4; r++)
                {
                    Buffer.MemoryCopy(
                        rowMajor + r * rowBytes + b * Q8_0BlockBytes,
                        weights + b * 4 * Q8_0BlockBytes + r * Q8_0BlockBytes,
                        Q8_0BlockBytes, Q8_0BlockBytes);
                }
            }

            for (int t = 0; t < n; t++)
                FillRandomQ8_0Blocks(xPtrs[t], blockCount, rng);

            // Reference: per-token VecDotQ8_0ScalarR4
            for (int t = 0; t < n; t++)
            {
                for (int r = 0; r < 4; r++)
                {
                    cRef[t * m + r] = MatMul.VecDotQ8_0ScalarR4(weights, r, xPtrs[t], blockCount);
                }
            }

            // Test: outer-product scalar
            MatMul.OuterProductQ8_0Scalar_4x3(
                weights, xPtrs[0], xPtrs[1], xPtrs[2],
                cOuter, blockCount, m);

            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    Assert.Equal(cRef[t * m + r], cOuter[t * m + r], 1e-3f);

            NativeMemory.AlignedFree(rowMajor);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            for (int t = 0; t < n; t++)
                NativeMemory.AlignedFree(xPtrs[t]);
            NativeMemory.AlignedFree(cOuter);
            NativeMemory.AlignedFree(cRef);
        }
    }

    // ──────────────────── AVX2 microkernel ────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(16)]
    [InlineData(18)]   // SmolLM-135M: 576/32
    [InlineData(48)]   // 1536/32
    [InlineData(128)]  // 4096/32
    public void OuterProductAvx2_4x3_MatchesScalar(int blockCount)
    {
        if (!Avx2.IsSupported)
            return;

        var rng = new Random(42);
        int m = 4;
        int n = 3;
        int rowBytes = blockCount * Q8_0BlockBytes;

        // Allocate R4-interleaved weights
        byte* weights = AllocAndFillR4Weights(4, blockCount, rng);
        byte*[] xPtrs = new byte*[n];
        for (int t = 0; t < n; t++)
        {
            xPtrs[t] = (byte*)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
            FillRandomQ8_0Blocks(xPtrs[t], blockCount, rng);
        }

        float* cScalar = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cAvx2 = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            MatMul.OuterProductQ8_0Scalar_4x3(
                weights, xPtrs[0], xPtrs[1], xPtrs[2],
                cScalar, blockCount, m);

            MatMul.OuterProductQ8_0Avx2_4x3(
                weights, xPtrs[0], xPtrs[1], xPtrs[2],
                cAvx2, blockCount, m);

            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    Assert.Equal(cScalar[t * m + r], cAvx2[t * m + r], 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            for (int t = 0; t < n; t++)
                NativeMemory.AlignedFree(xPtrs[t]);
            NativeMemory.AlignedFree(cScalar);
            NativeMemory.AlignedFree(cAvx2);
        }
    }

    // ──────────────────── Full GEMM tests ────────────────────

    [Theory]
    [InlineData(4, 3, 64)]     // 1 full group, exactly 3 tokens, K=64
    [InlineData(8, 3, 64)]     // 2 full groups
    [InlineData(8, 6, 64)]     // 2 full groups, 6 tokens (multiple 3-tiles)
    [InlineData(8, 7, 64)]     // 2 full groups, 7 tokens (tail token)
    [InlineData(4, 1, 64)]     // 1 full group, 1 token (all tail tokens for outer-product)
    [InlineData(16, 9, 128)]   // 4 groups, 9 tokens
    public void OuterProductGemm_MatchesPerTokenInterleaved(int m, int n, int k)
    {
        var rng = new Random(42);
        int blockCount = k / Q8_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;
        int fullGroups = m / 4;
        int tailRows = m % 4;

        // Create row-major weights, then repack to R4
        byte* rowMajorWeights = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * q8RowBytes), 64);
        for (int r = 0; r < m; r++)
            FillRandomQ8_0Blocks(rowMajorWeights + r * q8RowBytes, blockCount, rng);

        using var repacked = WeightRepacking.RepackR4((nint)rowMajorWeights, QuantizationType.Q8_0, m, k);

        // Quantized inputs
        byte* inputQ8 = (byte*)NativeMemory.AlignedAlloc((nuint)((long)n * q8RowBytes), 64);
        for (int t = 0; t < n; t++)
            FillRandomQ8_0Blocks(inputQ8 + t * q8RowBytes, blockCount, rng);

        float* cOuter = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cRef = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            // Reference: per-token interleaved ComputeRows
            for (int t = 0; t < n; t++)
            {
                MatMul.ComputeRowsQ8_0Interleaved(
                    (byte*)repacked.Ptr, inputQ8 + t * q8RowBytes,
                    cRef + t * m, fullGroups, tailRows, blockCount);
            }

            // Test: outer-product GEMM
            MatMul.OuterProductGemmQ8_0(
                (byte*)repacked.Ptr, inputQ8, cOuter,
                fullGroups, tailRows, blockCount, m, n);

            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    Assert.Equal(cRef[t * m + r], cOuter[t * m + r], 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree(rowMajorWeights);
            NativeMemory.AlignedFree(inputQ8);
            NativeMemory.AlignedFree(cOuter);
            NativeMemory.AlignedFree(cRef);
        }
    }

    // ──────────────────── Tail handling ────────────────────

    [Theory]
    [InlineData(5, 3, 64)]    // 1 full group + 1 tail row
    [InlineData(6, 3, 64)]    // 1 full group + 2 tail rows
    [InlineData(7, 4, 64)]    // 1 full group + 3 tail rows, tail token
    [InlineData(9, 5, 128)]   // 2 full groups + 1 tail row, 2 tail tokens
    [InlineData(37, 7, 128)]  // 9 full groups + 1 tail, tail tokens
    public void OuterProductGemm_TailRowsAndTokens_Correct(int m, int n, int k)
    {
        var rng = new Random(42);
        int blockCount = k / Q8_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;
        int fullGroups = m / 4;
        int tailRows = m % 4;

        byte* rowMajorWeights = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * q8RowBytes), 64);
        for (int r = 0; r < m; r++)
            FillRandomQ8_0Blocks(rowMajorWeights + r * q8RowBytes, blockCount, rng);

        using var repacked = WeightRepacking.RepackR4((nint)rowMajorWeights, QuantizationType.Q8_0, m, k);

        byte* inputQ8 = (byte*)NativeMemory.AlignedAlloc((nuint)((long)n * q8RowBytes), 64);
        for (int t = 0; t < n; t++)
            FillRandomQ8_0Blocks(inputQ8 + t * q8RowBytes, blockCount, rng);

        float* cOuter = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cRef = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            for (int t = 0; t < n; t++)
            {
                MatMul.ComputeRowsQ8_0Interleaved(
                    (byte*)repacked.Ptr, inputQ8 + t * q8RowBytes,
                    cRef + t * m, fullGroups, tailRows, blockCount);
            }

            MatMul.OuterProductGemmQ8_0(
                (byte*)repacked.Ptr, inputQ8, cOuter,
                fullGroups, tailRows, blockCount, m, n);

            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    Assert.Equal(cRef[t * m + r], cOuter[t * m + r], 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree(rowMajorWeights);
            NativeMemory.AlignedFree(inputQ8);
            NativeMemory.AlignedFree(cOuter);
            NativeMemory.AlignedFree(cRef);
        }
    }

    // ──────────────────── Parallel vs single-threaded ────────────────────

    [Theory]
    [InlineData(16, 6, 128)]   // 4 groups, 6 tokens
    [InlineData(37, 7, 128)]   // 9 groups + tail, tail tokens
    [InlineData(64, 9, 64)]    // 16 groups, 9 tokens
    public void OuterProductGemm_Parallel_MatchesSingleThreaded(int m, int n, int k)
    {
        var rng = new Random(42);
        int blockCount = k / Q8_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;
        int fullGroups = m / 4;
        int tailRows = m % 4;

        byte* rowMajorWeights = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * q8RowBytes), 64);
        for (int r = 0; r < m; r++)
            FillRandomQ8_0Blocks(rowMajorWeights + r * q8RowBytes, blockCount, rng);

        using var repacked = WeightRepacking.RepackR4((nint)rowMajorWeights, QuantizationType.Q8_0, m, k);

        byte* inputQ8 = (byte*)NativeMemory.AlignedAlloc((nuint)((long)n * q8RowBytes), 64);
        for (int t = 0; t < n; t++)
            FillRandomQ8_0Blocks(inputQ8 + t * q8RowBytes, blockCount, rng);

        float* cSingle = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cParallel = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            // Single-threaded
            MatMul.OuterProductGemmQ8_0(
                (byte*)repacked.Ptr, inputQ8, cSingle,
                fullGroups, tailRows, blockCount, m, n);

            // Parallel
            using var pool = new ComputeThreadPool(4);
            MatMul.OuterProductGemmQ8_0(
                (byte*)repacked.Ptr, inputQ8, cParallel,
                fullGroups, tailRows, blockCount, m, n, pool);

            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    Assert.Equal(cSingle[t * m + r], cParallel[t * m + r], 1e-4f);
        }
        finally
        {
            NativeMemory.AlignedFree(rowMajorWeights);
            NativeMemory.AlignedFree(inputQ8);
            NativeMemory.AlignedFree(cSingle);
            NativeMemory.AlignedFree(cParallel);
        }
    }

    // ──────────────────── Model-realistic dimensions ────────────────────

    [Theory]
    [InlineData(576, 3, 576)]     // SmolLM-135M: Q/K/V
    [InlineData(1536, 3, 576)]    // SmolLM-135M: gate/up
    [InlineData(576, 3, 1536)]    // SmolLM-135M: down
    [InlineData(576, 12, 576)]    // SmolLM-135M: longer prompt
    public void OuterProductGemm_RealisticDimensions_MatchesReference(int m, int n, int k)
    {
        var rng = new Random(42);
        int blockCount = k / Q8_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;
        int fullGroups = m / 4;
        int tailRows = m % 4;

        byte* rowMajorWeights = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * q8RowBytes), 64);
        for (int r = 0; r < m; r++)
            FillRandomQ8_0Blocks(rowMajorWeights + r * q8RowBytes, blockCount, rng);

        using var repacked = WeightRepacking.RepackR4((nint)rowMajorWeights, QuantizationType.Q8_0, m, k);

        byte* inputQ8 = (byte*)NativeMemory.AlignedAlloc((nuint)((long)n * q8RowBytes), 64);
        for (int t = 0; t < n; t++)
            FillRandomQ8_0Blocks(inputQ8 + t * q8RowBytes, blockCount, rng);

        float* cOuter = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cRef = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            for (int t = 0; t < n; t++)
            {
                MatMul.ComputeRowsQ8_0Interleaved(
                    (byte*)repacked.Ptr, inputQ8 + t * q8RowBytes,
                    cRef + t * m, fullGroups, tailRows, blockCount);
            }

            MatMul.OuterProductGemmQ8_0(
                (byte*)repacked.Ptr, inputQ8, cOuter,
                fullGroups, tailRows, blockCount, m, n);

            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    Assert.Equal(cRef[t * m + r], cOuter[t * m + r], 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree(rowMajorWeights);
            NativeMemory.AlignedFree(inputQ8);
            NativeMemory.AlignedFree(cOuter);
            NativeMemory.AlignedFree(cRef);
        }
    }

    // ──────────────────── Helpers ────────────────────

    /// <summary>
    /// Allocates and fills weights in R4-interleaved layout.
    /// </summary>
    private static byte* AllocAndFillR4Weights(int m, int blockCount, Random rng)
    {
        int rowBytes = blockCount * Q8_0BlockBytes;
        int groupBytes = 4 * rowBytes;
        int fullGroups = m / 4;

        // Fill row-major first
        byte* rowMajor = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * rowBytes), 64);
        for (int r = 0; r < m; r++)
            FillRandomQ8_0Blocks(rowMajor + r * rowBytes, blockCount, rng);

        // Repack to R4
        byte* r4 = (byte*)NativeMemory.AlignedAlloc((nuint)((long)fullGroups * groupBytes), 64);
        for (int g = 0; g < fullGroups; g++)
        {
            for (int b = 0; b < blockCount; b++)
            {
                for (int r = 0; r < 4; r++)
                {
                    Buffer.MemoryCopy(
                        rowMajor + (g * 4 + r) * rowBytes + b * Q8_0BlockBytes,
                        r4 + (long)g * groupBytes + b * 4 * Q8_0BlockBytes + r * Q8_0BlockBytes,
                        Q8_0BlockBytes, Q8_0BlockBytes);
                }
            }
        }

        NativeMemory.AlignedFree(rowMajor);
        return r4;
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
