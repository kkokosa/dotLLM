using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
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

    // ──────────────────── Tiled R4 GEMM ────────────────────

    /// <summary>
    /// GemmR4TiledQ8_0 must agree with the row-major tiled GEMM it is a candidate to replace
    /// for the prefill path. Covers M values with and without a 4-row remainder, and N values
    /// spanning single-token through a realistic prefill batch.
    /// </summary>
    [Theory]
    [InlineData(8, 1, 16)]
    [InlineData(8, 3, 16)]
    [InlineData(16, 7, 18)]    // SmolLM-135M: 576/32
    [InlineData(32, 16, 64)]   // Llama-3.2-1B: 2048/32
    [InlineData(37, 5, 18)]    // M not a multiple of 4 -> exercises the row-major tail
    [InlineData(130, 11, 48)]  // 1536/32, tail of 2 rows
    public void GemmR4Tiled_MatchesRowMajorGemm(int m, int n, int blockCount)
    {
        var rng = new Random(4242);
        int rowBytes = blockCount * Q8_0BlockBytes;
        int fullGroups = m / 4;
        int tailRows = m % 4;

        byte* rowMajor = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        byte* r4 = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        byte* input = (byte*)NativeMemory.AlignedAlloc((nuint)(n * rowBytes), 64);
        float* cR4 = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cRef = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            for (int r = 0; r < m; r++)
                FillRandomQ8_0Blocks(rowMajor + r * rowBytes, blockCount, rng);
            for (int t = 0; t < n; t++)
                FillRandomQ8_0Blocks(input + t * rowBytes, blockCount, rng);

            // Repack: full groups interleaved, tail rows appended row-major.
            for (int g = 0; g < fullGroups; g++)
                for (int b = 0; b < blockCount; b++)
                    for (int r = 0; r < 4; r++)
                        Buffer.MemoryCopy(
                            rowMajor + (g * 4 + r) * rowBytes + b * Q8_0BlockBytes,
                            r4 + g * 4 * rowBytes + (b * 4 + r) * Q8_0BlockBytes,
                            Q8_0BlockBytes, Q8_0BlockBytes);
            if (tailRows > 0)
                Buffer.MemoryCopy(
                    rowMajor + fullGroups * 4 * rowBytes,
                    r4 + fullGroups * 4 * rowBytes,
                    (long)tailRows * rowBytes, (long)tailRows * rowBytes);

            // Reference: per-token row-major ComputeRows, which is what the tiled GEMM dispatches to.
            for (int t = 0; t < n; t++)
                MatMul.ComputeRows(rowMajor, input + t * rowBytes, cRef + t * m, m, blockCount);

            MatMul.GemmR4TiledQ8_0(r4, input, cR4, fullGroups, tailRows, blockCount, m, n);

            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    Assert.Equal(cRef[t * m + r], cR4[t * m + r], 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree(rowMajor);
            NativeMemory.AlignedFree(r4);
            NativeMemory.AlignedFree(input);
            NativeMemory.AlignedFree(cR4);
            NativeMemory.AlignedFree(cRef);
        }
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

    // ──────────────────── K-Quant R4-tiled GEMM ────────────────────

    [Theory]
    [InlineData(QuantizationType.Q4_K, 3, 2, 1)]     // tail-only (fullGroups = 0, tailRows = 3)
    [InlineData(QuantizationType.Q4_K, 8, 3, 1)]
    [InlineData(QuantizationType.Q4_K, 37, 5, 2)]    // tailRows = 1, multiple super-blocks
    [InlineData(QuantizationType.Q4_K, 1027, 3, 8)]  // multi-tile per worker (fullGroups=256 > 4*tileGroups=224) + tailRows = 3
    [InlineData(QuantizationType.Q5_K, 14, 4, 2)]    // tailRows = 2
    [InlineData(QuantizationType.Q5_K, 1025, 3, 8)]  // multi-tile per worker (fullGroups=256 > 4*tileGroups=184) + tailRows = 1
    [InlineData(QuantizationType.Q6_K, 15, 3, 2)]    // tailRows = 3
    [InlineData(QuantizationType.Q6_K, 1026, 3, 8)]  // multi-tile per worker (fullGroups=256 > 4*tileGroups=156) + tailRows = 2
    public void OuterProductGemmKQuant_SingleAndParallel_MatchRowMajorReference(
        QuantizationType qt, int m, int n, int superBlockCount)
    {
        var rng = new Random(1337);
        int k = superBlockCount * 256;
        int blockBytes = qt switch
        {
            QuantizationType.Q4_K => 144,
            QuantizationType.Q5_K => 176,
            QuantizationType.Q6_K => 210,
            _ => throw new ArgumentOutOfRangeException(nameof(qt))
        };
        int rowBytes = superBlockCount * blockBytes;
        int q8kRowBytes = superBlockCount * MatMul.Q8_K_BlockBytes;
        int fullGroups = m / 4;
        int tailRows = m % 4;

        byte* rowMajor = (byte*)NativeMemory.AlignedAlloc((nuint)(m * rowBytes), 64);
        float* inputF32 = (float*)NativeMemory.AlignedAlloc((nuint)(n * k * sizeof(float)), 64);
        byte* inputQ8K = (byte*)NativeMemory.AlignedAlloc((nuint)(n * q8kRowBytes), 64);
        float* cSingle = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cParallel = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cRef = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            FillRandomKQuantBlocks(rowMajor, qt, m * superBlockCount, blockBytes, rng);
            for (int i = 0; i < n * k; i++)
                inputF32[i] = (rng.NextSingle() - 0.5f) * 2.0f;
            for (int t = 0; t < n; t++)
                MatMul.QuantizeF32ToQ8_K(inputF32 + t * k, inputQ8K + t * q8kRowBytes, k);

            using var repacked = WeightRepacking.RepackR4((nint)rowMajor, qt, m, k);
            using var pool = new ComputeThreadPool(4);

            switch (qt)
            {
                case QuantizationType.Q4_K:
                    MatMul.GemmQ4_K(rowMajor, inputF32, cRef, m, k, n, pool: null, inputQ8K);
                    MatMul.OuterProductGemmQ4_K((byte*)repacked.Ptr, inputQ8K, cSingle,
                        fullGroups, tailRows, superBlockCount, m, n, pool: null);
                    MatMul.OuterProductGemmQ4_K((byte*)repacked.Ptr, inputQ8K, cParallel,
                        fullGroups, tailRows, superBlockCount, m, n, pool);
                    break;
                case QuantizationType.Q5_K:
                    MatMul.GemmQ5_K(rowMajor, inputF32, cRef, m, k, n, pool: null, inputQ8K);
                    MatMul.OuterProductGemmQ5_K((byte*)repacked.Ptr, inputQ8K, cSingle,
                        fullGroups, tailRows, superBlockCount, m, n, pool: null);
                    MatMul.OuterProductGemmQ5_K((byte*)repacked.Ptr, inputQ8K, cParallel,
                        fullGroups, tailRows, superBlockCount, m, n, pool);
                    break;
                case QuantizationType.Q6_K:
                    MatMul.GemmQ6_K(rowMajor, inputF32, cRef, m, k, n, pool: null, inputQ8K);
                    MatMul.OuterProductGemmQ6_K((byte*)repacked.Ptr, inputQ8K, cSingle,
                        fullGroups, tailRows, superBlockCount, m, n, pool: null);
                    MatMul.OuterProductGemmQ6_K((byte*)repacked.Ptr, inputQ8K, cParallel,
                        fullGroups, tailRows, superBlockCount, m, n, pool);
                    break;
            }

            for (int i = 0; i < n * m; i++)
            {
                Assert.Equal(cRef[i], cSingle[i], 1e-3f);
                Assert.Equal(cSingle[i], cParallel[i], 1e-4f);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(rowMajor);
            NativeMemory.AlignedFree(inputF32);
            NativeMemory.AlignedFree(inputQ8K);
            NativeMemory.AlignedFree(cSingle);
            NativeMemory.AlignedFree(cParallel);
            NativeMemory.AlignedFree(cRef);
        }
    }

    [Fact]
    public void OuterProductGemmKQuant_ZeroDimensionsOrDepth_HandlesSafely()
    {
        float* c = stackalloc float[16];
        for (int i = 0; i < 16; i++)
            c[i] = 42.0f;

        using var pool = new ComputeThreadPool(2);

        // m = 0 or n = 0: no-op, leaves output untouched
        MatMul.OuterProductGemmQ4_K(null, null, c, 0, 0, 0, m: 0, n: 0, pool: null);
        MatMul.OuterProductGemmQ4_K(null, null, c, 0, 0, 0, m: 0, n: 0, pool);
        Assert.Equal(42.0f, c[0]);

        // superBlockCount = 0 with m > 0, n > 0: clears output without dividing by zero
        MatMul.OuterProductGemmQ4_K(null, null, c, 1, 0, superBlockCount: 0, m: 4, n: 2, pool: null);
        for (int i = 0; i < 8; i++)
            Assert.Equal(0.0f, c[i]);

        for (int i = 0; i < 8; i++)
            c[i] = 99.0f;
        MatMul.OuterProductGemmQ5_K(null, null, c, 1, 0, superBlockCount: 0, m: 4, n: 2, pool);
        for (int i = 0; i < 8; i++)
            Assert.Equal(0.0f, c[i]);
    }

    [Theory]
    [InlineData(QuantizationType.Q4_K)]
    [InlineData(QuantizationType.Q5_K)]
    [InlineData(QuantizationType.Q6_K)]
    public void TransformerModel_Forward_KQuantMultiTokenPrefill_MatchesSequentialSingleTokenWithKvCache(
        QuantizationType qt)
    {
        // HiddenSize = 2048 (8 super-blocks) ensures RowBytes >= InterleavedMinRowBytes (1024)
        // for Q4_K (8 * 144 = 1152), Q5_K (8 * 176 = 1408), and Q6_K (8 * 210 = 1680),
        // exercising TransformerModel.GemmInterleaved -> OuterProductGemmQ*_K during prefill.
        const int hiddenSize = 2048;
        const int numHeads = 8;
        const int numKvHeads = 4;
        const int headDim = 256;
        const int intermediateSize = 2048;
        const int vocabSize = 12;
        uint ggmlType = (uint)qt;

        int superBlocksPerRow = hiddenSize / 256;
        int blockBytes = qt switch
        {
            QuantizationType.Q4_K => 144,
            QuantizationType.Q5_K => 176,
            QuantizationType.Q6_K => 210,
            _ => throw new ArgumentOutOfRangeException(nameof(qt))
        };

        byte[] MakeF32Bytes(int count, int seed)
        {
            var rng = new Random(seed);
            float[] vals = new float[count];
            for (int i = 0; i < count; i++)
                vals[i] = 0.8f + (rng.NextSingle() * 0.4f);
            byte[] bytes = new byte[count * sizeof(float)];
            Buffer.BlockCopy(vals, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        byte[] MakeKQuantBytes(int rows, int seed)
        {
            var rng = new Random(seed);
            int totalBlocks = rows * superBlocksPerRow;
            byte[] bytes = new byte[totalBlocks * blockBytes];
            fixed (byte* ptr = bytes)
                FillRandomKQuantBlocks(ptr, qt, totalBlocks, blockBytes, rng);
            return bytes;
        }

        var ggufData = new GgufTestData(version: 3)
            .AddString("general.architecture", "llama")
            .AddTensor("token_embd.weight", [hiddenSize, vocabSize], 0u, MakeF32Bytes(vocabSize * hiddenSize, 1))
            .AddTensor("output_norm.weight", [hiddenSize], 0u, MakeF32Bytes(hiddenSize, 2))
            .AddTensor("output.weight", [hiddenSize, vocabSize], ggmlType, MakeKQuantBytes(vocabSize, 3))
            .AddTensor("blk.0.attn_norm.weight", [hiddenSize], 0u, MakeF32Bytes(hiddenSize, 4))
            .AddTensor("blk.0.attn_q.weight", [hiddenSize, numHeads * headDim], ggmlType, MakeKQuantBytes(numHeads * headDim, 5))
            .AddTensor("blk.0.attn_k.weight", [hiddenSize, numKvHeads * headDim], ggmlType, MakeKQuantBytes(numKvHeads * headDim, 6))
            .AddTensor("blk.0.attn_v.weight", [hiddenSize, numKvHeads * headDim], ggmlType, MakeKQuantBytes(numKvHeads * headDim, 7))
            .AddTensor("blk.0.attn_output.weight", [numHeads * headDim, hiddenSize], ggmlType, MakeKQuantBytes(hiddenSize, 8))
            .AddTensor("blk.0.ffn_norm.weight", [hiddenSize], 0u, MakeF32Bytes(hiddenSize, 9))
            .AddTensor("blk.0.ffn_gate.weight", [hiddenSize, intermediateSize], ggmlType, MakeKQuantBytes(intermediateSize, 10))
            .AddTensor("blk.0.ffn_up.weight", [hiddenSize, intermediateSize], ggmlType, MakeKQuantBytes(intermediateSize, 11))
            .AddTensor("blk.0.ffn_down.weight", [intermediateSize, hiddenSize], ggmlType, MakeKQuantBytes(hiddenSize, 12));

        string tempPath = ggufData.WriteToTempFile();
        try
        {
            var config = new ModelConfig
            {
                Architecture = DotLLM.Core.Configuration.Architecture.Llama,
                HiddenSize = hiddenSize,
                IntermediateSize = intermediateSize,
                NumLayers = 1,
                NumAttentionHeads = numHeads,
                NumKvHeads = numKvHeads,
                HeadDim = headDim,
                VocabSize = vocabSize,
                MaxSequenceLength = 32,
                NormEpsilon = 1e-5f,
            };

            using var ggufPrefill = GgufFile.Open(tempPath);
            using var prefillModel = TransformerModel.LoadFromGguf(ggufPrefill, config);
            using var prefillCache = new SimpleKvCache(1, numKvHeads, headDim, 32);

            using var ggufDecode = GgufFile.Open(tempPath);
            using var decodeModel = TransformerModel.LoadFromGguf(ggufDecode, config);
            using var decodeCache = new SimpleKvCache(1, numKvHeads, headDim, 32);

            int[] tokens = [1, 4, 7];
            int[] positions = [0, 1, 2];

            using var prefillTensor = prefillModel.Forward(tokens, positions, -1, prefillCache);
            var prefillLastRow = new ReadOnlySpan<float>(
                (float*)prefillTensor.DataPointer + (tokens.Length - 1) * vocabSize, vocabSize).ToArray();

            float[] decodeLastRow = [];
            for (int t = 0; t < tokens.Length; t++)
            {
                using var decodeTensor = decodeModel.Forward(
                    tokens.AsSpan(t, 1), positions.AsSpan(t, 1), -1, decodeCache);
                decodeLastRow = new ReadOnlySpan<float>((float*)decodeTensor.DataPointer, vocabSize).ToArray();
            }

            for (int v = 0; v < vocabSize; v++)
            {
                Assert.Equal(decodeLastRow[v], prefillLastRow[v], 1e-2f);
            }
        }
        finally
        {
            try { File.Delete(tempPath); } catch { /* best-effort */ }
        }
    }

    private static void FillRandomKQuantBlocks(byte* ptr, QuantizationType qt, int totalBlocks, int blockBytes, Random rng)
    {
        var span = new Span<byte>(ptr, totalBlocks * blockBytes);
        rng.NextBytes(span);
        for (int b = 0; b < totalBlocks; b++)
        {
            byte* block = ptr + b * blockBytes;
            if (qt is QuantizationType.Q4_K or QuantizationType.Q5_K)
            {
                Unsafe.WriteUnaligned(block, (Half)(0.001f + rng.NextSingle() * 0.004f));
                Unsafe.WriteUnaligned(block + 2, (Half)(0.0005f + rng.NextSingle() * 0.002f));
            }
            else if (qt == QuantizationType.Q6_K)
            {
                Unsafe.WriteUnaligned(block + 208, (Half)(0.001f + rng.NextSingle() * 0.004f));
            }
        }
    }
}
