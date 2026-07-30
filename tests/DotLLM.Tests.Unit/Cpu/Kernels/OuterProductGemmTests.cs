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

    // ──────────────────── AVX2-VNNI microkernel ────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(8)]    // K=256
    [InlineData(16)]
    [InlineData(18)]   // SmolLM-135M: 576/32
    [InlineData(32)]   // K=1024
    [InlineData(48)]   // K=1536
    [InlineData(128)]  // K=4096
    public void OuterProductVnni_4x3_MatchesScalar(int blockCount)
    {
        if (!AvxVnni.IsSupported)
            return;

        var rng = new Random(1234);
        int m = 4;
        int n = 3;
        int rowBytes = blockCount * Q8_0BlockBytes;

        byte* weights = AllocAndFillR4Weights(4, blockCount, rng);
        byte*[] xPtrs = new byte*[n];
        for (int t = 0; t < n; t++)
        {
            xPtrs[t] = (byte*)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
            FillRandomQ8_0Blocks(xPtrs[t], blockCount, rng);
        }

        float* cScalar = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cVnni = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            MatMul.OuterProductQ8_0Scalar_4x3(
                weights, xPtrs[0], xPtrs[1], xPtrs[2],
                cScalar, blockCount, m);

            MatMul.OuterProductQ8_0Vnni_4x3(
                weights, xPtrs[0], xPtrs[1], xPtrs[2],
                cVnni, blockCount, m);

            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    Assert.Equal(cScalar[t * m + r], cVnni[t * m + r], 1e-2f);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            for (int t = 0; t < n; t++)
                NativeMemory.AlignedFree(xPtrs[t]);
            NativeMemory.AlignedFree(cScalar);
            NativeMemory.AlignedFree(cVnni);
        }
    }

    // VNNI vs the AVX2 maddubs+madd microkernel: both vector paths fold each
    // block's int32 sum by dx*dw in the same order, so they should agree to a
    // tighter tolerance than either does to the scalar reference.
    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(18)]
    [InlineData(48)]
    [InlineData(128)]
    public void OuterProductVnni_4x3_MatchesAvx2(int blockCount)
    {
        if (!AvxVnni.IsSupported || !Avx2.IsSupported)
            return;

        var rng = new Random(777);
        int m = 4;
        int n = 3;
        int rowBytes = blockCount * Q8_0BlockBytes;

        byte* weights = AllocAndFillR4Weights(4, blockCount, rng);
        byte*[] xPtrs = new byte*[n];
        for (int t = 0; t < n; t++)
        {
            xPtrs[t] = (byte*)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
            FillRandomQ8_0Blocks(xPtrs[t], blockCount, rng);
        }

        float* cAvx2 = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cVnni = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            MatMul.OuterProductQ8_0Avx2_4x3(
                weights, xPtrs[0], xPtrs[1], xPtrs[2],
                cAvx2, blockCount, m);

            MatMul.OuterProductQ8_0Vnni_4x3(
                weights, xPtrs[0], xPtrs[1], xPtrs[2],
                cVnni, blockCount, m);

            for (int t = 0; t < n; t++)
                for (int r = 0; r < m; r++)
                    Assert.Equal(cAvx2[t * m + r], cVnni[t * m + r], 1e-3f);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            for (int t = 0; t < n; t++)
                NativeMemory.AlignedFree(xPtrs[t]);
            NativeMemory.AlignedFree(cAvx2);
            NativeMemory.AlignedFree(cVnni);
        }
    }

    // Discriminating sanity check: the VNNI microkernel parity test must FAIL
    // when the kernel output is deliberately perturbed. This guards against a
    // vacuous test (e.g. one that compares all-zero buffers, or a tolerance so
    // wide that a real tile bug slips through). We corrupt a single (token,row)
    // cell of the VNNI result and assert the comparison rejects it.
    [Fact]
    public void OuterProductVnni_4x3_ParityTestIsDiscriminating()
    {
        if (!AvxVnni.IsSupported)
            return;

        var rng = new Random(31337);
        int m = 4;
        int n = 3;
        int blockCount = 18;
        int rowBytes = blockCount * Q8_0BlockBytes;

        byte* weights = AllocAndFillR4Weights(4, blockCount, rng);
        byte*[] xPtrs = new byte*[n];
        for (int t = 0; t < n; t++)
        {
            xPtrs[t] = (byte*)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
            FillRandomQ8_0Blocks(xPtrs[t], blockCount, rng);
        }

        float* cScalar = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);
        float* cVnni = (float*)NativeMemory.AlignedAlloc((nuint)(n * m * sizeof(float)), 64);

        try
        {
            MatMul.OuterProductQ8_0Scalar_4x3(
                weights, xPtrs[0], xPtrs[1], xPtrs[2], cScalar, blockCount, m);
            MatMul.OuterProductQ8_0Vnni_4x3(
                weights, xPtrs[0], xPtrs[1], xPtrs[2], cVnni, blockCount, m);

            // Unperturbed: every cell must match (and must be meaningfully nonzero,
            // so the comparison is not trivially satisfied by zeros).
            float maxAbs = 0;
            for (int i = 0; i < n * m; i++)
            {
                Assert.Equal(cScalar[i], cVnni[i], 1e-2f);
                maxAbs = MathF.Max(maxAbs, MathF.Abs(cScalar[i]));
            }
            Assert.True(maxAbs > 1e-3f, "reference output is ~0; parity check would be vacuous");

            // Perturb one cell (token=1, row=2) by an amount far exceeding tolerance
            // and confirm the same equality assertion now fails — i.e. the test
            // discriminates broken from correct output.
            int idx = 1 * m + 2;
            cVnni[idx] += 1.0f;
            Assert.ThrowsAny<Xunit.Sdk.XunitException>(
                () => Assert.Equal(cScalar[idx], cVnni[idx], 1e-2f));
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            for (int t = 0; t < n; t++)
                NativeMemory.AlignedFree(xPtrs[t]);
            NativeMemory.AlignedFree(cScalar);
            NativeMemory.AlignedFree(cVnni);
        }
    }

    // Full GEMM through the public OuterProductGemmQ8_0 dispatch, which routes
    // to the VNNI microkernel on this CPU (AvxVnni.IsSupported). Exercises
    // discriminating shapes: multi-block K, full tile, all tail combinations,
    // and a large (M>=128, N>=32) shape.
    [Theory]
    [InlineData(4, 3, 64)]       // single full tile, K=64 (2 blocks)
    [InlineData(8, 6, 256)]      // 2 groups, 2 token-tiles, K=256 (8 blocks)
    [InlineData(4, 3, 1024)]     // deep K (32 blocks)
    [InlineData(7, 5, 64)]       // row tail (m%4) + token tail (n%3)
    [InlineData(5, 4, 128)]      // row tail + token tail, K=128
    [InlineData(13, 11, 256)]    // 3 groups + 1 tail row, token tail
    [InlineData(128, 32, 64)]    // large: 32 groups, 32 tokens
    [InlineData(132, 33, 256)]   // large with row + token tails, deep K
    public void OuterProductGemmVnni_Dispatch_MatchesReference(int m, int n, int k)
    {
        if (!AvxVnni.IsSupported)
            return;

        var rng = new Random(0xBEEF ^ (m * 131 + n) * 17 + k);
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

    // ──────────────────── F32 outer-product GEMM ────────────────────
    //
    // These exercise the new `OuterProductGemm.OuterProductGemmF32` kernel
    // (companion to the Q8_0 outer-product kernels above). Parity is checked
    // against:
    //   1. `MatMul.GemmF32Scalar` — the canonical reference, no FMA.
    //   2. `MatMul.GemmF32` — the production tiled path, FMA-enabled.
    //   3. `OuterProductGemm.OuterProductGemmF32Scalar` — internal cross-check.
    //
    // FMA vs separate-mul-add reorders the rounding, so we use a relative
    // tolerance that scales with K (the contraction dim accumulates error
    // linearly in K for uniformly-distributed inputs).

    [Theory]
    [InlineData(4, 3, 8)]        // smallest fully-vectorisable tile
    [InlineData(4, 3, 64)]       // single AVX2 group × 8
    [InlineData(8, 6, 64)]       // 2 row-tiles × 2 token-tiles
    [InlineData(12, 9, 128)]     // 3 row-tiles × 3 token-tiles, K=128
    [InlineData(16, 12, 256)]    // 4 row-tiles × 4 token-tiles
    [InlineData(32, 16, 512)]    // prefill-shaped
    public void OuterProductGemmF32_Scalar_MatchesReference(int m, int n, int k)
    {
        RunF32ParityCase(m, n, k, useVector: false);
    }

    [Theory]
    [InlineData(4, 3, 8)]
    [InlineData(4, 3, 64)]
    [InlineData(8, 6, 64)]
    [InlineData(12, 9, 128)]
    [InlineData(16, 12, 256)]
    [InlineData(32, 16, 512)]
    [InlineData(128, 32, 1024)]  // bigger prefill
    public void OuterProductGemmF32_Avx2_MatchesReference(int m, int n, int k)
    {
        RunF32ParityCase(m, n, k, useVector: true);
    }

    // Tail-handling cases: shapes where M, N, or K are not multiples of the
    // microkernel tile (4 rows × 3 tokens × 8-wide vector).

    [Theory]
    [InlineData(5, 3, 64)]       // row tail (m % 4 != 0)
    [InlineData(7, 3, 64)]
    [InlineData(4, 4, 64)]       // token tail (n % 3 != 0)
    [InlineData(4, 5, 64)]
    [InlineData(7, 5, 64)]       // both tails
    [InlineData(13, 11, 33)]     // K tail (k % 8 != 0) + row + token tails
    [InlineData(17, 9, 65)]      // K tail with FMA group
    public void OuterProductGemmF32_HandlesAllTails(int m, int n, int k)
    {
        RunF32ParityCase(m, n, k, useVector: true);
        RunF32ParityCase(m, n, k, useVector: false);
    }

    // Edge cases.

    [Theory]
    [InlineData(1, 1, 1)]        // degenerate scalar
    [InlineData(1, 3, 64)]       // single row — all-row-tail
    [InlineData(4, 1, 64)]       // single token — all-token-tail
    [InlineData(1, 1, 4096)]     // pure inner product
    [InlineData(4, 3, 1)]        // K=1 — every tile collapses to single FMA
    public void OuterProductGemmF32_EdgeCases(int m, int n, int k)
    {
        RunF32ParityCase(m, n, k, useVector: true);
        RunF32ParityCase(m, n, k, useVector: false);
    }

    private static void RunF32ParityCase(int m, int n, int k, bool useVector)
    {
        if (useVector && (!Avx2.IsSupported || !Fma.IsSupported))
        {
            // Vector path falls back to scalar; covered by the scalar case.
            return;
        }

        var rng = new Random(0xC0FFEE ^ (m * 31 + n) * 31 + k);
        long aLen = (long)m * k;
        long bLen = (long)n * k;
        long cLen = (long)n * m;

        float* a = (float*)NativeMemory.AlignedAlloc((nuint)(aLen * sizeof(float)), 64);
        float* b = (float*)NativeMemory.AlignedAlloc((nuint)(bLen * sizeof(float)), 64);
        float* cOuter = (float*)NativeMemory.AlignedAlloc((nuint)(cLen * sizeof(float)), 64);
        float* cRefScalar = (float*)NativeMemory.AlignedAlloc((nuint)(cLen * sizeof(float)), 64);
        float* cRefGemm = (float*)NativeMemory.AlignedAlloc((nuint)(cLen * sizeof(float)), 64);

        try
        {
            // Use range [-1, 1] — typical of post-normalisation activations and
            // weight distributions. Larger magnitudes pile up error sooner.
            for (long i = 0; i < aLen; i++) a[i] = rng.NextSingle() * 2f - 1f;
            for (long i = 0; i < bLen; i++) b[i] = rng.NextSingle() * 2f - 1f;

            // Reference: `MatMul.GemmF32Scalar` — canonical scalar inner product.
            MatMul.GemmF32Scalar(a, b, cRefScalar, m, k, n);

            // Reference: `MatMul.GemmF32` — production path (FMA inside TensorPrimitives).
            MatMul.GemmF32(a, b, cRefGemm, m, k, n);

            if (useVector)
                OuterProductGemm.OuterProductGemmF32(a, b, cOuter, m, k, n);
            else
                OuterProductGemm.OuterProductGemmF32Scalar(a, b, cOuter, m, k, n);

            if (!useVector)
            {
                // Scalar path: accumulates the same terms in the same ascending
                // index order as `MatMul.GemmF32Scalar`, and .NET does not
                // auto-contract `mul + add` to FMA — so bit-exact equality is
                // achievable and required.
                for (long i = 0; i < cLen; i++)
                {
                    Assert.Equal(cRefScalar[i], cOuter[i]);
                }
            }
            else
            {
                // Vector path: AVX2 FMA + horizontal reduction reorders the
                // summation vs the scalar inner product. Use an absolute
                // tolerance that scales with √K — for unit-magnitude operands
                // and float32 (ULP ≈ 1.2e-7) the standard error of an unbiased
                // K-term sum grows as √K.
                float absTol = 4e-6f * MathF.Sqrt(k);

                for (long i = 0; i < cLen; i++)
                {
                    float diff = MathF.Abs(cOuter[i] - cRefGemm[i]);
                    Assert.True(
                        diff <= absTol,
                        $"shape m={m} n={n} k={k} idx={i}: outer={cOuter[i]:R} ref={cRefGemm[i]:R} diff={diff:R} tol={absTol:R}");
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(a);
            NativeMemory.AlignedFree(b);
            NativeMemory.AlignedFree(cOuter);
            NativeMemory.AlignedFree(cRefScalar);
            NativeMemory.AlignedFree(cRefGemm);
        }
    }
}
