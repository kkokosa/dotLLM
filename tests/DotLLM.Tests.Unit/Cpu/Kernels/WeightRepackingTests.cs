using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class WeightRepackingTests
{
    // Block sizes / group sizes per quant type
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int Q5_0BlockBytes = 22;
    private const int Q5_0GroupSize = 32;
    private const int Q4_K_BlockBytes = 144;
    private const int Q5_K_BlockBytes = 176;
    private const int Q6_K_BlockBytes = 210;
    private const int KQuantGroupSize = 256;

    // ──────────────────── Roundtrip tests ────────────────────

    [Theory]
    [InlineData(QuantizationType.Q8_0, 8, 128)]   // 2 full groups, 0 tail
    [InlineData(QuantizationType.Q8_0, 12, 64)]    // 3 full groups, 0 tail
    [InlineData(QuantizationType.Q5_0, 8, 64)]     // 2 full groups, 0 tail
    [InlineData(QuantizationType.Q5_0, 4, 96)]     // 1 full group, 0 tail
    public void RepackR4_Roundtrip_BlocksAtExpectedPositions(QuantizationType qt, int m, int k)
    {
        var (blockBytes, groupSize) = WeightRepacking.GetBlockInfo(qt);
        int blocksPerRow = k / groupSize;
        int rowBytes = blocksPerRow * blockBytes;
        long totalBytes = (long)m * rowBytes;

        // Allocate source with known pattern: each block's first byte = row*blocksPerRow + blockCol
        byte* src = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        try
        {
            for (int row = 0; row < m; row++)
            {
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = src + (long)row * rowBytes + (long)b * blockBytes;
                    // Fill block with identifiable pattern
                    byte tag = (byte)(row * blocksPerRow + b);
                    for (int i = 0; i < blockBytes; i++)
                        block[i] = (byte)(tag ^ i); // unique per block+offset
                }
            }

            using var repacked = WeightRepacking.RepackR4((nint)src, qt, m, k);
            byte* dst = (byte*)repacked.Ptr;

            int fullGroups = m / 4;

            // Verify interleaved layout: for group g, block b, row r:
            // dst[g * 4 * rowBytes + b * 4 * blockBytes + r * blockBytes + i]
            // should equal src[row * rowBytes + b * blockBytes + i]
            for (int g = 0; g < fullGroups; g++)
            {
                for (int b = 0; b < blocksPerRow; b++)
                {
                    for (int r = 0; r < 4; r++)
                    {
                        int srcRow = g * 4 + r;
                        byte* srcBlock = src + (long)srcRow * rowBytes + (long)b * blockBytes;
                        byte* dstBlock = dst + (long)g * 4 * rowBytes
                                         + (long)b * 4 * blockBytes + (long)r * blockBytes;

                        for (int i = 0; i < blockBytes; i++)
                        {
                            Assert.Equal(srcBlock[i], dstBlock[i]);
                        }
                    }
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(src);
        }
    }

    // ──────────────────── Tail row tests ────────────────────

    [Theory]
    [InlineData(5, 1)]  // 1 full group, 1 tail
    [InlineData(6, 2)]  // 1 full group, 2 tail
    [InlineData(7, 3)]  // 1 full group, 3 tail
    [InlineData(9, 1)]  // 2 full groups, 1 tail
    public void RepackR4_TailRows_PreservedInOriginalOrder(int m, int expectedTail)
    {
        const QuantizationType qt = QuantizationType.Q8_0;
        const int k = 64; // 2 blocks per row
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        long totalBytes = (long)m * rowBytes;

        byte* src = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        try
        {
            // Fill with row-identifiable pattern
            for (int row = 0; row < m; row++)
                for (int i = 0; i < rowBytes; i++)
                    src[row * rowBytes + i] = (byte)(row * 17 + i);

            using var repacked = WeightRepacking.RepackR4((nint)src, qt, m, k);

            Assert.Equal(expectedTail, repacked.TailRows);
            Assert.Equal(m / 4, repacked.FullGroupCount);

            // Verify tail rows are byte-for-byte identical to source
            int tailStartRow = (m / 4) * 4;
            byte* tailSrc = src + (long)tailStartRow * rowBytes;
            byte* tailDst = repacked.TailPtr;

            for (int r = 0; r < expectedTail; r++)
            {
                for (int i = 0; i < rowBytes; i++)
                {
                    Assert.Equal(tailSrc[r * rowBytes + i], tailDst[r * rowBytes + i]);
                }
            }
        }
        finally
        {
            NativeMemory.AlignedFree(src);
        }
    }

    [Fact]
    public void RepackR4_ExactMultipleOf4_NoTailRows()
    {
        const int m = 8, k = 64;
        long totalBytes = (long)m * (k / Q8_0GroupSize) * Q8_0BlockBytes;

        byte* src = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        try
        {
            new Span<byte>(src, (int)totalBytes).Fill(0xAB);
            using var repacked = WeightRepacking.RepackR4((nint)src, QuantizationType.Q8_0, m, k);

            Assert.Equal(0, repacked.TailRows);
            Assert.Equal(2, repacked.FullGroupCount);
        }
        finally
        {
            NativeMemory.AlignedFree(src);
        }
    }

    // ──────────────────── Alignment test ────────────────────

    [Theory]
    [InlineData(QuantizationType.Q8_0)]
    [InlineData(QuantizationType.Q5_0)]
    [InlineData(QuantizationType.Q4_K)]
    [InlineData(QuantizationType.Q5_K)]
    [InlineData(QuantizationType.Q6_K)]
    public void RepackR4_Ptr_Is64ByteAligned(QuantizationType qt)
    {
        var (blockBytes, groupSize) = WeightRepacking.GetBlockInfo(qt);
        int k = groupSize * 2; // 2 blocks per row
        int m = 8;
        int rowBytes = 2 * blockBytes;
        long totalBytes = (long)m * rowBytes;

        byte* src = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        try
        {
            new Span<byte>(src, (int)totalBytes).Clear();
            using var repacked = WeightRepacking.RepackR4((nint)src, qt, m, k);

            Assert.Equal(0, repacked.Ptr % 64);
        }
        finally
        {
            NativeMemory.AlignedFree(src);
        }
    }

    // ──────────────────── Numerical equivalence: Q8_0 ────────────────────

    [Theory]
    [InlineData(8, 128)]    // 2 groups, 0 tail
    [InlineData(9, 128)]    // 2 groups, 1 tail
    [InlineData(11, 64)]    // 2 groups, 3 tail
    [InlineData(16, 256)]   // 4 groups, 0 tail
    [InlineData(4, 32)]     // 1 group, 0 tail — minimum
    public void ComputeRowsQ8_0Interleaved_MatchesOriginal(int m, int k)
    {
        var rng = new Random(42);
        int blockCount = k / Q8_0GroupSize;
        int rowBytes = blockCount * Q8_0BlockBytes;

        // Generate random Q8_0 weights: Half scale + 32 sbytes per block
        byte* weights = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * rowBytes), 64);
        try
        {
            FillRandomQ8_0Weights(rng, weights, m, blockCount);

            // Generate random f32 input and quantize to Q8_0
            float[] xF32 = new float[k];
            for (int i = 0; i < k; i++) xF32[i] = rng.NextSingle() * 2f - 1f;

            int xQ8Bytes = blockCount * Q8_0BlockBytes;
            byte* xQ8 = stackalloc byte[xQ8Bytes];
            fixed (float* xp = xF32)
                MatMul.QuantizeF32ToQ8_0(xp, xQ8, k);

            // Reference: row-major ComputeRows
            float* refResult = stackalloc float[m];
            MatMul.ComputeRows(weights, xQ8, refResult, m, blockCount);

            // Repack and compute interleaved
            using var repacked = WeightRepacking.RepackR4((nint)weights, QuantizationType.Q8_0, m, k);
            float* intResult = stackalloc float[m];
            MatMul.ComputeRowsQ8_0Interleaved((byte*)repacked.Ptr, xQ8, intResult,
                repacked.FullGroupCount, repacked.TailRows, blockCount);

            // Compare — must be exactly equal (same integer arithmetic)
            for (int i = 0; i < m; i++)
                Assert.Equal(refResult[i], intResult[i]);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
        }
    }

    // ──────────────────── Numerical equivalence: Q5_0 ────────────────────

    [Theory]
    [InlineData(8, 64)]
    [InlineData(9, 64)]
    [InlineData(12, 128)]
    public void ComputeRowsQ5_0Interleaved_MatchesOriginal(int m, int k)
    {
        var rng = new Random(42);
        int blockCount = k / Q5_0GroupSize;
        int rowBytes = blockCount * Q5_0BlockBytes;

        byte* weights = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * rowBytes), 64);
        try
        {
            FillRandomQ5_0Weights(rng, weights, m, blockCount);

            // Generate random input and quantize to Q8_1
            float[] xF32 = new float[k];
            for (int i = 0; i < k; i++) xF32[i] = rng.NextSingle() * 2f - 1f;

            int xQ8Bytes = blockCount * MatMul.Q8_1BlockBytes;
            byte* xQ8 = stackalloc byte[xQ8Bytes];
            fixed (float* xp = xF32)
                MatMul.QuantizeF32ToQ8_1(xp, xQ8, k);

            // Reference: GemvQ5_0 uses ComputeRowsQ5_0 internally
            float* refResult = stackalloc float[m];
            MatMul.ComputeRowsQ5_0(weights, xQ8, refResult, m, blockCount);

            // Interleaved
            using var repacked = WeightRepacking.RepackR4((nint)weights, QuantizationType.Q5_0, m, k);
            float* intResult = stackalloc float[m];
            MatMul.ComputeRowsQ5_0Interleaved((byte*)repacked.Ptr, xQ8, intResult,
                repacked.FullGroupCount, repacked.TailRows, blockCount);

            for (int i = 0; i < m; i++)
                Assert.Equal(refResult[i], intResult[i]);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
        }
    }

    // ──────────────────── Numerical equivalence: K-quants ────────────────────

    [Theory]
    [InlineData(8, 512)]    // 2 groups, 0 tail
    [InlineData(9, 512)]    // 2 groups, 1 tail
    [InlineData(4, 256)]    // 1 group, minimum
    public void ComputeRowsQ4_KInterleaved_MatchesOriginal(int m, int k)
    {
        VerifyKQuantInterleaved(m, k, QuantizationType.Q4_K, Q4_K_BlockBytes,
            static (w, x, r, rows, sb) => MatMul.ComputeRowsQ4_K(w, x, r, rows, sb),
            static (w, x, r, fg, tr, sb) => MatMul.ComputeRowsQ4_KInterleaved(w, x, r, fg, tr, sb));
    }

    [Theory]
    [InlineData(8, 512)]
    [InlineData(9, 512)]
    public void ComputeRowsQ5_KInterleaved_MatchesOriginal(int m, int k)
    {
        VerifyKQuantInterleaved(m, k, QuantizationType.Q5_K, Q5_K_BlockBytes,
            static (w, x, r, rows, sb) => MatMul.ComputeRowsQ5_K(w, x, r, rows, sb),
            static (w, x, r, fg, tr, sb) => MatMul.ComputeRowsQ5_KInterleaved(w, x, r, fg, tr, sb));
    }

    [Theory]
    [InlineData(8, 512)]
    [InlineData(9, 512)]
    public void ComputeRowsQ6_KInterleaved_MatchesOriginal(int m, int k)
    {
        VerifyKQuantInterleaved(m, k, QuantizationType.Q6_K, Q6_K_BlockBytes,
            static (w, x, r, rows, sb) => MatMul.ComputeRowsQ6_K(w, x, r, rows, sb),
            static (w, x, r, fg, tr, sb) => MatMul.ComputeRowsQ6_KInterleaved(w, x, r, fg, tr, sb));
    }

    // ──────────────────── IsRepackable ────────────────────

    [Theory]
    [InlineData(QuantizationType.Q8_0, true)]
    [InlineData(QuantizationType.Q5_0, true)]
    [InlineData(QuantizationType.Q4_K, true)]
    [InlineData(QuantizationType.Q5_K, true)]
    [InlineData(QuantizationType.Q6_K, true)]
    [InlineData(QuantizationType.F32, false)]
    [InlineData(QuantizationType.F16, false)]
    public void IsRepackable_CorrectForAllTypes(QuantizationType qt, bool expected)
    {
        Assert.Equal(expected, WeightRepacking.IsRepackable(qt));
    }

    // ──────────────────── GetBlockInfo ────────────────────

    [Theory]
    [InlineData(QuantizationType.Q8_0, 34, 32)]
    [InlineData(QuantizationType.Q5_0, 22, 32)]
    [InlineData(QuantizationType.Q4_K, 144, 256)]
    [InlineData(QuantizationType.Q5_K, 176, 256)]
    [InlineData(QuantizationType.Q6_K, 210, 256)]
    [InlineData(QuantizationType.F32, 0, 0)]
    public void GetBlockInfo_CorrectValues(QuantizationType qt, int expectedBlockBytes, int expectedGroupSize)
    {
        var (blockBytes, groupSize) = WeightRepacking.GetBlockInfo(qt);
        Assert.Equal(expectedBlockBytes, blockBytes);
        Assert.Equal(expectedGroupSize, groupSize);
    }

    // ──────────────────── RepackedWeight properties ────────────────────

    [Fact]
    public void RepackedWeight_Properties_CorrectValues()
    {
        const int m = 10, k = 128;
        int blockCount = k / Q8_0GroupSize; // 4
        int rowBytes = blockCount * Q8_0BlockBytes; // 136
        long totalBytes = (long)m * rowBytes;

        byte* src = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        try
        {
            new Span<byte>(src, (int)totalBytes).Clear();
            using var rw = WeightRepacking.RepackR4((nint)src, QuantizationType.Q8_0, m, k);

            Assert.Equal(2, rw.FullGroupCount);     // 10 / 4
            Assert.Equal(2, rw.TailRows);            // 10 % 4
            Assert.Equal(4, rw.BlocksPerRow);         // 128 / 32
            Assert.Equal(Q8_0BlockBytes, rw.BlockBytes);
            Assert.Equal(totalBytes, rw.AllocatedBytes);
            Assert.Equal(blockCount * Q8_0BlockBytes, rw.RowBytes);
        }
        finally
        {
            NativeMemory.AlignedFree(src);
        }
    }

    // ──────────────────── Dispose safety ────────────────────

    [Fact]
    public void RepackedWeight_Dispose_DoesNotThrow()
    {
        const int m = 4, k = 32;
        int rowBytes = (k / Q8_0GroupSize) * Q8_0BlockBytes;
        long totalBytes = (long)m * rowBytes;

        byte* src = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        new Span<byte>(src, (int)totalBytes).Clear();

        var rw = WeightRepacking.RepackR4((nint)src, QuantizationType.Q8_0, m, k);
        NativeMemory.AlignedFree(src);

        // Dispose should not throw
        rw.Dispose();
    }

    [Fact]
    public void RepackedWeight_Default_Dispose_DoesNotThrow()
    {
        // Default struct (Ptr = 0) — Dispose should be safe
        var rw = default(WeightRepacking.RepackedWeight);
        rw.Dispose(); // Should not throw
    }

    // ──────────────────── K-quant helpers ────────────────────

    private delegate void ComputeRowsMajorDelegate(byte* w, byte* x, float* r, int rows, int sb);
    private delegate void ComputeRowsInterleavedDelegate(byte* w, byte* x, float* r, int fg, int tr, int sb);

    private static void VerifyKQuantInterleaved(int m, int k, QuantizationType qt, int blockBytes,
        ComputeRowsMajorDelegate computeRowsMajor,
        ComputeRowsInterleavedDelegate computeRowsInterleaved)
    {
        var rng = new Random(42);
        int superBlockCount = k / KQuantGroupSize;
        int rowBytes = superBlockCount * blockBytes;

        byte* weights = (byte*)NativeMemory.AlignedAlloc((nuint)((long)m * rowBytes), 64);
        try
        {
            // Fill with random bytes (realistic enough for dot-product equivalence testing)
            var span = new Span<byte>(weights, m * rowBytes);
            for (int i = 0; i < span.Length; i++)
                span[i] = (byte)rng.Next(256);

            // Quantize input to Q8_K
            float[] xF32 = new float[k];
            for (int i = 0; i < k; i++) xF32[i] = rng.NextSingle() * 2f - 1f;

            int q8kBytes = superBlockCount * MatMul.Q8_K_BlockBytes;
            byte* xQ8K = (byte*)NativeMemory.AlignedAlloc((nuint)q8kBytes, 64);
            try
            {
                fixed (float* xp = xF32)
                    MatMul.QuantizeF32ToQ8_K(xp, xQ8K, k);

                // Reference: row-major
                float* refResult = stackalloc float[m];
                computeRowsMajor(weights, xQ8K, refResult, m, superBlockCount);

                // Interleaved
                using var repacked = WeightRepacking.RepackR4((nint)weights, qt, m, k);
                float* intResult = stackalloc float[m];
                computeRowsInterleaved((byte*)repacked.Ptr, xQ8K, intResult,
                    repacked.FullGroupCount, repacked.TailRows, superBlockCount);

                // K-quant interleaved accumulates per-super-block then sums, while row-major
                // accumulates all super-blocks in one call. Different summation order = different
                // float rounding, so use relative tolerance instead of exact equality.
                for (int i = 0; i < m; i++)
                {
                    float expected = refResult[i];
                    float actual = intResult[i];
                    if (float.IsNaN(expected) && float.IsNaN(actual)) continue;
                    float tolerance = Math.Max(1e-3f, Math.Abs(expected) * 1e-5f);
                    Assert.True(Math.Abs(expected - actual) <= tolerance,
                        $"Row {i}: expected {expected}, actual {actual}, diff {Math.Abs(expected - actual)}");
                }
            }
            finally
            {
                NativeMemory.AlignedFree(xQ8K);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
        }
    }

    // ──────────────────── Weight generation helpers ────────────────────

    private static void FillRandomQ8_0Weights(Random rng, byte* weights, int m, int blockCount)
    {
        for (int row = 0; row < m; row++)
        {
            for (int b = 0; b < blockCount; b++)
            {
                byte* block = weights + (long)row * blockCount * Q8_0BlockBytes + (long)b * Q8_0BlockBytes;
                // Half scale (2 bytes)
                Half scale = (Half)(rng.NextSingle() * 0.5f);
                Unsafe.WriteUnaligned(block, scale);
                // 32 quantized sbytes
                for (int i = 0; i < 32; i++)
                    block[2 + i] = (byte)(rng.Next(256) - 128);
            }
        }
    }

    private static void FillRandomQ5_0Weights(Random rng, byte* weights, int m, int blockCount)
    {
        for (int row = 0; row < m; row++)
        {
            for (int b = 0; b < blockCount; b++)
            {
                byte* block = weights + (long)row * blockCount * Q5_0BlockBytes + (long)b * Q5_0BlockBytes;
                // Q5_0 layout: Half d (2) + uint qh (4) + byte[16] qs
                Half d = (Half)(rng.NextSingle() * 0.5f);
                Unsafe.WriteUnaligned(block, d);
                uint qh = (uint)rng.Next();
                Unsafe.WriteUnaligned(block + 2, qh);
                for (int i = 0; i < 16; i++)
                    block[6 + i] = (byte)rng.Next(256);
            }
        }
    }
}
