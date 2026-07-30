using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Parity tests for <c>MatMul.VecDotQ8_0Vnni_4RowsR4</c>, the AVX-VNNI kernel for the
/// R4-interleaved Q8_0 layout (#414). The kernel is correct but measured slower than its AVX2
/// sibling, so it is not the dispatched default; these tests keep it correct for anyone re-running
/// the comparison on other hardware, and pin the dispatch that decision produced.
/// </summary>
/// <remarks>
/// Block counts are chosen to discriminate rather than to be round: the VNNI kernel consumes
/// blocks in pairs, so odd counts (1, 3, 17) exercise the single-block AVX2 tail and even counts
/// (2, 18, 48, 128) exercise only the paired loop. 1 is the degenerate case where the paired loop
/// never runs at all, and 128 spans enough K to expose any accumulator drift.
/// </remarks>
public sealed unsafe class MatMulR4VnniTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    /// <summary>True when the hardware can execute <c>VecDotQ8_0Vnni_4RowsR4</c> at all.</summary>
    private static bool VnniAvailable => Avx512BW.IsSupported && AvxVnni.IsSupported;

    public static TheoryData<int> DiscriminatingBlockCounts() => [1, 2, 3, 8, 17, 18, 48, 128];

    [Theory]
    [MemberData(nameof(DiscriminatingBlockCounts))]
    public void VecDotQ8_0Vnni_4RowsR4_MatchesScalarR4(int blockCount)
    {
        if (!VnniAvailable) return;   // dispatch wiring is pinned by DispatchPrefersAvx2Tier

        RunSingleGroup(blockCount, seed: 42, out float[] vnni, out float[] scalar, out _);

        for (int r = 0; r < 4; r++)
            AssertClose(scalar[r], vnni[r], blockCount, $"row {r}");
    }

    [Theory]
    [MemberData(nameof(DiscriminatingBlockCounts))]
    public void VecDotQ8_0Vnni_4RowsR4_MatchesAvx2R4(int blockCount)
    {
        if (!VnniAvailable) return;   // dispatch wiring is pinned by DispatchPrefersAvx2Tier

        RunSingleGroup(blockCount, seed: 7, out float[] vnni, out _, out float[] avx2);

        for (int r = 0; r < 4; r++)
            AssertClose(avx2[r], vnni[r], blockCount, $"row {r}");
    }

    /// <summary>
    /// Self-check: the parity assertions above are only meaningful if they can fail. Perturbing a
    /// single weight byte in one row must change that row's result and leave the other three
    /// untouched — which also proves the kernel reads each row from its own interleaved offset
    /// rather than accidentally aliasing one row four times.
    /// </summary>
    [Theory]
    [MemberData(nameof(DiscriminatingBlockCounts))]
    public void VecDotQ8_0Vnni_4RowsR4_DetectsPerturbedWeight(int blockCount)
    {
        if (!VnniAvailable) return;   // dispatch wiring is pinned by DispatchPrefersAvx2Tier

        const int perturbedRow = 2;
        int k = blockCount * Q8_0GroupSize;

        byte* weights = AllocQ8_0Rows(new Random(11), 4, blockCount, out nuint weightBytes);
        try
        {
            byte* xQ8 = stackalloc byte[blockCount * Q8_0BlockBytes];
            QuantizeRandomInput(new Random(12), xQ8, k);

            float[] baseline = new float[4];
            float[] perturbed = new float[4];

            using (var rw = WeightRepacking.RepackR4((nint)weights, QuantizationType.Q8_0, 4, k))
            fixed (float* res = baseline)
                MatMul.VecDotQ8_0Vnni_4RowsR4((byte*)rw.Ptr, xQ8, blockCount, res);

            // Flip a quant in the last block of the perturbed row, past the paired loop's reach on
            // odd block counts so the tail path is covered too.
            byte* target = weights + ((long)perturbedRow * blockCount + (blockCount - 1)) * Q8_0BlockBytes + 2;
            *target = (byte)(*target ^ 0x7F);

            using (var rw = WeightRepacking.RepackR4((nint)weights, QuantizationType.Q8_0, 4, k))
            fixed (float* res = perturbed)
                MatMul.VecDotQ8_0Vnni_4RowsR4((byte*)rw.Ptr, xQ8, blockCount, res);

            Assert.NotEqual(baseline[perturbedRow], perturbed[perturbedRow]);
            for (int r = 0; r < 4; r++)
            {
                if (r == perturbedRow) continue;
                Assert.Equal(baseline[r], perturbed[r]);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
            _ = weightBytes;
        }
    }

    /// <summary>
    /// Dispatch-level parity: whichever tier <c>ComputeRowsQ8_0Interleaved</c> selects must agree
    /// with the row-major reference. Shapes include tail rows (m % 4 != 0) and odd block counts,
    /// which the pre-existing coverage in <c>WeightRepackingTests</c> does not reach.
    /// </summary>
    [Theory]
    [InlineData(4, 1)]      // 1 group, no tail, degenerate K
    [InlineData(8, 3)]      // 2 groups, odd blocks
    [InlineData(9, 17)]     // 2 groups + 1 tail row, odd blocks
    [InlineData(11, 18)]    // 2 groups + 3 tail rows, even blocks
    [InlineData(16, 48)]    // 4 groups, no tail
    [InlineData(13, 128)]   // 3 groups + 1 tail row, long K
    public void ComputeRowsQ8_0Interleaved_MatchesRowMajor(int m, int blockCount)
    {
        int k = blockCount * Q8_0GroupSize;

        byte* weights = AllocQ8_0Rows(new Random(1234), m, blockCount, out _);
        try
        {
            byte* xQ8 = stackalloc byte[blockCount * Q8_0BlockBytes];
            QuantizeRandomInput(new Random(5678), xQ8, k);

            float[] rowMajor = new float[m];
            float[] interleaved = new float[m];

            fixed (float* res = rowMajor)
                MatMul.ComputeRows(weights, xQ8, res, m, blockCount);

            using var rw = WeightRepacking.RepackR4((nint)weights, QuantizationType.Q8_0, m, k);
            fixed (float* res = interleaved)
                MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xQ8, res,
                    rw.FullGroupCount, rw.TailRows, blockCount);

            for (int i = 0; i < m; i++)
                AssertClose(rowMajor[i], interleaved[i], blockCount, $"row {i}");
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
        }
    }

    /// <summary>
    /// Pins the dispatch choice. <c>ComputeRowsQ8_0Interleaved</c> deliberately prefers the AVX2
    /// kernel over the wider VNNI one, which measured 1.7-1.9x slower — so the selected tier must
    /// be bit-identical to <c>VecDotQ8_0Avx2_4RowsR4</c>. A difference of even one ULP means the
    /// dispatch changed, which given the measurements would be a performance regression rather
    /// than a correctness one, and would otherwise go unnoticed.
    /// </summary>
    [Theory]
    [MemberData(nameof(DiscriminatingBlockCounts))]
    public void ComputeRowsQ8_0Interleaved_DispatchPrefersAvx2Tier(int blockCount)
    {
        if (!Avx2.IsSupported) return;

        int k = blockCount * Q8_0GroupSize;

        byte* weights = AllocQ8_0Rows(new Random(99), 4, blockCount, out _);
        try
        {
            byte* xQ8 = stackalloc byte[blockCount * Q8_0BlockBytes];
            QuantizeRandomInput(new Random(100), xQ8, k);

            float[] viaDispatch = new float[4];
            float[] viaAvx2 = new float[4];

            using var rw = WeightRepacking.RepackR4((nint)weights, QuantizationType.Q8_0, 4, k);

            fixed (float* res = viaDispatch)
                MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xQ8, res,
                    rw.FullGroupCount, rw.TailRows, blockCount);

            fixed (float* res = viaAvx2)
                MatMul.VecDotQ8_0Avx2_4RowsR4((byte*)rw.Ptr, xQ8, blockCount, res);

            for (int r = 0; r < 4; r++)
                Assert.Equal(viaAvx2[r], viaDispatch[r]);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
        }
    }

    // ──────────────────── Helpers ────────────────────

    /// <summary>
    /// Runs all three single-group (4-row) Q8_0 R4 kernels over the same random inputs.
    /// </summary>
    private static void RunSingleGroup(int blockCount, int seed,
        out float[] vnni, out float[] scalar, out float[] avx2)
    {
        int k = blockCount * Q8_0GroupSize;
        vnni = new float[4];
        scalar = new float[4];
        avx2 = new float[4];

        byte* weights = AllocQ8_0Rows(new Random(seed), 4, blockCount, out _);
        try
        {
            byte* xQ8 = stackalloc byte[blockCount * Q8_0BlockBytes];
            QuantizeRandomInput(new Random(seed + 1), xQ8, k);

            using var rw = WeightRepacking.RepackR4((nint)weights, QuantizationType.Q8_0, 4, k);
            Assert.Equal(1, rw.FullGroupCount);
            Assert.Equal(0, rw.TailRows);

            byte* groupBase = (byte*)rw.Ptr;

            fixed (float* res = vnni)
                MatMul.VecDotQ8_0Vnni_4RowsR4(groupBase, xQ8, blockCount, res);

            fixed (float* res = avx2)
                MatMul.VecDotQ8_0Avx2_4RowsR4(groupBase, xQ8, blockCount, res);

            for (int r = 0; r < 4; r++)
                scalar[r] = MatMul.VecDotQ8_0ScalarR4(groupBase, r, xQ8, blockCount);
        }
        finally
        {
            NativeMemory.AlignedFree(weights);
        }
    }

    /// <summary>
    /// Allocates random row-major Q8_0 weights: a <see cref="Half"/> scale plus 32 quants per block.
    /// </summary>
    /// <remarks>
    /// Quants are drawn from [-127, 127], not the full sbyte range. This matches the format as
    /// actually produced — Q8_0 quantization is <c>round(x / (max|x| / 127))</c>, so -128 never
    /// occurs — and it matters here because the vectorized kernels reduce signed products via the
    /// abs/sign idiom (<c>Avx2.Sign(vx, vx)</c> paired with <c>Avx2.Sign(vw, vx)</c>). |-128| is not
    /// representable in an sbyte, so a -128 quant would make the SIMD paths disagree with exact
    /// scalar arithmetic on input the format cannot contain.
    /// </remarks>
    private static byte* AllocQ8_0Rows(Random rng, int m, int blockCount, out nuint byteCount)
    {
        byteCount = (nuint)((long)m * blockCount * Q8_0BlockBytes);
        byte* weights = (byte*)NativeMemory.AlignedAlloc(byteCount, 64);

        for (int row = 0; row < m; row++)
        {
            for (int b = 0; b < blockCount; b++)
            {
                byte* block = weights + ((long)row * blockCount + b) * Q8_0BlockBytes;
                Unsafe.WriteUnaligned(block, (Half)(rng.NextSingle() * 0.5f));
                for (int i = 0; i < Q8_0GroupSize; i++)
                    block[2 + i] = (byte)(sbyte)rng.Next(-127, 128);
            }
        }
        return weights;
    }

    private static void QuantizeRandomInput(Random rng, byte* xQ8, int k)
    {
        float[] xF32 = new float[k];
        for (int i = 0; i < k; i++) xF32[i] = rng.NextSingle() * 2f - 1f;
        fixed (float* xp = xF32)
            MatMul.QuantizeF32ToQ8_0(xp, xQ8, k);
    }

    /// <summary>
    /// Compares against a relative tolerance. The integer reduction is exact on every path, so the
    /// only divergence is FP32 reassociation of the per-block <c>dw*dx*isum</c> terms across
    /// different accumulator widths; that error grows with block count, hence the sqrt scaling.
    /// </summary>
    private static void AssertClose(float expected, float actual, int blockCount, string because)
    {
        float tolerance = 1e-5f * MathF.Sqrt(blockCount) * MathF.Max(1f, MathF.Abs(expected));
        Assert.True(MathF.Abs(expected - actual) <= tolerance,
            $"{because}: expected {expected:R}, got {actual:R} " +
            $"(delta {MathF.Abs(expected - actual):R} > tolerance {tolerance:R})");
    }
}
