using DotLLM.Cpu.Kernels;
using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

/// <summary>
/// Tests for <see cref="AttentionF16"/>.
///
/// CPU reference: <see cref="Attention.ExecuteScalar"/> (FP32) fed with the
/// FP16 inputs converted to FP32 (Half→float is lossless).
///
/// Tolerance: 1e-2f.  Two sources of error vs the FP32 scalar reference:
///   1. Different summation order (tiled GPU vs sequential CPU).
///   2. float→half truncation on the GPU output (~0.1% relative error).
/// </summary>
public sealed class AttentionF16Tests
{
    private const float Tol = 1e-2f;

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static Half[] BuildHalf(int length, int seed)
    {
        var rng = new Random(seed);
        var arr = new Half[length];
        for (int i = 0; i < arr.Length; i++)
            arr[i] = (Half)(rng.NextSingle() * 2f - 1f);
        return arr;
    }

    private static float[] ToFloat(Half[] src)
    {
        var dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
        return dst;
    }

    private static void AssertEqual(float[] expected, Half[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], (float)actual[i], Tol);
    }

    /// <summary>
    /// Runs CPU scalar reference (FP32) and Metal FP16 kernel, returns both outputs.
    /// </summary>
    private static (float[] cpu, Half[] gpu) Run(
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int slidingWindow,
        int seedQ, int seedK, int seedV)
    {
        Half[] qH = BuildHalf(seqQ  * numHeads   * headDim, seedQ);
        Half[] kH = BuildHalf(seqKv * numKvHeads * headDim, seedK);
        Half[] vH = BuildHalf(seqKv * numKvHeads * headDim, seedV);

        // CPU: convert half→float (lossless), then scalar attention.
        float[] cpuOut = new float[seqQ * numHeads * headDim];
        Attention.ExecuteScalar(ToFloat(qH), ToFloat(kH), ToFloat(vH), cpuOut,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
            slidingWindow > 0 ? slidingWindow : (int?)null);

        // GPU: Metal FP16 kernel.
        Half[] gpuOut = new Half[seqQ * numHeads * headDim];
        using var ctx = new MetalContext();
        AttentionF16.Execute(ctx, qH, kH, vH, gpuOut,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow);

        return (cpuOut, gpuOut);
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    [MetalTestFact]
    public void MHA_SingleToken_SingleHead_MatchesCpu()
    {
        var (cpu, gpu) = Run(
            seqQ: 1, seqKv: 1,
            numHeads: 1, numKvHeads: 1, headDim: 32,
            positionOffset: 0, slidingWindow: 0,
            seedQ: 1, seedK: 2, seedV: 3);

        AssertEqual(cpu, gpu);
    }

    [MetalTestFact]
    public void MHA_SingleToken_MultiHead_MatchesCpu()
    {
        var (cpu, gpu) = Run(
            seqQ: 1, seqKv: 8,
            numHeads: 4, numKvHeads: 4, headDim: 64,
            positionOffset: 7, slidingWindow: 0,
            seedQ: 4, seedK: 5, seedV: 6);

        AssertEqual(cpu, gpu);
    }

    [MetalTestFact]
    public void MHA_MultiToken_Prefill_MatchesCpu()
    {
        var (cpu, gpu) = Run(
            seqQ: 8, seqKv: 8,
            numHeads: 4, numKvHeads: 4, headDim: 64,
            positionOffset: 0, slidingWindow: 0,
            seedQ: 7, seedK: 8, seedV: 9);

        AssertEqual(cpu, gpu);
    }

    [MetalTestFact]
    public void MHA_Decode_WithCachedContext_MatchesCpu()
    {
        const int cached = 16;
        var (cpu, gpu) = Run(
            seqQ: 1, seqKv: cached + 1,
            numHeads: 4, numKvHeads: 4, headDim: 64,
            positionOffset: cached, slidingWindow: 0,
            seedQ: 10, seedK: 11, seedV: 12);

        AssertEqual(cpu, gpu);
    }

    [MetalTestFact]
    public void GQA_4to1_MatchesCpu()
    {
        var (cpu, gpu) = Run(
            seqQ: 1, seqKv: 16,
            numHeads: 4, numKvHeads: 1, headDim: 64,
            positionOffset: 15, slidingWindow: 0,
            seedQ: 13, seedK: 14, seedV: 15);

        AssertEqual(cpu, gpu);
    }

    [MetalTestFact]
    public void GQA_8to2_MatchesCpu()
    {
        var (cpu, gpu) = Run(
            seqQ: 1, seqKv: 32,
            numHeads: 8, numKvHeads: 2, headDim: 64,
            positionOffset: 31, slidingWindow: 0,
            seedQ: 16, seedK: 17, seedV: 18);

        AssertEqual(cpu, gpu);
    }

    [MetalTestFact]
    public void SlidingWindow_MatchesCpu()
    {
        // Same off-by-one as AttentionF32Tests: CUDA/Metal masks distance > N,
        // CPU ExecuteScalar uses exactly N tokens visible.
        // Pass cpuWindow = gpuWindow + 1 to align visible sets.
        const int gpuWindow = 4;
        const int cpuWindow = gpuWindow + 1;

        Half[]  qH = BuildHalf(1 * 2 * 64, 19);
        Half[]  kH = BuildHalf(16 * 2 * 64, 20);
        Half[]  vH = BuildHalf(16 * 2 * 64, 21);

        float[] cpuRef = new float[1 * 2 * 64];
        Attention.ExecuteScalar(ToFloat(qH), ToFloat(kH), ToFloat(vH), cpuRef,
            seqQ: 1, seqKv: 16, numHeads: 2, numKvHeads: 2, headDim: 64,
            positionOffset: 15, slidingWindowSize: cpuWindow);

        Half[] gpuOut = new Half[1 * 2 * 64];
        using var ctx = new MetalContext();
        AttentionF16.Execute(ctx, qH, kH, vH, gpuOut,
            seqQ: 1, seqKv: 16, numHeads: 2, numKvHeads: 2, headDim: 64,
            positionOffset: 15, slidingWindow: gpuWindow);

        AssertEqual(cpuRef, gpuOut);
    }

    [MetalTestFact]
    public void LargeHeadDim_128_MatchesCpu()
    {
        var (cpu, gpu) = Run(
            seqQ: 1, seqKv: 32,
            numHeads: 4, numKvHeads: 4, headDim: 128,
            positionOffset: 31, slidingWindow: 0,
            seedQ: 22, seedK: 23, seedV: 24);

        AssertEqual(cpu, gpu);
    }

    [MetalTestFact]
    public void LargeSeqKv_StressesKvTiling_MatchesCpu()
    {
        // seqKv = 512 > TILE_KV = 256: exercises the tile loop.
        var (cpu, gpu) = Run(
            seqQ: 1, seqKv: 512,
            numHeads: 2, numKvHeads: 2, headDim: 64,
            positionOffset: 511, slidingWindow: 0,
            seedQ: 25, seedK: 26, seedV: 27);

        AssertEqual(cpu, gpu);
    }
}
