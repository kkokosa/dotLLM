using DotLLM.Cpu.Kernels;
using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

/// <summary>
/// Tests for <see cref="AttentionF32"/>.
///
/// CPU reference: <see cref="Attention.ExecuteScalar"/> — a plain scalar
/// implementation of scaled dot-product attention with causal masking and GQA.
///
/// Tolerance: 1e-3f.  The GPU kernel accumulates in a different summation order
/// than the CPU scalar reference (tiled with simd_sum vs. sequential), so a
/// small relative tolerance is expected and acceptable.
/// </summary>
public sealed class AttentionF32Tests
{
    // ── Helpers ───────────────────────────────────────────────────────────────

    private const float Tol = 1e-3f;

    private static float[] BuildRandom(int length, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var arr = new float[length];
        for (int i = 0; i < arr.Length; i++)
            arr[i] = (rng.NextSingle() * 2f - 1f) * scale;
        return arr;
    }

    private static void AssertEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], Tol);
    }

    private static (float[] cpu, float[] gpu) Run(
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
        int positionOffset, int slidingWindow,
        int seedQ, int seedK, int seedV)
    {
        float[] q = BuildRandom(seqQ  * numHeads    * headDim, seedQ);
        float[] k = BuildRandom(seqKv * numKvHeads  * headDim, seedK);
        float[] v = BuildRandom(seqKv * numKvHeads  * headDim, seedV);

        float[] cpuOut = new float[seqQ * numHeads * headDim];
        Attention.ExecuteScalar(q, k, v, cpuOut,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);

        float[] gpuOut = new float[seqQ * numHeads * headDim];
        using var ctx = new MetalContext();
        AttentionF32.Execute(ctx, q, k, v, gpuOut,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow);

        return (cpuOut, gpuOut);
    }

    // ── Basic MHA tests ───────────────────────────────────────────────────────

    [MetalTestFact]
    public void MHA_SingleToken_SingleHead_MatchesCpu()
    {
        // Simplest possible case: seqQ=1, seqKv=1, 1 head.
        // Only one KV token, always visible (pos_q=0, tkv=0, 0<=0 ✓).
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
        // Prefill: seqQ == seqKv, positionOffset = 0.
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
        // Decode step: seqQ=1, seqKv=cached+1, positionOffset=cached.
        const int cached = 16;
        var (cpu, gpu) = Run(
            seqQ: 1, seqKv: cached + 1,
            numHeads: 4, numKvHeads: 4, headDim: 64,
            positionOffset: cached, slidingWindow: 0,
            seedQ: 10, seedK: 11, seedV: 12);

        AssertEqual(cpu, gpu);
    }

    // ── GQA tests ─────────────────────────────────────────────────────────────

    [MetalTestFact]
    public void GQA_4to1_MatchesCpu()
    {
        // 4 query heads sharing 1 KV head (MQA).
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
        // 8 query heads, 2 KV heads — each KV head shared by 4 query heads.
        var (cpu, gpu) = Run(
            seqQ: 1, seqKv: 32,
            numHeads: 8, numKvHeads: 2, headDim: 64,
            positionOffset: 31, slidingWindow: 0,
            seedQ: 16, seedK: 17, seedV: 18);

        AssertEqual(cpu, gpu);
    }

    // ── Sliding window tests ───────────────────────────────────────────────────

    [MetalTestFact]
    public void SlidingWindow_LimitsAttendedTokens_MatchesCpu()
    {
        // CUDA/Metal convention: mask if (pos_q - tkv > sliding_window)
        //   → tokens at distance [0..sliding_window] are visible (sliding_window+1 tokens).
        // CPU ExecuteScalar convention: earliestVisible = pos_q - slidingWindowSize + 1
        //   → exactly slidingWindowSize tokens visible.
        // To align both on the same visible set, pass slidingWindowSize = gpuWindow + 1 to CPU.
        const int gpuWindow = 4; // Metal: distance > 4 masked → tokens 11..15 visible (5 tokens)
        const int cpuWindow = gpuWindow + 1; // CPU: 5 tokens visible → matches CUDA/Metal

        float[] q = BuildRandom(1 * 2 * 64, 19);
        float[] k = BuildRandom(16 * 2 * 64, 20);
        float[] v = BuildRandom(16 * 2 * 64, 21);

        float[] cpuRef = new float[1 * 2 * 64];
        Attention.ExecuteScalar(q, k, v, cpuRef,
            seqQ: 1, seqKv: 16, numHeads: 2, numKvHeads: 2, headDim: 64,
            positionOffset: 15, slidingWindowSize: cpuWindow);

        float[] gpuOut = new float[1 * 2 * 64];
        using var ctx = new MetalContext();
        AttentionF32.Execute(ctx, q, k, v, gpuOut,
            seqQ: 1, seqKv: 16, numHeads: 2, numKvHeads: 2, headDim: 64,
            positionOffset: 15, slidingWindow: gpuWindow);

        AssertEqual(cpuRef, gpuOut);
    }

    // ── Large head dim — stresses the grid-stride loops ──────────────────────

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

    // ── Edge cases ────────────────────────────────────────────────────────────

    [MetalTestFact]
    public void AllMasked_SingleTokenSees_OnlySelf()
    {
        // seqQ=1, seqKv=4, positionOffset=0 → only token 0 is visible, tokens 1-3 masked.
        // The query at position 0 can only attend to KV position 0.
        // Output must equal V[0] (single softmax weight = 1).
        const int headDim = 32;
        float[] q = BuildRandom(headDim, seed: 30);
        float[] k = BuildRandom(4 * headDim, seed: 31);
        float[] v = BuildRandom(4 * headDim, seed: 32);

        float[] cpuOut = new float[headDim];
        float[] gpuOut = new float[headDim];

        Attention.ExecuteScalar(q, k, v, cpuOut,
            seqQ: 1, seqKv: 4, numHeads: 1, numKvHeads: 1, headDim: headDim,
            positionOffset: 0);

        using var ctx = new MetalContext();
        AttentionF32.Execute(ctx, q, k, v, gpuOut,
            seqQ: 1, seqKv: 4, numHeads: 1, numKvHeads: 1, headDim: headDim,
            positionOffset: 0);

        // When only one token is visible the output is exactly V[0].
        for (int d = 0; d < headDim; d++)
            Assert.Equal(v[d], gpuOut[d], Tol);
        AssertEqual(cpuOut, gpuOut);
    }
}
