using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public sealed class PerHeadRmsNormTests
{
    // Scalar CPU reference — no existing CPU kernel for per-head variant.
    // Mirrors per_head_rmsnorm_f32.cu exactly: one (token, head) pair at a time.
    private static float[] CpuExpected(
        float[] qk, float[] weight,
        int numHeads, int headDim, int seqLen, float eps)
    {
        float[] result = (float[])qk.Clone();
        for (int t = 0; t < seqLen; t++)
        for (int h = 0; h < numHeads; h++)
        {
            int offset = t * numHeads * headDim + h * headDim;

            float sumSq = 0f;
            for (int i = 0; i < headDim; i++)
                sumSq += result[offset + i] * result[offset + i];

            float ri = 1f / MathF.Sqrt(sumSq / headDim + eps);

            for (int i = 0; i < headDim; i++)
                result[offset + i] = result[offset + i] * ri * weight[i];
        }
        return result;
    }

    // ── Single token, single head ─────────────────────────────────────────────

    [MetalTestFact]
    public void SingleToken_SingleHead_MatchesCpu()
    {
        const int numHeads = 1, headDim = 8, seqLen = 1;
        const float eps = 1e-5f;
        var rng = new Random(1);

        float[] qk     = new float[seqLen * numHeads * headDim];
        float[] weight = new float[headDim];
        for (int i = 0; i < qk.Length;     i++) qk[i]     = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < weight.Length; i++) weight[i] = rng.NextSingle() + 0.5f;

        float[] expected = CpuExpected(qk, weight, numHeads, headDim, seqLen, eps);

        using var ctx = new MetalContext();
        PerHeadRmsNorm.Execute(ctx, qk, weight, numHeads, headDim, seqLen, eps);

        for (int i = 0; i < qk.Length; i++) Assert.Equal(expected[i], qk[i], 1e-4f);
    }

    // ── Multiple heads per token ──────────────────────────────────────────────

    [MetalTestFact]
    public void SingleToken_MultipleHeads_EachHeadNormalizedIndependently()
    {
        const int numHeads = 4, headDim = 8, seqLen = 1;
        const float eps = 1e-5f;
        var rng = new Random(2);

        float[] qk     = new float[seqLen * numHeads * headDim];
        float[] weight = new float[headDim];
        for (int i = 0; i < qk.Length;     i++) qk[i]     = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < weight.Length; i++) weight[i] = rng.NextSingle() + 0.5f;

        float[] expected = CpuExpected(qk, weight, numHeads, headDim, seqLen, eps);

        using var ctx = new MetalContext();
        PerHeadRmsNorm.Execute(ctx, qk, weight, numHeads, headDim, seqLen, eps);

        for (int i = 0; i < qk.Length; i++) Assert.Equal(expected[i], qk[i], 1e-4f);
    }

    // ── Multiple tokens and heads ─────────────────────────────────────────────

    [MetalTestFact]
    public void MultipleTokens_MultipleHeads_MatchesCpu()
    {
        const int numHeads = 4, headDim = 16, seqLen = 3;
        const float eps = 1e-5f;
        var rng = new Random(3);

        float[] qk     = new float[seqLen * numHeads * headDim];
        float[] weight = new float[headDim];
        for (int i = 0; i < qk.Length;     i++) qk[i]     = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < weight.Length; i++) weight[i] = rng.NextSingle() + 0.5f;

        float[] expected = CpuExpected(qk, weight, numHeads, headDim, seqLen, eps);

        using var ctx = new MetalContext();
        PerHeadRmsNorm.Execute(ctx, qk, weight, numHeads, headDim, seqLen, eps);

        for (int i = 0; i < qk.Length; i++) Assert.Equal(expected[i], qk[i], 1e-4f);
    }

    // ── Each head is normalized independently ─────────────────────────────────
    // Two heads with different magnitudes must each have rms ≈ 1 after normalization
    // (when weight = 1). We verify rms(head) ≈ 1 for each head individually.

    [MetalTestFact]
    public void EachHead_HasUnitRmsAfterNorm_WhenWeightIsOne()
    {
        const int numHeads = 2, headDim = 8, seqLen = 1;
        const float eps = 1e-5f;

        // Head 0: small values, head 1: large values — different magnitudes
        float[] qk = new float[numHeads * headDim];
        for (int i = 0; i < headDim; i++) qk[i]            = 0.1f * (i + 1);
        for (int i = 0; i < headDim; i++) qk[headDim + i]  = 100f * (i + 1);
        float[] weight = Enumerable.Repeat(1f, headDim).ToArray();

        using var ctx = new MetalContext();
        PerHeadRmsNorm.Execute(ctx, qk, weight, numHeads, headDim, seqLen, eps);

        for (int h = 0; h < numHeads; h++)
        {
            float sumSq = 0f;
            for (int i = 0; i < headDim; i++)
            {
                float v = qk[h * headDim + i];
                sumSq += v * v;
            }
            float rms = MathF.Sqrt(sumSq / headDim);
            Assert.Equal(1f, rms, 1e-4f); // each head independently normalized
        }
    }

    // ── Llama-scale inputs (GQA style) ───────────────────────────────────────

    [MetalTestFact]
    public void LargeInput_MatchesCpu()
    {
        const int numHeads = 32, headDim = 128, seqLen = 4;
        const float eps = 1e-5f;
        var rng = new Random(42);

        float[] qk     = new float[seqLen * numHeads * headDim];
        float[] weight = new float[headDim];
        for (int i = 0; i < qk.Length;     i++) qk[i]     = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < weight.Length; i++) weight[i] = rng.NextSingle() + 0.5f;

        float[] expected = CpuExpected(qk, weight, numHeads, headDim, seqLen, eps);

        using var ctx = new MetalContext();
        PerHeadRmsNorm.Execute(ctx, qk, weight, numHeads, headDim, seqLen, eps);

        for (int i = 0; i < qk.Length; i++) Assert.Equal(expected[i], qk[i], 1e-4f);
    }
}

// ── Per-head RMSNorm FP16 ─────────────────────────────────────────────────────

public sealed class PerHeadRmsNormF16Tests
{
    // FP16 ULP at typical values (~1) is ~0.001; use relative-ish absolute tol 1e-2f.
    private const float Tol = 1e-2f;

    // CPU reference: same scalar loop as PerHeadRmsNormTests.CpuExpected, but on
    // Half arrays (convert Half→float, normalize, return Half[]).
    private static Half[] CpuExpected(
        Half[] qk, Half[] weight,
        int numHeads, int headDim, int seqLen, float eps)
    {
        float[] result = Array.ConvertAll(qk, h => (float)h);
        float[] wF     = Array.ConvertAll(weight, h => (float)h);

        for (int t = 0; t < seqLen; t++)
        for (int h = 0; h < numHeads; h++)
        {
            int offset = t * numHeads * headDim + h * headDim;

            float sumSq = 0f;
            for (int i = 0; i < headDim; i++)
                sumSq += result[offset + i] * result[offset + i];

            float ri = 1f / MathF.Sqrt(sumSq / headDim + eps);

            for (int i = 0; i < headDim; i++)
                result[offset + i] = result[offset + i] * ri * wF[i];
        }
        return Array.ConvertAll(result, v => (Half)v);
    }

    private static void AssertClose(Half[] expected, Half[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal((float)expected[i], (float)actual[i], Tol);
    }

    // ── Single token, single head ─────────────────────────────────────────────

    [MetalTestFact]
    public void SingleToken_SingleHead_MatchesCpu()
    {
        const int numHeads = 1, headDim = 8, seqLen = 1;
        const float eps = 1e-5f;
        var rng = new Random(1);

        Half[] qk     = new Half[seqLen * numHeads * headDim];
        Half[] weight = new Half[headDim];
        for (int i = 0; i < qk.Length;     i++) qk[i]     = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < weight.Length; i++) weight[i] = (Half)(rng.NextSingle() + 0.5f);

        Half[] expected = CpuExpected(qk, weight, numHeads, headDim, seqLen, eps);
        Half[] qkCopy   = (Half[])qk.Clone();

        using var ctx = new MetalContext();
        PerHeadRmsNormF16.Execute(ctx, qkCopy, weight, numHeads, headDim, seqLen, eps);

        AssertClose(expected, qkCopy);
    }

    // ── Multiple heads per token ──────────────────────────────────────────────

    [MetalTestFact]
    public void SingleToken_MultipleHeads_EachHeadNormalizedIndependently()
    {
        const int numHeads = 4, headDim = 8, seqLen = 1;
        const float eps = 1e-5f;
        var rng = new Random(2);

        Half[] qk     = new Half[seqLen * numHeads * headDim];
        Half[] weight = new Half[headDim];
        for (int i = 0; i < qk.Length;     i++) qk[i]     = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < weight.Length; i++) weight[i] = (Half)(rng.NextSingle() + 0.5f);

        Half[] expected = CpuExpected(qk, weight, numHeads, headDim, seqLen, eps);
        Half[] qkCopy   = (Half[])qk.Clone();

        using var ctx = new MetalContext();
        PerHeadRmsNormF16.Execute(ctx, qkCopy, weight, numHeads, headDim, seqLen, eps);

        AssertClose(expected, qkCopy);
    }

    // ── Multiple tokens and heads ─────────────────────────────────────────────

    [MetalTestFact]
    public void MultipleTokens_MultipleHeads_MatchesCpu()
    {
        const int numHeads = 4, headDim = 16, seqLen = 3;
        const float eps = 1e-5f;
        var rng = new Random(3);

        Half[] qk     = new Half[seqLen * numHeads * headDim];
        Half[] weight = new Half[headDim];
        for (int i = 0; i < qk.Length;     i++) qk[i]     = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < weight.Length; i++) weight[i] = (Half)(rng.NextSingle() + 0.5f);

        Half[] expected = CpuExpected(qk, weight, numHeads, headDim, seqLen, eps);
        Half[] qkCopy   = (Half[])qk.Clone();

        using var ctx = new MetalContext();
        PerHeadRmsNormF16.Execute(ctx, qkCopy, weight, numHeads, headDim, seqLen, eps);

        AssertClose(expected, qkCopy);
    }

    // ── Each head independently normalized (unit-rms check) ───────────────────

    [MetalTestFact]
    public void EachHead_HasUnitRmsAfterNorm_WhenWeightIsOne()
    {
        const int numHeads = 2, headDim = 8, seqLen = 1;
        const float eps = 1e-5f;

        Half[] qk = new Half[numHeads * headDim];
        for (int i = 0; i < headDim; i++) qk[i]           = (Half)(0.1f * (i + 1));
        for (int i = 0; i < headDim; i++) qk[headDim + i] = (Half)(100f * (i + 1));
        Half[] weight = Enumerable.Repeat((Half)1f, headDim).ToArray();

        using var ctx = new MetalContext();
        PerHeadRmsNormF16.Execute(ctx, qk, weight, numHeads, headDim, seqLen, eps);

        for (int h = 0; h < numHeads; h++)
        {
            float sumSq = 0f;
            for (int i = 0; i < headDim; i++)
            {
                float v = (float)qk[h * headDim + i];
                sumSq += v * v;
            }
            float rms = MathF.Sqrt(sumSq / headDim);
            Assert.Equal(1f, rms, Tol);
        }
    }

    // ── Llama-scale inputs ────────────────────────────────────────────────────

    [MetalTestFact]
    public void LargeInput_MatchesCpu()
    {
        const int numHeads = 32, headDim = 128, seqLen = 4;
        const float eps = 1e-5f;
        var rng = new Random(42);

        Half[] qk     = new Half[seqLen * numHeads * headDim];
        Half[] weight = new Half[headDim];
        for (int i = 0; i < qk.Length;     i++) qk[i]     = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < weight.Length; i++) weight[i] = (Half)(rng.NextSingle() + 0.5f);

        Half[] expected = CpuExpected(qk, weight, numHeads, headDim, seqLen, eps);
        Half[] qkCopy   = (Half[])qk.Clone();

        using var ctx = new MetalContext();
        PerHeadRmsNormF16.Execute(ctx, qkCopy, weight, numHeads, headDim, seqLen, eps);

        AssertClose(expected, qkCopy);
    }
}
