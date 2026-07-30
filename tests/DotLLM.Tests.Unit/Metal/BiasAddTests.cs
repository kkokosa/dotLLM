using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

// ── BiasAddF32 ────────────────────────────────────────────────────────────────
// output_f32[t, i] += float(bias_f16[i])  — port of bias_add_f32.cu

public sealed class BiasAddF32Tests
{
    // Build a Half[] from float values (exact for small integers / powers of 2)
    private static Half[] H(params float[] values) =>
        Array.ConvertAll(values, v => (Half)v);

    /// <summary>Scalar CPU reference matching bias_add_f32.cu logic.</summary>
    private static float[] CpuRef(float[] output, Half[] bias, int dim, int seqLen)
    {
        var result = (float[])output.Clone();
        for (int t = 0; t < seqLen; t++)
            for (int i = 0; i < dim; i++)
                result[t * dim + i] += (float)bias[i];
        return result;
    }

    [MetalTestFact]
    public void SingleToken_BiasIsAdded()
    {
        float[] output = [1.0f, 2.0f, 3.0f, 4.0f];
        Half[]  bias   = H(10.0f, 20.0f, 30.0f, 40.0f);

        using var ctx = new MetalContext();
        BiasAddF32.Execute(ctx, output, bias, dim: 4, seqLen: 1);

        Assert.Equal([11.0f, 22.0f, 33.0f, 44.0f], output);
    }

    [MetalTestFact]
    public void MultipleTokens_SameBiasAppliedToEach()
    {
        // seqLen=3, dim=2
        // token 0: [1, 2], token 1: [3, 4], token 2: [5, 6]
        // bias: [10, 20]
        float[] output = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f];
        Half[]  bias   = H(10.0f, 20.0f);

        using var ctx = new MetalContext();
        BiasAddF32.Execute(ctx, output, bias, dim: 2, seqLen: 3);

        Assert.Equal([11.0f, 22.0f, 13.0f, 24.0f, 15.0f, 26.0f], output);
    }

    [MetalTestFact]
    public void ZeroBias_OutputUnchanged()
    {
        float[] output   = [1.0f, 2.0f, 3.0f, 4.0f];
        Half[]  bias     = H(0.0f, 0.0f, 0.0f, 0.0f);
        float[] expected = [1.0f, 2.0f, 3.0f, 4.0f];

        using var ctx = new MetalContext();
        BiasAddF32.Execute(ctx, output, bias, dim: 4, seqLen: 1);

        Assert.Equal(expected, output);
    }

    [MetalTestFact]
    public void NegativeBias_SubtractsCorrectly()
    {
        float[] output = [5.0f, 5.0f, 5.0f];
        Half[]  bias   = H(-1.0f, -2.0f, -3.0f);

        using var ctx = new MetalContext();
        BiasAddF32.Execute(ctx, output, bias, dim: 3, seqLen: 1);

        Assert.Equal([4.0f, 3.0f, 2.0f], output);
    }

    [MetalTestFact]
    public void ScalarReference_MatchesMetal()
    {
        var rng = new Random(42);
        const int dim    = 512;
        const int seqLen = 8;

        float[] output = new float[dim * seqLen];
        Half[]  bias   = new Half[dim];

        for (int i = 0; i < output.Length; i++) output[i] = rng.NextSingle() * 10f - 5f;
        for (int i = 0; i < bias.Length;   i++) bias[i]   = (Half)(rng.NextSingle() * 2f - 1f);

        float[] expected = CpuRef(output, bias, dim, seqLen);

        using var ctx = new MetalContext();
        BiasAddF32.Execute(ctx, output, bias, dim, seqLen);

        for (int i = 0; i < output.Length; i++)
            Assert.Equal(expected[i], output[i], 1e-2f); // FP16 bias → FP32 cast ~0.1% error
    }

    [MetalTestFact]
    public void BiasMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            BiasAddF32.Execute(ctx, new float[8], H(0f, 0f, 0f), dim: 4, seqLen: 2));
    }

    [MetalTestFact]
    public void OutputSizeMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            BiasAddF32.Execute(ctx, new float[7], H(0f, 0f, 0f, 0f), dim: 4, seqLen: 2));
    }

    [MetalTestFact]
    public void EmptyOutput_ReturnsSilently()
    {
        using var ctx = new MetalContext();
        BiasAddF32.Execute(ctx, [], [], dim: 0, seqLen: 0);
    }
}

// ── BiasAddF16 ────────────────────────────────────────────────────────────────
// output_f16[t, i] += bias_f16[i]  — port of bias_add.cu, vectorized half2

public sealed class BiasAddF16Tests
{
    private static Half[] H(params float[] values) =>
        Array.ConvertAll(values, v => (Half)v);

    /// <summary>Scalar CPU reference matching bias_add.cu logic.</summary>
    private static Half[] CpuRef(Half[] output, Half[] bias, int dim, int seqLen)
    {
        var result = (Half[])output.Clone();
        for (int t = 0; t < seqLen; t++)
            for (int i = 0; i < dim; i++)
                result[t * dim + i] = (Half)((float)result[t * dim + i] + (float)bias[i]);
        return result;
    }

    [MetalTestFact]
    public void SingleToken_BiasIsAdded()
    {
        Half[] output = H(1.0f, 2.0f, 3.0f, 4.0f);
        Half[] bias   = H(10.0f, 20.0f, 30.0f, 40.0f);

        using var ctx = new MetalContext();
        BiasAddF16.Execute(ctx, output, bias, dim: 4, seqLen: 1);

        Assert.Equal(H(11.0f, 22.0f, 33.0f, 44.0f), output);
    }

    [MetalTestFact]
    public void MultipleTokens_SameBiasAppliedToEach()
    {
        // seqLen=3, dim=4 (even, exercises half2 path)
        Half[] output = H(1f, 2f, 3f, 4f,   5f, 6f, 7f, 8f,   9f, 10f, 11f, 12f);
        Half[] bias   = H(10f, 20f, 30f, 40f);

        using var ctx = new MetalContext();
        BiasAddF16.Execute(ctx, output, bias, dim: 4, seqLen: 3);

        Half[] expected = H(11f, 22f, 33f, 44f,   15f, 26f, 37f, 48f,   19f, 30f, 41f, 52f);
        Assert.Equal(expected, output);
    }

    [MetalTestFact]
    public void ZeroBias_OutputUnchanged()
    {
        Half[] output   = H(1.0f, 2.0f, 3.0f, 4.0f);
        Half[] bias     = H(0.0f, 0.0f, 0.0f, 0.0f);
        Half[] expected = H(1.0f, 2.0f, 3.0f, 4.0f);

        using var ctx = new MetalContext();
        BiasAddF16.Execute(ctx, output, bias, dim: 4, seqLen: 1);

        Assert.Equal(expected, output);
    }

    [MetalTestFact]
    public void NegativeBias_SubtractsCorrectly()
    {
        Half[] output = H(5.0f, 5.0f, 5.0f, 5.0f);
        Half[] bias   = H(-1.0f, -2.0f, -3.0f, -4.0f);

        using var ctx = new MetalContext();
        BiasAddF16.Execute(ctx, output, bias, dim: 4, seqLen: 1);

        Assert.Equal(H(4.0f, 3.0f, 2.0f, 1.0f), output);
    }

    [MetalTestFact]
    public void ScalarReference_MatchesMetal()
    {
        var rng = new Random(99);
        const int dim    = 256; // even — exercises half2 path
        const int seqLen = 4;

        Half[] output = new Half[dim * seqLen];
        Half[] bias   = new Half[dim];

        for (int i = 0; i < output.Length; i++) output[i] = (Half)(rng.NextSingle() * 4f - 2f);
        for (int i = 0; i < bias.Length;   i++) bias[i]   = (Half)(rng.NextSingle() * 2f - 1f);

        Half[] expected = CpuRef(output, bias, dim, seqLen);

        using var ctx = new MetalContext();
        BiasAddF16.Execute(ctx, output, bias, dim, seqLen);

        const float tol = 1e-2f;
        for (int i = 0; i < output.Length; i++)
            Assert.Equal((float)expected[i], (float)output[i], tol);
    }

    [MetalTestFact]
    public void BiasMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            BiasAddF16.Execute(ctx, new Half[8], H(0f, 0f, 0f), dim: 4, seqLen: 2));
    }

    [MetalTestFact]
    public void OutputSizeMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            BiasAddF16.Execute(ctx, new Half[7], H(0f, 0f, 0f, 0f), dim: 4, seqLen: 2));
    }

    [MetalTestFact]
    public void EmptyOutput_ReturnsSilently()
    {
        using var ctx = new MetalContext();
        BiasAddF16.Execute(ctx, [], [], dim: 0, seqLen: 0);
    }
}
