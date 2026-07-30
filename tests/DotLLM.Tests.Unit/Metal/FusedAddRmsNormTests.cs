using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public sealed class FusedAddRmsNormTests
{
    // Scalar CPU reference — mirrors fused_add_rmsnorm_f16.cu exactly, including
    // the FP16 truncation of the residual between Pass 1 and Pass 2.
    private static (Half[] residualOut, Half[] output) CpuExpected(
        Half[] residual, Half[] x, Half[] weight, int n, int seqLen, float eps)
    {
        Half[] res = (Half[])residual.Clone();
        Half[] out_ = new Half[seqLen * n];

        for (int row = 0; row < seqLen; row++)
        {
            int offset = row * n;

            // Pass 1: add in FP32, store sum back as FP16, accumulate sum²
            float sumSq = 0f;
            for (int i = 0; i < n; i++)
            {
                float r   = (float)res[offset + i];
                float xi  = (float)x[offset + i];
                float sum = r + xi;
                res[offset + i] = (Half)sum;   // FP16 truncation here
                sumSq += sum * sum;            // accumulated from FP32 sum
            }

            // Pass 2: normalize — reads from residual (already truncated to FP16)
            float rmsInv = 1f / MathF.Sqrt(sumSq / n + eps);
            for (int i = 0; i < n; i++)
            {
                float v = (float)res[offset + i];
                float w = (float)weight[i];
                out_[offset + i] = (Half)(v * rmsInv * w);
            }
        }

        return (res, out_);
    }

    // ── Single token ─────────────────────────────────────────────────────────

    [MetalTestFact]
    public void SingleToken_OutputMatchesCpu()
    {
        const int n = 8, seqLen = 1;
        const float eps = 1e-5f;
        var rng = new Random(1);

        Half[] residual = Rand(rng, n);
        Half[] x        = Rand(rng, n);
        Half[] weight   = RandPositive(rng, n);
        Half[] output   = new Half[n];

        var (expRes, expOut) = CpuExpected(residual, x, weight, n, seqLen, eps);
        Half[] residualMetal = (Half[])residual.Clone();

        using var ctx = new MetalContext();
        FusedAddRmsNorm.Execute(ctx, residualMetal, x, weight, output, n, seqLen, eps);

        for (int i = 0; i < n; i++) Assert.Equal(expOut[i], output[i]);
    }

    // ── Residual is updated in-place ─────────────────────────────────────────

    [MetalTestFact]
    public void SingleToken_ResidualUpdatedInPlace()
    {
        const int n = 8, seqLen = 1;
        const float eps = 1e-5f;
        var rng = new Random(2);

        Half[] residual = Rand(rng, n);
        Half[] x        = Rand(rng, n);
        Half[] weight   = RandPositive(rng, n);
        Half[] output   = new Half[n];

        var (expRes, _) = CpuExpected(residual, x, weight, n, seqLen, eps);
        Half[] residualMetal = (Half[])residual.Clone();

        using var ctx = new MetalContext();
        FusedAddRmsNorm.Execute(ctx, residualMetal, x, weight, output, n, seqLen, eps);

        // residual must contain FP16(residual[i] + x[i]) after the call
        for (int i = 0; i < n; i++) Assert.Equal(expRes[i], residualMetal[i]);
    }

    // ── Multiple tokens ───────────────────────────────────────────────────────

    [MetalTestFact]
    public void MultipleTokens_MatchesCpu()
    {
        const int n = 16, seqLen = 4;
        const float eps = 1e-5f;
        var rng = new Random(3);

        Half[] residual = Rand(rng, seqLen * n);
        Half[] x        = Rand(rng, seqLen * n);
        Half[] weight   = RandPositive(rng, n);
        Half[] output   = new Half[seqLen * n];

        var (expRes, expOut) = CpuExpected(residual, x, weight, n, seqLen, eps);
        Half[] residualMetal = (Half[])residual.Clone();

        using var ctx = new MetalContext();
        FusedAddRmsNorm.Execute(ctx, residualMetal, x, weight, output, n, seqLen, eps);

        for (int i = 0; i < output.Length;   i++) Assert.Equal(expOut[i], output[i]);
        for (int i = 0; i < residual.Length; i++) Assert.Equal(expRes[i], residualMetal[i]);
    }

    // ── Zero x: residual unchanged, output = RMSNorm(residual) ───────────────

    [MetalTestFact]
    public void ZeroX_ResidualUnchanged_OutputIsRmsNormOfResidual()
    {
        const int n = 8, seqLen = 1;
        const float eps = 1e-5f;
        var rng = new Random(4);

        Half[] residual = Rand(rng, n);
        Half[] x        = new Half[n];  // all zero
        Half[] weight   = RandPositive(rng, n);
        Half[] output   = new Half[n];

        var (expRes, expOut) = CpuExpected(residual, x, weight, n, seqLen, eps);
        Half[] residualOrig  = (Half[])residual.Clone();
        Half[] residualMetal = (Half[])residual.Clone();

        using var ctx = new MetalContext();
        FusedAddRmsNorm.Execute(ctx, residualMetal, x, weight, output, n, seqLen, eps);

        // residual + 0 = residual (exact in FP16)
        for (int i = 0; i < n; i++) Assert.Equal(residualOrig[i], residualMetal[i]);
        for (int i = 0; i < n; i++) Assert.Equal(expOut[i], output[i]);
    }

    // ── Epsilon prevents NaN on near-zero input ───────────────────────────────

    [MetalTestFact]
    public void NearZeroInput_EpsilonPreventsNaN()
    {
        const int n = 8, seqLen = 1;
        const float eps = 1e-5f;

        // residual = -x → sum = 0 → would be NaN without eps
        Half[] residual = Enumerable.Range(0, n).Select(i => (Half)(i * 0.001f)).ToArray();
        Half[] x        = residual.Select(h => (Half)(-(float)h)).ToArray();
        Half[] weight   = Enumerable.Repeat((Half)1f, n).ToArray();
        Half[] output   = new Half[n];

        using var ctx = new MetalContext();
        FusedAddRmsNorm.Execute(ctx, residual, x, weight, output, n, seqLen, eps);

        for (int i = 0; i < n; i++)
        {
            Assert.False(float.IsNaN((float)output[i]),      $"output[{i}] is NaN");
            Assert.False(float.IsInfinity((float)output[i]), $"output[{i}] is Infinity");
        }
    }

    // ── Large input (Llama-like hidden dim) ───────────────────────────────────

    [MetalTestFact]
    public void LargeInput_MatchesCpu()
    {
        const int n = 128, seqLen = 8;
        const float eps = 1e-5f;
        var rng = new Random(42);

        Half[] residual = Rand(rng, seqLen * n);
        Half[] x        = Rand(rng, seqLen * n);
        Half[] weight   = RandPositive(rng, n);
        Half[] output   = new Half[seqLen * n];

        var (expRes, expOut) = CpuExpected(residual, x, weight, n, seqLen, eps);
        Half[] residualMetal = (Half[])residual.Clone();

        using var ctx = new MetalContext();
        FusedAddRmsNorm.Execute(ctx, residualMetal, x, weight, output, n, seqLen, eps);

        for (int i = 0; i < output.Length;   i++) Assert.Equal(expOut[i], output[i]);
        for (int i = 0; i < residual.Length; i++) Assert.Equal(expRes[i], residualMetal[i]);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static Half[] Rand(Random rng, int n)
        => Enumerable.Range(0, n).Select(_ => (Half)(rng.NextSingle() * 2f - 1f)).ToArray();

    private static Half[] RandPositive(Random rng, int n)
        => Enumerable.Range(0, n).Select(_ => (Half)(rng.NextSingle() + 0.5f)).ToArray();
}
