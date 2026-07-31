using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Unit tests for <see cref="Mamba3QkNorm"/> — the thin RMSNorm wrapper used by
/// Mamba-3 to normalize B and C coefficient tensors of shape
/// <c>[T, n_group, d_state]</c> independently along each <c>[d_state]</c> slice.
/// </summary>
public sealed class Mamba3QkNormTests
{
    /// <summary>
    /// Groups are normalized independently: mutating one group's slice must not
    /// affect another group's values. Pinning this guards against a loop-index bug
    /// where two slices share a reduction accumulator.
    /// </summary>
    [Fact]
    public void GroupsIndependent_NormalizationDoesNotLeak()
    {
        const int T = 2;
        const int G = 3;
        const int D = 4;

        // Construct bc with very different RMS magnitudes per group so that a cross-
        // group leak would be obvious:
        //   group 0 small  (RMS ~ 1)
        //   group 1 medium (RMS ~ 10)
        //   group 2 large  (RMS ~ 100)
        float[] bc = new float[T * G * D];
        for (int t = 0; t < T; t++)
            for (int g = 0; g < G; g++)
            {
                float scale = g switch { 0 => 1f, 1 => 10f, _ => 100f };
                for (int d = 0; d < D; d++)
                    bc[(t * G + g) * D + d] = scale * (d + 1); // 1·s, 2·s, 3·s, 4·s
            }

        float[] weight = [1f, 1f, 1f, 1f];

        Mamba3QkNorm.Execute(bc, weight, 0f, T, G, D);

        // Each group's [1,2,3,4] scaled by its own scale normalizes to the same
        // unit-RMS vector: expected_d = (d+1) / sqrt((1+4+9+16)/4) = (d+1) / sqrt(7.5).
        float invRms = 1.0f / MathF.Sqrt((1f + 4f + 9f + 16f) / 4f);
        for (int t = 0; t < T; t++)
            for (int g = 0; g < G; g++)
                for (int d = 0; d < D; d++)
                {
                    float expected = (d + 1) * invRms;
                    float actual = bc[(t * G + g) * D + d];
                    Assert.Equal(expected, actual, 1e-4f);
                }
    }

    /// <summary>
    /// With weight = 1 and epsilon = 0, a unit-RMS input must come out equal to
    /// itself. This is the degenerate sanity check — confirms no spurious scaling.
    /// </summary>
    [Fact]
    public void UnitWeightUnitRms_Identity()
    {
        const int T = 1;
        const int G = 1;
        const int D = 4;

        // Input [1, 1, 1, 1] has sum-of-squares = 4, RMS = sqrt(4/4) = 1, so the
        // normalization is a no-op when weight = 1 and epsilon = 0.
        float[] bc = [1f, 1f, 1f, 1f];
        float[] weight = [1f, 1f, 1f, 1f];
        float[] input_copy = (float[])bc.Clone();

        Mamba3QkNorm.Execute(bc, weight, 0f, T, G, D);

        for (int i = 0; i < D; i++)
            Assert.Equal(input_copy[i], bc[i], 1e-6f);
    }

    /// <summary>
    /// Alias-safety: the API is explicitly in-place on <c>bc</c>. Verify that
    /// running in-place matches a reference computed via per-slice
    /// <see cref="RmsNorm.Execute"/> into a non-aliased output buffer.
    /// </summary>
    [Fact]
    public void InPlace_MatchesPerSliceRmsNormReference()
    {
        const int T = 3;
        const int G = 2;
        const int D = 8;

        var rng = new Random(7);
        float[] bc = new float[T * G * D];
        for (int i = 0; i < bc.Length; i++)
            bc[i] = rng.NextSingle() * 4f - 2f;
        float[] weight = new float[D];
        for (int i = 0; i < D; i++)
            weight[i] = rng.NextSingle() + 0.5f;
        const float eps = 1e-5f;

        // Reference: loop over slices and call RmsNorm.Execute into a *separate*
        // output buffer so we know the in-place aliasing path in the kernel isn't
        // fooling us.
        float[] bcClone = (float[])bc.Clone();
        float[] reference = new float[bc.Length];
        for (int i = 0; i < T * G; i++)
        {
            ReadOnlySpan<float> inSlice = bcClone.AsSpan(i * D, D);
            Span<float> outSlice = reference.AsSpan(i * D, D);
            RmsNorm.Execute(inSlice, weight, eps, outSlice);
        }

        // In-place kernel call.
        Mamba3QkNorm.Execute(bc, weight, eps, T, G, D);

        for (int i = 0; i < bc.Length; i++)
            Assert.Equal(reference[i], bc[i], 1e-5f);
    }
}
