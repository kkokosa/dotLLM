using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Correctness anchor for <see cref="Mamba3DataRoPE"/>. The data-dependent
/// cumulative-angle recurrence is the place where a sign flip on the cumsum or
/// a wrong pair ordering would produce plausible-but-silently-wrong outputs, so
/// these tests pin down both conventions against hand-computed values and
/// exercise rotation identities.
/// </summary>
public sealed class Mamba3DataRoPETests
{
    /// <summary>
    /// θ = 0 ⇒ raw_angles = 0 ⇒ cum_angles = 0 ⇒ rotation is identity. B and C
    /// must come out unchanged.
    /// </summary>
    [Fact]
    public void ZeroTheta_IsIdentity()
    {
        const int seqLen = 3;
        const int nHead = 2;
        const int dState = 4;
        int halfDState = dState / 2;

        float[] b = new float[seqLen * nHead * dState];
        float[] c = new float[seqLen * nHead * dState];
        // Fill B and C with distinct non-trivial values.
        for (int i = 0; i < b.Length; i++) { b[i] = 0.1f * (i + 1); c[i] = -0.2f * (i + 1); }
        float[] bOrig = (float[])b.Clone();
        float[] cOrig = (float[])c.Clone();

        // dt is non-zero; theta is zero — product is zero regardless.
        float[] dt = [0.5f, 0.5f, 0.25f, 0.25f, 0.1f, 0.1f]; // [T=3, nHead=2]
        float[] theta = new float[seqLen * halfDState]; // all zeros

        Mamba3DataRoPE.Execute(b, c, dt, theta, seqLen, nHead, dState);

        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(bOrig[i], b[i], 1e-6f);
            Assert.Equal(cOrig[i], c[i], 1e-6f);
        }
    }

    /// <summary>
    /// Constant dt and theta over time: cum_angles at step t is exactly
    /// <c>-(t+1) * dt * theta</c>. Hand-compute at T=3 with nHead=1, dState=2 so
    /// there's a single pair to rotate, and verify against
    /// <c>R(φ_t) · (b[0], b[1])</c> at each step.
    /// </summary>
    [Fact]
    public void ConstantDt_LinearAngleGrowth_MatchesHandComputed()
    {
        const int seqLen = 3;
        const int nHead = 1;
        const int dState = 2;
        // halfDState == 1, implicit in the 1-entry theta array below.

        float[] b = new float[seqLen * nHead * dState];
        float[] c = new float[seqLen * nHead * dState];
        // Each token has its own fresh (b0, b1) to rotate, so track three independent pairs.
        b[0] = 1f; b[1] = 0f;   // t=0: (1, 0)
        b[2] = 2f; b[3] = 0f;   // t=1: (2, 0)
        b[4] = 0f; b[5] = 3f;   // t=2: (0, 3)
        // C gets the same pattern so we can assert both paths.
        Array.Copy(b, c, b.Length);

        const float dtVal = 0.5f;
        const float thVal = 0.25f;
        float[] dt = [dtVal, dtVal, dtVal];           // [T=3, nHead=1]
        float[] theta = [thVal, thVal, thVal];        // [T=3, halfDState=1]

        // Expected angles: cum_angle[t] = -(t+1) * dt * theta.
        float phi0 = -1f * dtVal * thVal;
        float phi1 = -2f * dtVal * thVal;
        float phi2 = -3f * dtVal * thVal;

        Mamba3DataRoPE.Execute(b, c, dt, theta, seqLen, nHead, dState);

        // t=0: (1, 0) rotated by phi0 -> (cos phi0, sin phi0)
        Assert.Equal(MathF.Cos(phi0), b[0], 1e-6f);
        Assert.Equal(MathF.Sin(phi0), b[1], 1e-6f);
        // t=1: (2, 0) rotated by phi1 -> (2 cos phi1, 2 sin phi1)
        Assert.Equal(2f * MathF.Cos(phi1), b[2], 1e-6f);
        Assert.Equal(2f * MathF.Sin(phi1), b[3], 1e-6f);
        // t=2: (0, 3) rotated by phi2 -> (-3 sin phi2, 3 cos phi2)
        Assert.Equal(-3f * MathF.Sin(phi2), b[4], 1e-6f);
        Assert.Equal(3f * MathF.Cos(phi2), b[5], 1e-6f);

        // C must match B since inputs and weights are identical and the rotation
        // is applied to both with the same angles.
        for (int i = 0; i < b.Length; i++)
            Assert.Equal(b[i], c[i], 1e-6f);
    }

    /// <summary>
    /// Applying rotation φ then a second rotation −φ (by flipping the sign of
    /// dt) must return the original tensors to within FP32 precision. This
    /// guards against any accidental asymmetry in the pair layout or cum-sum sign.
    /// </summary>
    [Fact]
    public void Rotation_IsReversibleBySignFlippedDt()
    {
        const int seqLen = 2;
        const int nHead = 2;
        const int dState = 4;
        int halfDState = dState / 2;

        float[] b = new float[seqLen * nHead * dState];
        float[] c = new float[seqLen * nHead * dState];
        var rng = new Random(1729);
        for (int i = 0; i < b.Length; i++) { b[i] = (float)(rng.NextDouble() - 0.5); c[i] = (float)(rng.NextDouble() - 0.5); }
        float[] bOrig = (float[])b.Clone();
        float[] cOrig = (float[])c.Clone();

        float[] dt = new float[seqLen * nHead];
        for (int i = 0; i < dt.Length; i++) dt[i] = (float)rng.NextDouble();
        float[] theta = new float[seqLen * halfDState];
        for (int i = 0; i < theta.Length; i++) theta[i] = 0.3f + (float)rng.NextDouble();

        // Forward rotate.
        Mamba3DataRoPE.Execute(b, c, dt, theta, seqLen, nHead, dState);

        // Now apply the inverse: use -dt (which flips the sign of every raw_angle,
        // giving a cumsum that is the negative of the forward run, so each step's
        // accumulated angle is the negative of the forward step's). Because the
        // rotation is a pure SO(2) matrix, R(-φ) · R(φ) = I.
        float[] dtNeg = new float[dt.Length];
        for (int i = 0; i < dt.Length; i++) dtNeg[i] = -dt[i];
        Mamba3DataRoPE.Execute(b, c, dtNeg, theta, seqLen, nHead, dState);

        for (int i = 0; i < b.Length; i++)
        {
            Assert.Equal(bOrig[i], b[i], 1e-5f);
            Assert.Equal(cOrig[i], c[i], 1e-5f);
        }
    }

    /// <summary>
    /// The SIMD-ish public <see cref="Mamba3DataRoPE.Execute"/> and the pure
    /// scalar <see cref="Mamba3DataRoPE.ExecuteScalar"/> must agree on a random
    /// non-trivial input to within FP32 rounding differences.
    /// </summary>
    [Fact]
    public void Scalar_And_SimdPaths_Match()
    {
        const int seqLen = 5;
        const int nHead = 3;
        const int dState = 8;
        int halfDState = dState / 2;

        var rng = new Random(20260418);
        float[] b1 = new float[seqLen * nHead * dState];
        float[] c1 = new float[seqLen * nHead * dState];
        float[] dt = new float[seqLen * nHead];
        float[] theta = new float[seqLen * halfDState];

        for (int i = 0; i < b1.Length; i++) b1[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < c1.Length; i++) c1[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < dt.Length; i++) dt[i] = (float)(rng.NextDouble() * 0.5);
        for (int i = 0; i < theta.Length; i++) theta[i] = (float)(rng.NextDouble() * 2.0);

        float[] b2 = (float[])b1.Clone();
        float[] c2 = (float[])c1.Clone();

        Mamba3DataRoPE.Execute(b1, c1, dt, theta, seqLen, nHead, dState);
        Mamba3DataRoPE.ExecuteScalar(b2, c2, dt, theta, seqLen, nHead, dState);

        // Both implementations should produce bit-close outputs. 1e-5 absolute
        // tolerates fused-multiply-add vs pure scalar rounding differences.
        for (int i = 0; i < b1.Length; i++)
        {
            Assert.Equal(b2[i], b1[i], 1e-5f);
            Assert.Equal(c2[i], c1[i], 1e-5f);
        }
    }

    /// <summary>
    /// Even-dimension invariant: d_state must be even so adjacent pairs make sense.
    /// Passing odd <c>dState</c> must throw.
    /// </summary>
    /// <summary>
    /// B and C are each rotated in place, so a shared/overlapping buffer would be
    /// rotated twice. That must be rejected, not silently mis-rotated.
    /// </summary>
    [Fact]
    public void OverlappingBAndC_Throws()
    {
        const int seqLen = 2, nHead = 1, dState = 4;
        float[] buffer = new float[seqLen * nHead * dState];
        float[] dt = new float[seqLen * nHead];
        float[] theta = new float[seqLen * dState / 2];

        Assert.Throws<ArgumentException>(() =>
            Mamba3DataRoPE.Execute(buffer, buffer, dt, theta, seqLen, nHead, dState));

        // Partial overlap is rejected too (spans need not be identical).
        float[] wide = new float[seqLen * nHead * dState + 1];
        Assert.Throws<ArgumentException>(() =>
            Mamba3DataRoPE.Execute(
                wide.AsSpan(0, buffer.Length), wide.AsSpan(1, buffer.Length),
                dt, theta, seqLen, nHead, dState));
    }

    [Fact]
    public void OddDState_Throws()
    {
        float[] b = new float[3];
        float[] c = new float[3];
        float[] dt = [0f];
        float[] theta = new float[1];
        Assert.Throws<ArgumentException>(() =>
            Mamba3DataRoPE.Execute(b, c, dt, theta, seqLen: 1, nHead: 1, dState: 3));
    }
}
