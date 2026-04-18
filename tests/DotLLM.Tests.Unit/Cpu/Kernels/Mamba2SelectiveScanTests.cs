using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Correctness anchor for <see cref="Mamba2SelectiveScan"/>. The scan is the numerical
/// heart of Nemotron-H and sign/head-order bugs produce plausible-looking gibberish,
/// so these tests hand-compute tiny cases and compare element-wise.
/// </summary>
public sealed class Mamba2SelectiveScanTests
{
    /// <summary>
    /// Single-token scan, zero initial state, zero A (dA=1 identity),
    /// dt=0 (softplus(0)=ln2), B=C=1 for everything. Hand-computed.
    /// </summary>
    [Fact]
    public void SingleToken_ZeroStateIdentityDecay_MatchesHandComputed()
    {
        const int nHead = 2;
        const int headDim = 2;
        const int dState = 2;
        const int nGroup = 1;
        const int seqLen = 1;
        int dInner = nHead * headDim; // 4

        float[] state = new float[nHead * headDim * dState]; // 8 floats, all zero
        float[] x = [1f, 2f, 3f, 4f];
        float[] dt = [0f, 0f];                           // softplus(0) = ln(2)
        float[] a = [0f, 0f];                            // exp(dt_sp * 0) = 1
        float[] b = [1f, 1f];                            // [T=1, nGroup=1, dState=2]
        float[] c = [1f, 1f];                            // [T=1, nGroup=1, dState=2]
        float[] y = new float[seqLen * dInner];

        Mamba2SelectiveScan.Execute(
            state, x, dt, a, b, c, y,
            nHead, headDim, dState, nGroup, seqLen);

        float ln2 = MathF.Log(2f);

        // y[h, i] = headDim-dimensional dot where state becomes x[i] * ln2 per k
        // After the step state[h, i, k] = x[h*headDim+i] * ln2 for all k, and sum = dState * state value.
        Assert.Equal(dState * 1f * ln2, y[0], 1e-5f);          // h=0, i=0, x=1
        Assert.Equal(dState * 2f * ln2, y[1], 1e-5f);          // h=0, i=1, x=2
        Assert.Equal(dState * 3f * ln2, y[2], 1e-5f);          // h=1, i=0, x=3
        Assert.Equal(dState * 4f * ln2, y[3], 1e-5f);          // h=1, i=1, x=4

        // Sanity check state was updated.
        for (int idx = 0; idx < state.Length; idx++)
            Assert.NotEqual(0f, state[idx]);
    }

    /// <summary>
    /// Two-token scan verifies state carries forward and is decayed by dA.
    /// With dt_sp=1, A=-ln(2) -> dA = exp(-ln2) = 0.5. State halves each step.
    /// </summary>
    [Fact]
    public void TwoTokens_NegativeA_StateDecays()
    {
        const int nHead = 1;
        const int headDim = 1;
        const int dState = 1;
        const int nGroup = 1;
        const int seqLen = 2;

        // Want dt_sp = 1 exactly. softplus(x) = 1 -> exp(x) = e-1 -> x = log(e-1).
        float dtVal = MathF.Log(MathF.E - 1f);
        float negLn2 = -MathF.Log(2f); // A per head

        float[] state = [0f];
        float[] x = [1f, 2f];              // two tokens, one channel
        float[] dt = [dtVal, dtVal];
        float[] a = [negLn2];               // dA = exp(1 * -ln2) = 0.5
        float[] b = [1f, 1f];               // per-token B
        float[] c = [1f, 1f];               // per-token C
        float[] y = new float[2];

        Mamba2SelectiveScan.Execute(state, x, dt, a, b, c, y, nHead, headDim, dState, nGroup, seqLen);

        // t=0: state <- 0*0.5 + 1*(1*1) = 1;   y[0] = 1*1 = 1
        // t=1: state <- 1*0.5 + 1*(2*1) = 2.5; y[1] = 2.5*1 = 2.5
        Assert.Equal(1.0f, y[0], 1e-5f);
        Assert.Equal(2.5f, y[1], 1e-5f);
        Assert.Equal(2.5f, state[0], 1e-5f);
    }

    /// <summary>
    /// Verifies that groups correctly share B/C across heads in the same group
    /// and do NOT cross-contaminate between groups.
    /// n_head=4, n_group=2 -> heads 0,1 share group 0, heads 2,3 share group 1.
    /// </summary>
    [Fact]
    public void Groups_ShareBCWithinButNotAcross()
    {
        const int nHead = 4;
        const int headDim = 1;
        const int dState = 1;
        const int nGroup = 2;
        const int seqLen = 1;

        float[] state = new float[nHead * headDim * dState]; // [4]
        float[] x = [1f, 1f, 1f, 1f];
        float[] dt = new float[nHead];                        // softplus(0) = ln(2)
        float[] a = new float[nHead];                         // dA = 1
        // B split by group: group 0 -> 10, group 1 -> 100
        float[] b = [10f, 100f]; // [T=1, nGroup=2, dState=1]
        float[] c = [1f, 1f];
        float[] y = new float[4];

        Mamba2SelectiveScan.Execute(state, x, dt, a, b, c, y, nHead, headDim, dState, nGroup, seqLen);

        float ln2 = MathF.Log(2f);
        // state[h,0,0] = x[h]*ln2 * B[g] (with B shared per group and initial state 0)
        // y = state * c = state * 1
        Assert.Equal(10f * ln2, y[0], 1e-5f); // h=0, g=0
        Assert.Equal(10f * ln2, y[1], 1e-5f); // h=1, g=0
        Assert.Equal(100f * ln2, y[2], 1e-5f); // h=2, g=1
        Assert.Equal(100f * ln2, y[3], 1e-5f); // h=3, g=1
    }

    /// <summary>
    /// A=0 means dA=1, so state accumulates without decay. Feeding B=0 after step 1
    /// should freeze the state at step 1's value; y then just projects with C.
    /// </summary>
    [Fact]
    public void ZeroA_AccumulatesUnbounded()
    {
        const int nHead = 1;
        const int headDim = 1;
        const int dState = 1;
        const int nGroup = 1;
        const int seqLen = 3;

        float[] state = [0f];
        float[] x = [1f, 1f, 1f];
        float[] dt = [0f, 0f, 0f];       // softplus(0) = ln(2)
        float[] a = [0f];                // dA = 1
        float[] b = [1f, 1f, 1f];
        float[] c = [1f, 1f, 1f];
        float[] y = new float[3];

        Mamba2SelectiveScan.Execute(state, x, dt, a, b, c, y, nHead, headDim, dState, nGroup, seqLen);

        // state evolves: 0 -> ln2 -> 2*ln2 -> 3*ln2
        float ln2 = MathF.Log(2f);
        Assert.Equal(1f * ln2, y[0], 1e-5f);
        Assert.Equal(2f * ln2, y[1], 1e-5f);
        Assert.Equal(3f * ln2, y[2], 1e-5f);
        Assert.Equal(3f * ln2, state[0], 1e-5f);
    }
}
