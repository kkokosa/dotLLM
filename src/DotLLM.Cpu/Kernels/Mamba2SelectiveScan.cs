using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Mamba2 selective state-space scan for NVIDIA Nemotron-H SSM layers.
/// Scalar reference port of <c>ggml_compute_forward_ssm_scan_f32</c> (llama.cpp,
/// Mamba-2 branch where <c>A</c> is scalar per head — one element per head, not a
/// full <c>[head_dim, d_state]</c> matrix as in Mamba-1).
/// </summary>
/// <remarks>
/// <para>
/// The selective scan is the recurrent heart of Mamba2. For each time step it
/// advances a per-head state matrix and projects it to an output through the
/// token-dependent coefficients B, C and dt:
/// </para>
/// <code>
/// dt_sp = softplus(dt[t, h])
/// dA    = exp(dt_sp * A[h])          // A stored negative by converter; exp(dt*A) decays
/// g     = h / (n_head / n_group)     // B, C are shared across heads within a group
/// for i in 0..head_dim:
///     x_dt = x[t, h*head_dim + i] * dt_sp
///     sum  = 0
///     for k in 0..d_state:
///         state[h, i, k] = state[h, i, k] * dA + B[t, g*d_state + k] * x_dt
///         sum += state[h, i, k] * C[t, g*d_state + k]
///     y[t, h*head_dim + i] = sum
/// </code>
/// <para>
/// After the scan the caller must add the D skip term and apply the z-gate
/// (<c>NemotronHTransformerModel</c>'s SSM branch handles this).
/// </para>
/// <para>
/// This kernel writes updated state back in-place and produces y as a contiguous
/// <c>[T, d_inner]</c> row-major tensor.
/// </para>
/// </remarks>
public static class Mamba2SelectiveScan
{
    /// <summary>
    /// Runs the selective scan over <paramref name="seqLen"/> tokens.
    /// </summary>
    /// <param name="state">
    /// SSM hidden state, shape <c>[n_head, head_dim, d_state]</c> row-major,
    /// length <c>n_head * head_dim * d_state</c>. Updated in place.
    /// </param>
    /// <param name="x">Post-conv activations, shape <c>[T, d_inner]</c> row-major.</param>
    /// <param name="dt">Time-step parameter, shape <c>[T, n_head]</c> row-major. Already bias-added.</param>
    /// <param name="a">A parameter, length <c>n_head</c> (scalar per head, Mamba2).</param>
    /// <param name="b">B coefficient, shape <c>[T, n_group, d_state]</c> row-major.</param>
    /// <param name="c">C coefficient, shape <c>[T, n_group, d_state]</c> row-major.</param>
    /// <param name="y">Output, shape <c>[T, d_inner]</c> row-major. Written.</param>
    /// <param name="nHead">Number of heads.</param>
    /// <param name="headDim">Channels per head (<c>d_inner / n_head</c>).</param>
    /// <param name="dState">SSM state width (typically 128).</param>
    /// <param name="nGroup">Number of B/C groups.</param>
    /// <param name="seqLen">Number of tokens in this step.</param>
    [SkipLocalsInit]
    public static void Execute(
        Span<float> state,
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        ReadOnlySpan<float> c,
        Span<float> y,
        int nHead,
        int headDim,
        int dState,
        int nGroup,
        int seqLen)
    {
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (nGroup <= 0) throw new ArgumentOutOfRangeException(nameof(nGroup));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nHead % nGroup != 0)
            throw new ArgumentException($"n_head ({nHead}) must be divisible by n_group ({nGroup}).");

        int dInner = nHead * headDim;
        int headsPerGroup = nHead / nGroup;

        if (state.Length < (long)nHead * headDim * dState)
            throw new ArgumentException("state span too small.", nameof(state));
        if (x.Length < (long)seqLen * dInner)
            throw new ArgumentException("x span too small.", nameof(x));
        if (dt.Length < (long)seqLen * nHead)
            throw new ArgumentException("dt span too small.", nameof(dt));
        if (a.Length < nHead)
            throw new ArgumentException("a span too small.", nameof(a));
        if (b.Length < (long)seqLen * nGroup * dState)
            throw new ArgumentException("b span too small.", nameof(b));
        if (c.Length < (long)seqLen * nGroup * dState)
            throw new ArgumentException("c span too small.", nameof(c));
        if (y.Length < (long)seqLen * dInner)
            throw new ArgumentException("y span too small.", nameof(y));

        // Per-head state stride into the [n_head, head_dim, d_state] buffer.
        int stateStrideHead = headDim * dState;

        for (int t = 0; t < seqLen; t++)
        {
            ReadOnlySpan<float> xRow = x.Slice(t * dInner, dInner);
            ReadOnlySpan<float> dtRow = dt.Slice(t * nHead, nHead);
            ReadOnlySpan<float> bRow = b.Slice(t * nGroup * dState, nGroup * dState);
            ReadOnlySpan<float> cRow = c.Slice(t * nGroup * dState, nGroup * dState);
            Span<float> yRow = y.Slice(t * dInner, dInner);

            for (int h = 0; h < nHead; h++)
            {
                float dtSp = SoftPlus(dtRow[h]);
                // A is stored negative by the GGUF converter; exp(dtSp * A[h]) is in (0,1) -> stable decay.
                float dA = MathF.Exp(dtSp * a[h]);

                int g = h / headsPerGroup;
                ReadOnlySpan<float> bGroup = bRow.Slice(g * dState, dState);
                ReadOnlySpan<float> cGroup = cRow.Slice(g * dState, dState);

                int xHeadOffset = h * headDim;
                int stateHeadOffset = h * stateStrideHead;

                for (int i = 0; i < headDim; i++)
                {
                    float xDt = xRow[xHeadOffset + i] * dtSp;
                    int stateRowOffset = stateHeadOffset + i * dState;
                    Span<float> stateRow = state.Slice(stateRowOffset, dState);

                    float sumf = 0f;
                    for (int k = 0; k < dState; k++)
                    {
                        // Selective SSM recurrence.
                        float s = stateRow[k] * dA + bGroup[k] * xDt;
                        stateRow[k] = s;
                        sumf += s * cGroup[k];
                    }
                    yRow[xHeadOffset + i] = sumf;
                }
            }
        }
    }

    /// <summary>
    /// Numerically stable <c>softplus(x) = log(1 + exp(x))</c>. Matches the
    /// llama.cpp SSM scan implementation which uses the same fast path.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SoftPlus(float x)
    {
        // For large positive x, exp(x) overflows and the answer is ~x. For very
        // negative x, log1p(exp(x)) underflows to 0. Mirror llama.cpp's approach.
        if (x > 20f) return x;
        if (x < -20f) return MathF.Exp(x); // log(1 + eps) ~= eps for tiny eps
        return MathF.Log(1f + MathF.Exp(x));
    }
}
