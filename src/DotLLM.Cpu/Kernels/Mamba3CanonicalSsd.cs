using System.Buffers;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Canonical Mamba-3 attention-style SSD scan. Mirrors the inner loop of the
/// <c>state-spaces/mamba</c> Triton kernels <c>mamba3_siso_combined</c> (SISO)
/// and <c>mamba3_mimo_combined</c> (MIMO) from commit <c>7438488</c>.
/// </summary>
/// <remarks>
/// <para>
/// Structurally this is <b>not</b> the trapezoidal-rule α/β/γ recurrence used by
/// the (now-retired) minimal-reference Mamba3SelectiveScan kernel. The canonical
/// kernel instead runs an attention-like decay-state update fused with a
/// pre-rotation QK-dot skip:
/// </para>
/// <code>
///   γ_t        = DT_t · trap_t                                          // trapezoidal weight
///   shift_γ_t  = DT_{t+1} · (1 - trap_{t+1})    // 0 at t = L-1          // next-step weight
///   scale_t    = γ_t + shift_γ_t                                        // applied to K only
///
///   Q_t        = rope(C_t + C_bias, cum_angles_t)        // already rotated by caller
///   K_t        = rope(B_t + B_bias, cum_angles_t)        // already rotated by caller
///   K_scaled_t = K_t · scale_t                                          // per-head scalar
///
///   h_t[p, n]  = exp(ADT_t) · h_{t-1}[p, n]  +  V_t[p] · K_scaled_t[n]   // SSM state update
///   y_scan_t[p] = Σ_n Q_t[n] · h_t[p, n]                                // state readout
///   skip_t     = D_h + γ_t · (C_t + C_bias) · (B_t + B_bias)            // pre-RoPE dot
///   y_t[p]     = y_scan_t[p] + skip_t · V_t[p]
///   y_t       *= silu(Z_t)                                              // only if hasZ
/// </code>
/// <para>
/// This is a one-to-one port of the pure-Python reference in
/// <c>tests/.../Fixtures/Mamba3/capture_fixtures_canonical.py</c>
/// (<c>canonical_siso_scan</c> / <c>canonical_mimo_scan</c>), which was
/// validated algebraically against the Triton kernel's Phase-1 / Phase-2
/// inner loops (see <c>mamba3_siso_fwd.py:256-425</c>).
/// </para>
/// <para>
/// MIMO adds a rank axis R to B, C (pre-RoPE), expands V through <c>mimo_x</c>,
/// gates through <c>mimo_z</c> per-rank, and contracts through <c>mimo_o</c>
/// after the per-rank y is computed. The hidden state <c>h</c> stays
/// rank-contracted — canonical handles this by summing <c>K_scaled</c> over
/// the rank axis inside the state update; V enters unexpanded (the rank
/// expansion of V is folded into the scan via the K.sum trick, matching the
/// Triton kernel's internal contract).
/// </para>
/// <para>
/// <b>Decode continuity.</b> <c>state</c> is threaded in / out to let the
/// block resume mid-sequence. The canonical scan has no β-term so it does
/// not maintain a <c>prev_Bx</c> buffer like
/// the trapezoidal recurrence; the second recurrent slot is the
/// <c>shifted_gamma</c> boundary, which the block supplies already folded
/// into <c>scale</c>. For single-shot prefill this lookahead comes from the
/// next token within the same call; for streaming decode the caller must
/// pass a <c>scale</c> series that encodes the right boundary condition
/// (typically <c>scale = γ</c> at the final token of the current chunk and
/// <c>scale = shifted_γ + γ</c> stitched across the chunk boundary by the
/// caller).
/// </para>
/// <para>
/// <b>Alias safety.</b> <c>b</c> and <c>c</c> are read-only (post-RoPE).
/// <c>y</c> must not overlap any input. <c>state</c> is read/written per
/// token; caller owns its storage. Scalar reference only — SIMD deferred.
/// </para>
/// </remarks>
public static class Mamba3CanonicalSsd
{
    /// <summary>Stackalloc threshold in floats for the per-head rank-sum scratch.</summary>
    private const int StackAllocFloatThreshold = 2048;

    /// <summary>
    /// State update for one <c>(t, h)</c> pair:
    /// <c>h[p, n] = decay · h[p, n] + V[p] · (Σ_r K_r[n]) · scale</c>.
    /// </summary>
    /// <remarks>
    /// The rank sum depends only on <c>n</c>, so it is materialised once per head
    /// rather than recomputed for every channel <c>p</c> — dropping an
    /// <c>O(headDim)</c> factor from the rank reduction. The summation order over
    /// <c>r</c> is unchanged, so results are bit-identical to the naive form.
    /// Also used by the streaming chunk-boundary adjustment with
    /// <paramref name="decay"/> = 1 and <paramref name="scale"/> = the boundary
    /// coefficient, which has exactly the same shape.
    /// </remarks>
    [SkipLocalsInit]
    private static void UpdateStateRankSummed(
        Span<float> state,
        ReadOnlySpan<float> k,
        ReadOnlySpan<float> vSlice,
        int stateBase,
        int kHeadBase,
        int kRankStride,
        float decay,
        float scale,
        int nRank,
        int headDim,
        int dState)
    {
        float[]? rented = null;
        Span<float> kSum = dState <= StackAllocFloatThreshold
            ? stackalloc float[dState]
            : (rented = ArrayPool<float>.Shared.Rent(dState)).AsSpan(0, dState);

        try
        {
            for (int n = 0; n < dState; n++)
            {
                float s = 0f;
                for (int r = 0; r < nRank; r++)
                    s += k[kHeadBase + r * kRankStride + n];
                kSum[n] = s * scale;
            }

            for (int p = 0; p < headDim; p++)
            {
                float vp = vSlice[p];
                int row = stateBase + p * dState;
                for (int n = 0; n < dState; n++)
                    state[row + n] = decay * state[row + n] + vp * kSum[n];
            }
        }
        finally
        {
            if (rented is not null)
                ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Runs the canonical SISO SSD scan over <paramref name="seqLen"/> tokens.
    /// </summary>
    /// <param name="state">
    /// SSM hidden state, shape <c>[nHead, headDim, dState]</c> row-major.
    /// Updated in place. Zero on first call of a fresh sequence.
    /// </param>
    /// <param name="v">
    /// Value / <c>x</c> per head, shape <c>[T, nHead, headDim]</c> row-major.
    /// </param>
    /// <param name="qRoped">
    /// Rotated biased C, shape <c>[T, nHead, dState]</c> row-major.
    /// </param>
    /// <param name="kRoped">
    /// Rotated biased B, shape <c>[T, nHead, dState]</c> row-major.
    /// </param>
    /// <param name="qkPreDot">
    /// Pre-RoPE dot product <c>Σ_n (C+C_bias)_n · (B+B_bias)_n</c>,
    /// shape <c>[T, nHead]</c>.
    /// </param>
    /// <param name="scale">
    /// <c>γ_t + shifted_γ_t</c>, shape <c>[T, nHead]</c>.
    /// </param>
    /// <param name="gamma">
    /// <c>DT_t · trap_t</c>, shape <c>[T, nHead]</c>.
    /// </param>
    /// <param name="adt">
    /// <c>_A_t · DT_t</c> (already negative-clamped), shape <c>[T, nHead]</c>.
    /// </param>
    /// <param name="d">Per-head skip coefficient, shape <c>[nHead]</c>.</param>
    /// <param name="z">
    /// Gate <c>Z</c> per head, shape <c>[T, nHead, headDim]</c>, or empty span
    /// to skip the silu(z) gate.
    /// </param>
    /// <param name="y">
    /// Output, shape <c>[T, nHead, headDim]</c> row-major. Written.
    /// </param>
    /// <param name="seqLen">Number of tokens T.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head.</param>
    /// <param name="dState">State width.</param>
    [SkipLocalsInit]
    public static void ExecuteSiso(
        Span<float> state,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> qRoped,
        ReadOnlySpan<float> kRoped,
        ReadOnlySpan<float> qkPreDot,
        ReadOnlySpan<float> scale,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> adt,
        ReadOnlySpan<float> d,
        ReadOnlySpan<float> z,
        Span<float> y,
        int seqLen,
        int nHead,
        int headDim,
        int dState)
    {
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));

        long stateElems = (long)nHead * headDim * dState;
        long vElems = (long)seqLen * nHead * headDim;
        long bcElems = (long)seqLen * nHead * dState;
        long hdrElems = (long)seqLen * nHead;

        if (state.Length < stateElems) throw new ArgumentException("state span too small.", nameof(state));
        if (v.Length < vElems) throw new ArgumentException("v span too small.", nameof(v));
        if (qRoped.Length < bcElems) throw new ArgumentException("qRoped span too small.", nameof(qRoped));
        if (kRoped.Length < bcElems) throw new ArgumentException("kRoped span too small.", nameof(kRoped));
        if (qkPreDot.Length < hdrElems) throw new ArgumentException("qkPreDot span too small.", nameof(qkPreDot));
        if (scale.Length < hdrElems) throw new ArgumentException("scale span too small.", nameof(scale));
        if (gamma.Length < hdrElems) throw new ArgumentException("gamma span too small.", nameof(gamma));
        if (adt.Length < hdrElems) throw new ArgumentException("adt span too small.", nameof(adt));
        if (d.Length < nHead) throw new ArgumentException("d span too small.", nameof(d));
        if (!z.IsEmpty && z.Length < vElems) throw new ArgumentException("z span too small.", nameof(z));
        if (y.Length < vElems) throw new ArgumentException("y span too small.", nameof(y));

        if (seqLen == 0) return;

        bool hasZ = !z.IsEmpty;
        int bcHeadStride = dState;
        int vHeadStride = headDim;

        for (int t = 0; t < seqLen; t++)
        {
            int vTokBase = t * nHead * headDim;
            int bcTokBase = t * nHead * dState;
            int hdrTokBase = t * nHead;

            for (int h = 0; h < nHead; h++)
            {
                float decay = MathF.Exp(adt[hdrTokBase + h]);
                float scl = scale[hdrTokBase + h];
                float gm = gamma[hdrTokBase + h];
                float qkp = qkPreDot[hdrTokBase + h];
                float skip = d[h] + gm * qkp;

                int vBase = vTokBase + h * vHeadStride;
                int bcBase = bcTokBase + h * bcHeadStride;
                int stateBase = h * headDim * dState;
                ReadOnlySpan<float> q = qRoped.Slice(bcBase, dState);
                ReadOnlySpan<float> k = kRoped.Slice(bcBase, dState);
                ReadOnlySpan<float> vSlice = v.Slice(vBase, headDim);

                // Per-head h update + readout. State layout: h[p, n].
                for (int p = 0; p < headDim; p++)
                {
                    float vp = vSlice[p];
                    int stateRowBase = stateBase + p * dState;
                    float yScan = 0f;
                    for (int n = 0; n < dState; n++)
                    {
                        // h_new[p,n] = decay * h_old[p,n] + V[p] * K[n] * scale
                        float newState = decay * state[stateRowBase + n] + vp * (k[n] * scl);
                        state[stateRowBase + n] = newState;
                        yScan += q[n] * newState;
                    }

                    float yOut = yScan + skip * vp;
                    if (hasZ)
                    {
                        float zv = z[vBase + p];
                        // silu(z) = z * sigmoid(z) = z / (1 + exp(-z)).
                        float silu = zv / (1f + MathF.Exp(-zv));
                        yOut *= silu;
                    }
                    y[vBase + p] = yOut;
                }
            }
        }
    }

    /// <summary>
    /// Runs the canonical MIMO SSD scan over <paramref name="seqLen"/> tokens.
    /// Rank-expands <paramref name="z"/> through <paramref name="mimoZ"/> per
    /// rank, computes a per-rank y, then contracts the rank axis away through
    /// <paramref name="mimoO"/>.
    /// </summary>
    /// <remarks>
    /// The <paramref name="v"/> input is <em>not</em> pre-expanded — canonical
    /// folds the V rank expansion into the state update by summing the
    /// per-rank K over the rank axis and using the unexpanded V. This matches
    /// the Triton kernel contract exactly (h is shape <c>[nHead, headDim, dState]</c>,
    /// no rank dim).
    /// </remarks>
    /// <param name="state">SSM hidden state, <c>[nHead, headDim, dState]</c>. In-place.</param>
    /// <param name="v">V per head, <c>[T, nHead, headDim]</c>.</param>
    /// <param name="qRoped">Rotated biased C per rank, <c>[T, nRank, nHead, dState]</c>.</param>
    /// <param name="kRoped">Rotated biased B per rank, <c>[T, nRank, nHead, dState]</c>.</param>
    /// <param name="qkPreDotSum">Sum over rank of the pre-rotation QK dot, <c>[T, nHead]</c>.</param>
    /// <param name="scale">Per-token per-head scale, <c>[T, nHead]</c>.</param>
    /// <param name="gamma">DT·trap per-token per-head, <c>[T, nHead]</c>.</param>
    /// <param name="adt">_A·DT per-token per-head, <c>[T, nHead]</c>.</param>
    /// <param name="d">Per-head skip coefficient, <c>[nHead]</c>.</param>
    /// <param name="z">Gate, <c>[T, nHead, headDim]</c> or empty.</param>
    /// <param name="mimoZ">Gate rank expansion, <c>[nHead, nRank, headDim]</c>.</param>
    /// <param name="mimoO">Output rank contraction, <c>[nHead, nRank, headDim]</c>.</param>
    /// <param name="y">Output, <c>[T, nHead, headDim]</c>. Written.</param>
    /// <param name="yPerRank">
    /// Optional per-rank output buffer, <c>[T, nRank, nHead, headDim]</c>.
    /// Pass empty span to skip. Useful for fixture-level comparators (canonical
    /// exposes <c>y_pre_contract</c>).
    /// </param>
    /// <param name="seqLen">Token count T.</param>
    /// <param name="nRank">MIMO rank R.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="headDim">Channels per head.</param>
    /// <param name="dState">State width.</param>
    [SkipLocalsInit]
    public static void ExecuteMimo(
        Span<float> state,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> qRoped,
        ReadOnlySpan<float> kRoped,
        ReadOnlySpan<float> qkPreDotSum,
        ReadOnlySpan<float> scale,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> adt,
        ReadOnlySpan<float> d,
        ReadOnlySpan<float> z,
        ReadOnlySpan<float> mimoZ,
        ReadOnlySpan<float> mimoO,
        Span<float> y,
        Span<float> yPerRank,
        int seqLen,
        int nRank,
        int nHead,
        int headDim,
        int dState)
    {
        if (nRank <= 0) throw new ArgumentOutOfRangeException(nameof(nRank));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));

        long stateElems = (long)nHead * headDim * dState;
        long vElems = (long)seqLen * nHead * headDim;
        long bcElems = (long)seqLen * nRank * nHead * dState;
        long hdrElems = (long)seqLen * nHead;
        long mimoElems = (long)nHead * nRank * headDim;
        long yPerRankElems = (long)seqLen * nRank * nHead * headDim;

        if (state.Length < stateElems) throw new ArgumentException("state too small.", nameof(state));
        if (v.Length < vElems) throw new ArgumentException("v too small.", nameof(v));
        if (qRoped.Length < bcElems) throw new ArgumentException("qRoped too small.", nameof(qRoped));
        if (kRoped.Length < bcElems) throw new ArgumentException("kRoped too small.", nameof(kRoped));
        if (qkPreDotSum.Length < hdrElems) throw new ArgumentException("qkPreDotSum too small.", nameof(qkPreDotSum));
        if (scale.Length < hdrElems) throw new ArgumentException("scale too small.", nameof(scale));
        if (gamma.Length < hdrElems) throw new ArgumentException("gamma too small.", nameof(gamma));
        if (adt.Length < hdrElems) throw new ArgumentException("adt too small.", nameof(adt));
        if (d.Length < nHead) throw new ArgumentException("d too small.", nameof(d));
        if (!z.IsEmpty && z.Length < vElems) throw new ArgumentException("z too small.", nameof(z));
        if (mimoZ.Length < mimoElems) throw new ArgumentException("mimoZ too small.", nameof(mimoZ));
        if (mimoO.Length < mimoElems) throw new ArgumentException("mimoO too small.", nameof(mimoO));
        if (y.Length < vElems) throw new ArgumentException("y too small.", nameof(y));
        if (!yPerRank.IsEmpty && yPerRank.Length < yPerRankElems)
            throw new ArgumentException("yPerRank too small.", nameof(yPerRank));

        if (seqLen == 0) return;

        bool hasZ = !z.IsEmpty;
        bool writePerRank = !yPerRank.IsEmpty;
        int bcHeadStride = dState;
        int bcRankStride = nHead * dState;
        int bcTokStride = nRank * bcRankStride;

        // Strides into mimoZ / mimoO (shape [H, R, P] row-major).
        int mimoHeadStride = nRank * headDim;
        int mimoRankStride = headDim;

        // Inv-rank for evenly distributing the skip across ranks (canonical kernel
        // folds D / qk_dot into the per-rank output before mimo_o contraction;
        // dividing by R matches the reference canonical_mimo_scan).
        float invRank = 1f / nRank;

        for (int t = 0; t < seqLen; t++)
        {
            int vTokBase = t * nHead * headDim;
            int bcTokBase = t * bcTokStride;
            int hdrTokBase = t * nHead;
            int perRankTokBase = t * nRank * nHead * headDim;

            for (int h = 0; h < nHead; h++)
            {
                float decay = MathF.Exp(adt[hdrTokBase + h]);
                float scl = scale[hdrTokBase + h];
                float gm = gamma[hdrTokBase + h];
                float qkp = qkPreDotSum[hdrTokBase + h];
                float skip = d[h] + gm * qkp;

                int vBase = vTokBase + h * headDim;
                int stateBase = h * headDim * dState;
                ReadOnlySpan<float> vSlice = v.Slice(vBase, headDim);

                // h update: h_new[p,n] = decay * h_old[p,n] + V[p] * (Σ_r K_r[n]) * scale
                // Σ_r K_r[n] depends only on (t, h, n), so it is rank-summed once
                // per head instead of once per (p, n) — the summation order is
                // unchanged, so results stay bit-identical.
                UpdateStateRankSummed(
                    state, kRoped, vSlice,
                    stateBase, bcTokBase + h * bcHeadStride, bcRankStride,
                    decay, scl, nRank, headDim, dState);

                // Per-rank readout and contraction.
                for (int p = 0; p < headDim; p++)
                {
                    float vp = vSlice[p];
                    int stateRowBase = stateBase + p * dState;

                    float contracted = 0f;
                    for (int r = 0; r < nRank; r++)
                    {
                        int qBase = bcTokBase + r * bcRankStride + h * bcHeadStride;
                        // y_scan_r = Σ_n Q_r[n] * h[p, n]
                        float yScanR = 0f;
                        for (int n = 0; n < dState; n++)
                        {
                            yScanR += qRoped[qBase + n] * state[stateRowBase + n];
                        }
                        // per-rank y = y_scan_r + (skip / R) * V[p]
                        float yR = yScanR + skip * invRank * vp;

                        // Gate: silu(z_t * mimo_z[h, r, p]) — per-rank z via mimo_z rank expand.
                        if (hasZ)
                        {
                            int zIdx = vBase + p;
                            int mimoZIdx = h * mimoHeadStride + r * mimoRankStride + p;
                            float zGated = z[zIdx] * mimoZ[mimoZIdx];
                            float silu = zGated / (1f + MathF.Exp(-zGated));
                            yR *= silu;
                        }

                        if (writePerRank)
                        {
                            int perRankIdx = perRankTokBase + r * nHead * headDim + h * headDim + p;
                            yPerRank[perRankIdx] = yR;
                        }

                        // mimo_o contraction: y[h, p] += y_r * mimo_o[h, r, p]
                        int mimoOIdx = h * mimoHeadStride + r * mimoRankStride + p;
                        contracted += yR * mimoO[mimoOIdx];
                    }
                    y[vBase + p] = contracted;
                }
            }
        }
    }

    /// <summary>
    /// MIMO streaming variant — rank-aware analog of the SISO streaming path.
    /// Applies the canonical <c>shifted_γ[T_prev-1]</c> boundary adjustment
    /// BEFORE the scan (using the previous chunk's <c>kState</c> and
    /// <c>vState</c>), then runs the same rank-summed state update as
    /// <see cref="ExecuteMimo"/>, then persists this chunk's last-token
    /// post-RoPE (pre-scale) K / V on exit.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Derivation.</b> The canonical MIMO tilelang kernel
    /// (<c>mamba3_mimo_fwd.py:275-279,391-407</c>) writes <c>FINAL_K</c> as the
    /// last row of every rank slice of the rotated K and returns
    /// <c>FINAL_V = V[:, -1]</c> (no rank axis). Its internal state update is
    /// <c>states[n, p] += Σ_{cs} exp(dA_cs_rev) · Σ_r K_r[cs, n] · Ψ[r, p] · V[cs, p]</c>
    /// — the same <c>k_sum over r</c> that <see cref="ExecuteMimo"/> applies.
    /// The canonical kernel forcibly drops <c>shifted_γ[T-1] = 0</c>, so a
    /// split call misses the contribution
    /// <c>v_last · (Σ_r k_last[r]) · DT[0_new] · (1-trap[0_new])</c>
    /// that a single-shot forward would have folded into the last-token state
    /// update at the previous chunk. Injecting that term at the start of the
    /// current chunk (BEFORE the per-chunk scan) restores bit-equivalence,
    /// in line with the SISO streaming path.
    /// </para>
    /// <para>
    /// Cross-coupling between ranks is absent: the canonical MIMO k_state
    /// stores independent per-rank K vectors (<c>k_state [R, H, N]</c>) and
    /// the boundary equation only ever sums them to contribute to the single
    /// <c>ssm_state [H, P, N]</c> buffer. No <c>mimo_o</c> / <c>mimo_z</c>
    /// appear in the boundary update — they act at output time, not on the
    /// state carry.
    /// </para>
    /// <para>
    /// <b>Alias safety.</b> <paramref name="kState"/> and <paramref name="vState"/>
    /// are read at entry and written at exit; they must not overlap any other
    /// span. Passing empty spans is equivalent to <see cref="ExecuteMimo"/>
    /// with <c>shifted_γ[T-1] = 0</c> (one-shot semantics).
    /// </para>
    /// </remarks>
    /// <param name="state">SSM hidden state <c>[H, P, N]</c>. In-place.</param>
    /// <param name="v">V per head <c>[T, H, P]</c>.</param>
    /// <param name="qRoped">Rotated biased C per rank <c>[T, R, H, N]</c>.</param>
    /// <param name="kRoped">Rotated biased B per rank <c>[T, R, H, N]</c>.</param>
    /// <param name="qkPreDotSum">Σ_r pre-rotation QK dot, <c>[T, H]</c>.</param>
    /// <param name="scale">γ + shifted_γ, <c>[T, H]</c>. Caller assembles.</param>
    /// <param name="gamma">DT·trap, <c>[T, H]</c>.</param>
    /// <param name="adt">_A·DT, <c>[T, H]</c>.</param>
    /// <param name="dt">DT only, <c>[T, H]</c>. Used for the boundary
    /// adjustment coefficient (<c>DT[0] · (1-trap[0])</c>).</param>
    /// <param name="trap">trap only, <c>[T, H]</c>. Same as above.</param>
    /// <param name="d">Per-head skip coefficient <c>[H]</c>.</param>
    /// <param name="z">Gate <c>[T, H, P]</c> or empty.</param>
    /// <param name="mimoZ">Gate rank expansion <c>[H, R, P]</c>.</param>
    /// <param name="mimoO">Output rank contraction <c>[H, R, P]</c>.</param>
    /// <param name="kState">
    /// Previous chunk's last-token post-RoPE K per rank, <c>[R, H, N]</c>.
    /// Read at entry (consumed by the boundary adjustment) and written at
    /// exit (this chunk's last-token K). Pass empty span to disable the
    /// streaming boundary — equivalent to <see cref="ExecuteMimo"/>.
    /// </param>
    /// <param name="vState">
    /// Previous chunk's last-token V, <c>[H, P]</c>. Paired with
    /// <paramref name="kState"/>. Same empty-span semantics.
    /// </param>
    /// <param name="y">Output <c>[T, H, P]</c>. Written.</param>
    /// <param name="yPerRank">Optional per-rank output <c>[T, R, H, P]</c> or empty.</param>
    /// <param name="seqLen">Token count T for this chunk.</param>
    /// <param name="nRank">MIMO rank R.</param>
    /// <param name="nHead">Head count H.</param>
    /// <param name="headDim">Channels per head P.</param>
    /// <param name="dState">State width N.</param>
    [SkipLocalsInit]
    public static void ExecuteMimoStreaming(
        Span<float> state,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> qRoped,
        ReadOnlySpan<float> kRoped,
        ReadOnlySpan<float> qkPreDotSum,
        ReadOnlySpan<float> scale,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> adt,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> trap,
        ReadOnlySpan<float> d,
        ReadOnlySpan<float> z,
        ReadOnlySpan<float> mimoZ,
        ReadOnlySpan<float> mimoO,
        Span<float> kState,
        Span<float> vState,
        Span<float> y,
        Span<float> yPerRank,
        int seqLen,
        int nRank,
        int nHead,
        int headDim,
        int dState)
    {
        if (nRank <= 0) throw new ArgumentOutOfRangeException(nameof(nRank));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));

        long hdrElems = (long)seqLen * nHead;
        if (dt.Length < hdrElems) throw new ArgumentException("dt too small.", nameof(dt));
        if (trap.Length < hdrElems) throw new ArgumentException("trap too small.", nameof(trap));

        // kState/vState are a pair: both empty disables the streaming boundary,
        // both non-empty enables it. Reject the half-configured case explicitly
        // rather than reporting it as "vState too small".
        if (kState.IsEmpty != vState.IsEmpty)
            throw new ArgumentException(
                "kState and vState must both be empty (streaming boundary disabled) or both non-empty.",
                kState.IsEmpty ? nameof(kState) : nameof(vState));

        if (!kState.IsEmpty)
        {
            long kStateElems = (long)nRank * nHead * dState;
            long vStateElems = (long)nHead * headDim;
            if (kState.Length < kStateElems)
                throw new ArgumentException("kState too small.", nameof(kState));
            if (vState.Length < vStateElems)
                throw new ArgumentException("vState too small.", nameof(vState));

            // The boundary adjustment below MUTATES state, and the persist step
            // reads v/kRoped, all before ExecuteMimo validates anything. Check the
            // buffers those steps touch up front so an undersized argument raises
            // ArgumentException instead of IndexOutOfRangeException against a
            // half-updated state.
            if (state.Length < (long)nHead * headDim * dState)
                throw new ArgumentException("state too small.", nameof(state));
            if (v.Length < (long)seqLen * nHead * headDim)
                throw new ArgumentException("v too small.", nameof(v));
            if (kRoped.Length < (long)seqLen * nRank * nHead * dState)
                throw new ArgumentException("kRoped too small.", nameof(kRoped));
        }

        // ── Chunk-boundary adjustment (canonical mamba3_mimo_fwd rank-sum form) ──
        // ssm_state[h, p, n] += v_state[h, p] · (Σ_r k_state[r, h, n]) · DT[0, h] · (1 - trap[0, h])
        // Mirrors the SISO boundary (Mamba3Block.ApplyChunkBoundaryAdjustment)
        // with the rank dimension summed — same contract as the MIMO forward
        // scan's state update (ExecuteMimo line "kSum += kRoped[...]").
        if (!kState.IsEmpty && !vState.IsEmpty && seqLen > 0)
        {
            // kState layout: [R, H, N] row-major.
            int kRankStride = nHead * dState;
            int kHeadStride = dState;
            int vHeadStride = headDim;
            int stateHeadStride = headDim * dState;

            for (int h = 0; h < nHead; h++)
            {
                float coef = dt[h] * (1f - trap[h]);
                if (coef == 0f) continue;

                // Same shape as the scan's state update with decay = 1, so the
                // rank sum is shared (and hoisted out of the channel loop).
                UpdateStateRankSummed(
                    state, kState, vState.Slice(h * vHeadStride, headDim),
                    h * stateHeadStride, h * kHeadStride, kRankStride,
                    decay: 1f, scale: coef, nRank, headDim, dState);
            }
        }

        // ── Per-chunk MIMO scan — delegate to the existing ExecuteMimo body ──
        // The scan itself is identical in math to ExecuteMimo; re-using it
        // avoids a second parallel copy of the per-token rank-sum loop and
        // guarantees the one-shot == streaming identity when the boundary
        // buffers are all-zero on the first chunk.
        ExecuteMimo(
            state, v, qRoped, kRoped, qkPreDotSum,
            scale, gamma, adt, d, z, mimoZ, mimoO,
            y, yPerRank,
            seqLen, nRank, nHead, headDim, dState);

        // ── Persist chunk-boundary buffers for the next call ──
        // Canonical final_k_state: post-RoPE, pre-scale K of the last token
        // per rank. Our kRoped spans are exactly that — the scan applies
        // `scale` inline without mutating kRoped. Canonical final_v_state:
        // raw V of the last token (no rank axis).
        if (!kState.IsEmpty && !vState.IsEmpty && seqLen > 0)
        {
            int lastTok = seqLen - 1;
            // kRoped layout: [T, R, H, N] row-major.
            int kTokStride = nRank * nHead * dState;
            int kSrcRankStride = nHead * dState;
            int kDstRankStride = nHead * dState;
            for (int r = 0; r < nRank; r++)
            {
                ReadOnlySpan<float> src = kRoped.Slice(
                    lastTok * kTokStride + r * kSrcRankStride,
                    nHead * dState);
                Span<float> dst = kState.Slice(r * kDstRankStride, nHead * dState);
                src.CopyTo(dst);
            }
            // vState: [H, P] from v[lastTok, :, :] which is [T, H, P] row-major.
            ReadOnlySpan<float> vSrc = v.Slice(lastTok * nHead * headDim, nHead * headDim);
            vSrc.CopyTo(vState);
        }
    }
}
