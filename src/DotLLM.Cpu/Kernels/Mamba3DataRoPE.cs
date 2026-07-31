using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Rotation convention selector for <see cref="Mamba3DataRoPE"/>'s canonical
/// entry points. The minimal-reference path (kept for regression) always uses
/// <see cref="Pairwise"/> over the full <c>d_state</c>; the canonical
/// <c>state-spaces/mamba</c> kernels select between Pairwise (SISO,
/// <c>mamba3_siso_combined</c>) and <see cref="Halved"/> (MIMO,
/// <c>mamba3_mimo_combined</c>), and rotate only the first
/// <c>2 * num_rope_angles</c> channels when <c>rope_fraction &lt; 1</c>.
/// </summary>
public enum Mamba3RoPEMode
{
    /// <summary>
    /// Interleaved adjacent pairs <c>(x[2k], x[2k+1])</c>. SISO canonical and
    /// the minimal reference both use this convention.
    /// </summary>
    Pairwise = 0,

    /// <summary>
    /// Halved pairs <c>(x[k], x[k + d_state/2])</c>. Canonical MIMO kernel.
    /// </summary>
    Halved = 1,
}

/// <summary>
/// Data-dependent 2-D rotation applied to the Mamba-3 <c>B</c> and <c>C</c>
/// coefficients. Absorbs the block-diagonal complex rotation matrix <c>R_t</c>
/// from Proposition 3 of Lahoti et al. (ICLR 2026, arXiv 2603.15569) into the
/// real-valued <c>B, C</c> projections so the SSM scan kernel itself stays
/// identical to Mamba-2.
/// </summary>
/// <remarks>
/// <para>
/// Unlike standard position-based RoPE (<see cref="RoPE"/>) whose cos/sin
/// tables depend only on sequence index, the Mamba-3 rotation angles are
/// <b>data-dependent</b>: each per-token timestep <c>dt[t, h]</c> modulates a
/// per-token frequency projection <c>θ[t, k]</c> (projected from the input,
/// not a learned-per-head table), and the angles accumulate along time with a
/// <b>negative</b> cumulative sum:
/// </para>
/// <code>
/// raw_angles[t, h, k] =  dt[t, h] * theta[t, k]           // outer product
/// cum_angles[t, h, k] = -sum_{s=0..t} raw_angles[s, h, k] // NEGATIVE cumsum
/// </code>
/// <para>
/// Each <c>d_state</c>-dim slice of <c>B</c> and <c>C</c> is then split into
/// adjacent even/odd pairs <c>(v[2k], v[2k+1])</c> and rotated by the 2-D
/// matrix <c>R(φ) = [[cos φ, −sin φ], [sin φ, cos φ]]</c>:
/// </para>
/// <code>
/// v'[2k]   = cos·v[2k] − sin·v[2k+1]
/// v'[2k+1] = sin·v[2k] + cos·v[2k+1]
/// </code>
/// <para>
/// <b>Shape conventions (matched to <c>VikramKarLex/mamba3-minimal</c>'s
/// <c>apply_rope</c>):</b>
/// </para>
/// <list type="bullet">
///   <item><description><c>b</c>, <c>c</c> have shape <c>[T, n_head, d_state]</c> row-major.
///         The caller broadcasts any <c>[T, n_group, d_state]</c> pre-QkNorm tensor to
///         the per-head layout before calling (typically via the learnable <c>B_bias</c>
///         / <c>C_bias</c> add of shape <c>[n_head, d_state]</c>, which implicitly
///         performs the broadcast).</description></item>
///   <item><description><c>dt</c> is <c>[T, n_head]</c> post-softplus (same convention
///         as <c>Mamba2SelectiveScan</c>).</description></item>
///   <item><description><c>theta</c> is <c>[T, d_state/2]</c> — projected from the
///         input token (shared across heads). The task brief assumed a learned
///         <c>[n_head, d_state/2]</c> table; the reference impl uses a per-token
///         projection instead, and this kernel follows the reference.</description></item>
///   <item><description>Pair ordering is <b>interleaved</b> (even/odd adjacent), matching
///         <see cref="RoPE.ApplyRotation"/> (the "Norm" pairing) rather than GPT-NeoX
///         half-split pairing. The reference's <c>x[..., 0::2]</c> / <c>x[..., 1::2]</c>
///         corresponds exactly to this.</description></item>
///   <item><description>Angle sign is <b>negative</b> cumulative sum, locking the
///         VikramKarLex/mamba3-minimal convention.</description></item>
/// </list>
/// <para>
/// <b>Alias safety.</b> <c>b</c> and <c>c</c> are rotated in place; each
/// element is read and overwritten within the same inner-pair iteration
/// (even and odd of the same pair are read into locals before either is
/// written), so in-place is safe. <c>b</c> and <c>c</c> may be the same span
/// — the two rotations operate identically on their own buffer without
/// cross-interference. <c>dt</c> and <c>theta</c> are read-only and must not
/// overlap the output buffers.
/// </para>
/// </remarks>
public static class Mamba3DataRoPE
{
    /// <summary>Stackalloc threshold in floats for the per-sequence scratch buffer.</summary>
    private const int StackAllocFloatThreshold = 2048;

    /// <summary>
    /// Applies Mamba-3 data-dependent RoPE to <paramref name="b"/> and <paramref name="c"/>
    /// in place. Both are shape <c>[T, n_head, d_state]</c> row-major.
    /// </summary>
    /// <param name="b">
    /// Mamba-3 <c>B</c> coefficient, shape <c>[T, n_head, d_state]</c> row-major,
    /// length <c>T * n_head * d_state</c>. Modified in place.
    /// </param>
    /// <param name="c">
    /// Mamba-3 <c>C</c> coefficient, shape <c>[T, n_head, d_state]</c> row-major,
    /// length <c>T * n_head * d_state</c>. Modified in place.
    /// </param>
    /// <param name="dt">
    /// Post-softplus timestep, shape <c>[T, n_head]</c> row-major, length <c>T * n_head</c>.
    /// </param>
    /// <param name="theta">
    /// Per-token frequency projection, shape <c>[T, d_state/2]</c> row-major,
    /// length <c>T * d_state / 2</c>. Shared across heads.
    /// </param>
    /// <param name="seqLen">Number of tokens <c>T</c>.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="dState">State width (must be even).</param>
    [SkipLocalsInit]
    public static void Execute(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> theta,
        int seqLen,
        int nHead,
        int dState)
        => Execute(b, c, dt, theta, cumAnglePrev: ReadOnlySpan<float>.Empty,
                   cumAngleOut: Span<float>.Empty, seqLen, nHead, dState);

    /// <summary>
    /// Applies Mamba-3 data-dependent RoPE with explicit cumulative-angle continuity.
    /// Pass the previous call's final cum_angle via <paramref name="cumAnglePrev"/>
    /// and receive this call's final cum_angle via <paramref name="cumAngleOut"/>;
    /// this is what lets autoregressive decode resume the rotation phase where
    /// the previous call left off rather than resetting to 0 every step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Semantics mirror <c>VikramKarLex/mamba3-minimal</c>'s <c>InferenceCache.cum_angle</c>:
    /// </para>
    /// <code>
    /// cum_angles[t, h, k] = cumAnglePrev[h, k] − Σ_{s=0..t} dt[s, h] · theta[s, k]
    /// cumAngleOut[h, k]   = cum_angles[T-1, h, k]
    /// </code>
    /// <para>
    /// Pass an empty span for <paramref name="cumAnglePrev"/> to start from zeros
    /// (prefill / single-shot behaviour). Pass an empty span for
    /// <paramref name="cumAngleOut"/> if the caller does not need the final angle.
    /// When non-empty, both must have length <c>n_head · d_state/2</c> and row-major
    /// layout <c>[n_head, d_state/2]</c>. <paramref name="cumAnglePrev"/> and
    /// <paramref name="cumAngleOut"/> may alias (same buffer is fine — it is read in
    /// full before any writes). <paramref name="b"/> and <paramref name="c"/>, by
    /// contrast, must <b>not</b> overlap: each is rotated in place, so a shared
    /// buffer would be rotated twice. Overlap is rejected with an
    /// <see cref="ArgumentException"/>.
    /// </para>
    /// </remarks>
    /// <param name="b">As in the no-offset overload.</param>
    /// <param name="c">As in the no-offset overload.</param>
    /// <param name="dt">As in the no-offset overload.</param>
    /// <param name="theta">As in the no-offset overload.</param>
    /// <param name="cumAnglePrev">
    /// Starting cum_angle, shape <c>[n_head, d_state/2]</c>. Empty span = start from 0.
    /// </param>
    /// <param name="cumAngleOut">
    /// Receives the final cum_angle after the last token, shape <c>[n_head, d_state/2]</c>.
    /// Empty span = do not write.
    /// </param>
    /// <param name="seqLen">Number of tokens <c>T</c>.</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="dState">State width (must be even).</param>
    [SkipLocalsInit]
    public static void Execute(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> theta,
        ReadOnlySpan<float> cumAnglePrev,
        Span<float> cumAngleOut,
        int seqLen,
        int nHead,
        int dState)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if ((dState & 1) != 0)
            throw new ArgumentException($"d_state must be even for 2-D pair rotation, got {dState}.", nameof(dState));

        int halfDState = dState >> 1;
        long bcLen = (long)seqLen * nHead * dState;
        long dtLen = (long)seqLen * nHead;
        long thetaLen = (long)seqLen * halfDState;
        int cumLen = nHead * halfDState;

        if (b.Length < bcLen)
            throw new ArgumentException($"b length {b.Length} < T*n_head*d_state = {bcLen}.", nameof(b));
        if (c.Length < bcLen)
            throw new ArgumentException($"c length {c.Length} < T*n_head*d_state = {bcLen}.", nameof(c));
        // B and C are rotated by the same angles but are distinct tensors. If they
        // overlapped, the second rotation would re-rotate already-rotated data and
        // silently produce wrong values, so reject it rather than guess the intent.
        if (b.Overlaps(c))
            throw new ArgumentException("b and c must not overlap; each is rotated in place.", nameof(c));
        if (dt.Length < dtLen)
            throw new ArgumentException($"dt length {dt.Length} < T*n_head = {dtLen}.", nameof(dt));
        if (theta.Length < thetaLen)
            throw new ArgumentException($"theta length {theta.Length} < T*d_state/2 = {thetaLen}.", nameof(theta));
        if (!cumAnglePrev.IsEmpty && cumAnglePrev.Length < cumLen)
            throw new ArgumentException(
                $"cumAnglePrev length {cumAnglePrev.Length} < n_head*d_state/2 = {cumLen}.",
                nameof(cumAnglePrev));
        if (!cumAngleOut.IsEmpty && cumAngleOut.Length < cumLen)
            throw new ArgumentException(
                $"cumAngleOut length {cumAngleOut.Length} < n_head*d_state/2 = {cumLen}.",
                nameof(cumAngleOut));

        if (seqLen == 0)
        {
            // Pass-through: final angle equals the starting angle.
            if (!cumAngleOut.IsEmpty)
            {
                if (!cumAnglePrev.IsEmpty)
                    cumAnglePrev.Slice(0, cumLen).CopyTo(cumAngleOut.Slice(0, cumLen));
                else
                    cumAngleOut.Slice(0, cumLen).Clear();
            }
            return;
        }

        // Scratch for cum_angles of ONE time step at a time: [n_head, halfDState].
        // The recurrence is along time, so we can't vectorize cumsum across t without
        // materializing the whole table. We maintain the running cumulative angles in
        // a single [n_head * halfDState] buffer and reuse it across tokens.
        int scratchLen = cumLen;
        int trigLen = scratchLen; // cos/sin computed from the same buffer shape per step

        // Stack vs pool based on total bytes needed (cum + cos + sin).
        int totalFloats = scratchLen + 2 * trigLen;
        if (totalFloats <= StackAllocFloatThreshold)
        {
            Span<float> scratch = stackalloc float[totalFloats];
            ExecuteInto(b, c, dt, theta, cumAnglePrev, cumAngleOut,
                        seqLen, nHead, halfDState, scratch);
        }
        else
        {
            float[] rented = ArrayPool<float>.Shared.Rent(totalFloats);
            try
            {
                ExecuteInto(b, c, dt, theta, cumAnglePrev, cumAngleOut,
                            seqLen, nHead, halfDState,
                            rented.AsSpan(0, totalFloats));
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    [SkipLocalsInit]
    private static void ExecuteInto(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> theta,
        ReadOnlySpan<float> cumAnglePrev,
        Span<float> cumAngleOut,
        int seqLen,
        int nHead,
        int halfDState,
        Span<float> scratch)
    {
        int scratchLen = nHead * halfDState;
        Span<float> cumAngles = scratch.Slice(0, scratchLen);
        Span<float> cosTable  = scratch.Slice(scratchLen, scratchLen);
        Span<float> sinTable  = scratch.Slice(2 * scratchLen, scratchLen);

        // Seed cumulative angles from the caller's previous final angle (if provided)
        // so decode continuity works: cum_angles[t=0, h, k] = cumAnglePrev[h, k] − dt[0,h]·theta[0,k].
        if (cumAnglePrev.IsEmpty)
            cumAngles.Clear();
        else
            cumAnglePrev.Slice(0, scratchLen).CopyTo(cumAngles);

        int dState = halfDState * 2;
        int bcRowStride = nHead * dState;    // stride to the next token in B / C
        int bcHeadStride = dState;           // stride to the next head within a token
        int dtRowStride = nHead;
        int thetaRowStride = halfDState;

        for (int t = 0; t < seqLen; t++)
        {
            ReadOnlySpan<float> dtRow = dt.Slice(t * dtRowStride, nHead);
            ReadOnlySpan<float> thetaRow = theta.Slice(t * thetaRowStride, halfDState);

            // Update cumulative angles:
            //   cum_angles[t, h, k] = cum_angles[t-1, h, k] - dt[t, h] * theta[t, k]
            // Per-head: fused multiply-subtract across k lanes; theta[t, :] is the vector
            // and dt[t, h] is the broadcast scalar.
            for (int h = 0; h < nHead; h++)
            {
                Span<float> cumRow = cumAngles.Slice(h * halfDState, halfDState);
                float scale = -dtRow[h];
                // cum += scale * theta  (scale is negative, implementing the NEGATIVE cumsum).
                TensorPrimitives.MultiplyAdd(thetaRow, scale, cumRow, cumRow);
            }

            // Trig: cos/sin over the entire [n_head, halfDState] block at once.
            TensorPrimitives.Cos(cumAngles, cosTable);
            TensorPrimitives.Sin(cumAngles, sinTable);

            // Apply rotation to B and C for this token, per head.
            int bcTokenBase = t * bcRowStride;
            for (int h = 0; h < nHead; h++)
            {
                int bcBase = bcTokenBase + h * bcHeadStride;
                int trigBase = h * halfDState;

                Span<float> bHead = b.Slice(bcBase, dState);
                Span<float> cHead = c.Slice(bcBase, dState);
                ReadOnlySpan<float> cosSlice = cosTable.Slice(trigBase, halfDState);
                ReadOnlySpan<float> sinSlice = sinTable.Slice(trigBase, halfDState);

                ApplyPairRotation(bHead, cosSlice, sinSlice, halfDState);
                ApplyPairRotation(cHead, cosSlice, sinSlice, halfDState);
            }
        }

        // Export final cum_angle so the next call (decode step) can resume from here.
        if (!cumAngleOut.IsEmpty)
            cumAngles.CopyTo(cumAngleOut.Slice(0, scratchLen));
    }

    /// <summary>
    /// Applies interleaved pair rotation in place to a single <c>[d_state]</c> slice.
    /// Same convention as <see cref="RoPE.ApplyRotation"/> (pairs are <c>(v[2k], v[2k+1])</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    [SkipLocalsInit]
    private static void ApplyPairRotation(
        Span<float> vec,
        ReadOnlySpan<float> cos,
        ReadOnlySpan<float> sin,
        int halfDim)
    {
        // Read both halves into locals before writing — alias-safe on the same buffer.
        for (int k = 0; k < halfDim; k++)
        {
            float e = vec[2 * k];
            float o = vec[2 * k + 1];
            float co = cos[k];
            float si = sin[k];
            vec[2 * k]     = co * e - si * o;
            vec[2 * k + 1] = si * e + co * o;
        }
    }

    /// <summary>
    /// Scalar reference implementation kept for unit-test pinning. Identical numerics
    /// to <see cref="Execute(Span{float}, Span{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, int, int, int)"/>
    /// but without the <see cref="TensorPrimitives"/>
    /// fused-multiply-add or batched trig calls — every operation is a plain scalar
    /// floating-point op, so this is the ground truth.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> theta,
        int seqLen,
        int nHead,
        int dState)
    {
        if ((dState & 1) != 0)
            throw new ArgumentException($"d_state must be even for 2-D pair rotation, got {dState}.", nameof(dState));
        int halfDState = dState >> 1;

        // Per-head per-lane running cumulative sum (negative). Scalar reference is
        // only invoked from unit tests on tiny inputs, but we still honor the
        // project-wide "no managed tensor allocations" rule via ArrayPool fallback.
        int scratchLen = nHead * halfDState;
        float[] cumAnglesArr = System.Buffers.ArrayPool<float>.Shared.Rent(scratchLen);
        Span<float> cumAngles = cumAnglesArr.AsSpan(0, scratchLen);
        cumAngles.Clear();
        try
        {
            ExecuteScalarInto(b, c, dt, theta, seqLen, nHead, halfDState, cumAngles);
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(cumAnglesArr);
        }
    }

    [SkipLocalsInit]
    private static void ExecuteScalarInto(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> theta,
        int seqLen,
        int nHead,
        int halfDState,
        Span<float> cumAngles)
    {
        int dState = halfDState * 2;

        for (int t = 0; t < seqLen; t++)
        {
            int dtBase = t * nHead;
            int thetaBase = t * halfDState;
            int bcTokenBase = t * nHead * dState;

            for (int h = 0; h < nHead; h++)
            {
                float dtv = dt[dtBase + h];
                int cumBase = h * halfDState;
                int bcBase = bcTokenBase + h * dState;

                for (int k = 0; k < halfDState; k++)
                {
                    // Running negative cumulative sum: angle_{t,h,k} = -sum_{s<=t}(dt[s,h] * theta[s,k])
                    float thk = theta[thetaBase + k];
                    cumAngles[cumBase + k] -= dtv * thk;
                    float phi = cumAngles[cumBase + k];
                    float co = MathF.Cos(phi);
                    float si = MathF.Sin(phi);

                    // Rotate B pair.
                    float be = b[bcBase + 2 * k];
                    float bo = b[bcBase + 2 * k + 1];
                    b[bcBase + 2 * k]     = co * be - si * bo;
                    b[bcBase + 2 * k + 1] = si * be + co * bo;

                    // Rotate C pair (same angle).
                    float ce = c[bcBase + 2 * k];
                    float co2 = c[bcBase + 2 * k + 1];
                    c[bcBase + 2 * k]     = co * ce - si * co2;
                    c[bcBase + 2 * k + 1] = si * ce + co * co2;
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Canonical (state-spaces/mamba) entry point.
    //
    // The canonical Mamba-3 RoPE differs from the minimal reference in four
    // ways that are all expressed as parameters here:
    //
    //   1. Raw angles enter the kernel as a per-token projection
    //      (angles_raw, shape [T, num_rope_angles]) and are broadcast to
    //      [T, nHead, num_rope_angles] before use — the kernel always wraps
    //      them through tanh(·)·π first (see angle_dt.py:97-101 and
    //      mamba3_mimo_rotary_step.py:255).
    //   2. The cumulative sum is positive-sign (canonical kernel convention;
    //      sign is absorbed into the projection head rather than the kernel).
    //   3. After cumulation the angle is reduced mod 2π inline (canonical
    //      angle_dt.py:108 — purely a numeric-drift guard, mathematically a
    //      no-op).
    //   4. Only the first 2*num_rope_angles channels are rotated
    //      (partial rotation when rope_fraction < 1); the remainder of
    //      d_state is passed through unchanged.
    //
    // The kernel additionally exposes a rank axis to match canonical's
    // (B, L, R, H, N) layout for MIMO — set nRank=1 for SISO. The rotation
    // mode (pairwise vs halved) is selected by <paramref name="mode"/>.
    // Cumulative-angle continuity semantics mirror the minimal path so the
    // block can thread a cumAngle buffer across decode steps.
    //
    // Kept as a separate entry point because the minimal reference and the
    // minimal-baselined Block composer still call the original Execute.
    // Deletion of the minimal path is deferred to Stage P2b (Block rewrite).
    // ------------------------------------------------------------------------

    /// <summary>
    /// Applies canonical Mamba-3 data-dependent RoPE to <paramref name="b"/>
    /// and <paramref name="c"/> in place. Both are shape
    /// <c>[T, nRank, nHead, dState]</c> row-major (nRank=1 for SISO).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Angle computation</b> (per canonical's <c>angle_dt_fwd</c>):
    /// </para>
    /// <code>
    /// vals[t, h, s]   = tanh(angles_raw[t, s]) * π * dt[t, h]
    /// cum[t, h, s]    = cumAnglePrev[h, s] + Σ_{u=0..t} vals[u, h, s]   // cumsum (positive)
    /// cum[t, h, s]   %= 2π                                              // mod 2π inline
    /// </code>
    /// <para>
    /// <b>Rotation</b>: only the first <c>rotary_dim = 2 * numRopeAngles</c>
    /// channels of each <c>[dState]</c> slice are rotated; the tail is passed
    /// through. When <paramref name="mode"/> is <see cref="Mamba3RoPEMode.Pairwise"/>,
    /// the rotated region is split into interleaved pairs <c>(v[2k], v[2k+1])</c>.
    /// When <see cref="Mamba3RoPEMode.Halved"/>, the full <c>dState</c> is split in
    /// halves and the first <c>numRopeAngles</c> lanes of each half are rotated
    /// as pairs <c>(v[k], v[k + dState/2])</c>; the rotary tail of each half
    /// passes through unchanged (i.e. cos=1, sin=0 pad).
    /// </para>
    /// </remarks>
    /// <param name="b">B coefficient, shape <c>[T, nRank, nHead, dState]</c>. Modified in place.</param>
    /// <param name="c">C coefficient, shape <c>[T, nRank, nHead, dState]</c>. Modified in place.</param>
    /// <param name="anglesRaw">Per-token angle projection, shape <c>[T, numRopeAngles]</c> (shared across rank &amp; head).</param>
    /// <param name="dt">Post-softplus timestep, shape <c>[T, nHead]</c>.</param>
    /// <param name="cumAnglePrev">Seed cum_angle, shape <c>[nHead, numRopeAngles]</c>. Empty = start from 0.</param>
    /// <param name="cumAngleOut">Final cum_angle after last token, shape <c>[nHead, numRopeAngles]</c>. Empty = don't write.</param>
    /// <param name="seqLen">Number of tokens T.</param>
    /// <param name="nRank">MIMO rank R (SISO = 1).</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="dState">State width.</param>
    /// <param name="numRopeAngles">Rotated-pair count — rotates first <c>2 * numRopeAngles</c> channels.</param>
    /// <param name="mode">Pair ordering (Pairwise for SISO, Halved for MIMO).</param>
    [SkipLocalsInit]
    public static void ExecuteCanonical(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> anglesRaw,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> cumAnglePrev,
        Span<float> cumAngleOut,
        int seqLen,
        int nRank,
        int nHead,
        int dState,
        int numRopeAngles,
        Mamba3RoPEMode mode)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nRank <= 0) throw new ArgumentOutOfRangeException(nameof(nRank));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (numRopeAngles <= 0) throw new ArgumentOutOfRangeException(nameof(numRopeAngles));
        int rotaryDim = 2 * numRopeAngles;
        if (rotaryDim > dState)
            throw new ArgumentException($"2*numRopeAngles ({rotaryDim}) > dState ({dState}).", nameof(numRopeAngles));
        if (mode == Mamba3RoPEMode.Halved && (dState & 1) != 0)
            throw new ArgumentException($"Halved mode requires even dState, got {dState}.", nameof(dState));

        long bcLen = (long)seqLen * nRank * nHead * dState;
        long angLen = (long)seqLen * numRopeAngles;
        long dtLen = (long)seqLen * nHead;
        int cumLen = nHead * numRopeAngles;

        if (b.Length < bcLen)
            throw new ArgumentException($"b length {b.Length} < T*R*H*N = {bcLen}.", nameof(b));
        if (c.Length < bcLen)
            throw new ArgumentException($"c length {c.Length} < T*R*H*N = {bcLen}.", nameof(c));
        // See Execute: overlapping b/c would be rotated twice.
        if (b.Overlaps(c))
            throw new ArgumentException("b and c must not overlap; each is rotated in place.", nameof(c));
        if (anglesRaw.Length < angLen)
            throw new ArgumentException($"anglesRaw length {anglesRaw.Length} < T*numRopeAngles = {angLen}.", nameof(anglesRaw));
        if (dt.Length < dtLen)
            throw new ArgumentException($"dt length {dt.Length} < T*nHead = {dtLen}.", nameof(dt));
        if (!cumAnglePrev.IsEmpty && cumAnglePrev.Length < cumLen)
            throw new ArgumentException($"cumAnglePrev length {cumAnglePrev.Length} < nHead*numRopeAngles = {cumLen}.", nameof(cumAnglePrev));
        if (!cumAngleOut.IsEmpty && cumAngleOut.Length < cumLen)
            throw new ArgumentException($"cumAngleOut length {cumAngleOut.Length} < nHead*numRopeAngles = {cumLen}.", nameof(cumAngleOut));

        if (seqLen == 0)
        {
            if (!cumAngleOut.IsEmpty)
            {
                if (!cumAnglePrev.IsEmpty)
                    cumAnglePrev.Slice(0, cumLen).CopyTo(cumAngleOut.Slice(0, cumLen));
                else
                    cumAngleOut.Slice(0, cumLen).Clear();
            }
            return;
        }

        // scratch: running cum[nHead, numRopeAngles] + tanh(angles_raw)*π slab [numRopeAngles] + cos/sin tables [nHead, numRopeAngles]
        int cumSize = cumLen;
        int trigSize = cumLen;
        int vecSize = numRopeAngles;
        int totalFloats = cumSize + trigSize * 2 + vecSize;

        if (totalFloats <= StackAllocFloatThreshold)
        {
            Span<float> scratch = stackalloc float[totalFloats];
            ExecuteCanonicalInto(
                b, c, anglesRaw, dt, cumAnglePrev, cumAngleOut,
                seqLen, nRank, nHead, dState, numRopeAngles, mode,
                scratch);
        }
        else
        {
            float[] rented = ArrayPool<float>.Shared.Rent(totalFloats);
            try
            {
                ExecuteCanonicalInto(
                    b, c, anglesRaw, dt, cumAnglePrev, cumAngleOut,
                    seqLen, nRank, nHead, dState, numRopeAngles, mode,
                    rented.AsSpan(0, totalFloats));
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }
    }

    [SkipLocalsInit]
    private static void ExecuteCanonicalInto(
        Span<float> b,
        Span<float> c,
        ReadOnlySpan<float> anglesRaw,
        ReadOnlySpan<float> dt,
        ReadOnlySpan<float> cumAnglePrev,
        Span<float> cumAngleOut,
        int seqLen,
        int nRank,
        int nHead,
        int dState,
        int numRopeAngles,
        Mamba3RoPEMode mode,
        Span<float> scratch)
    {
        int cumSize = nHead * numRopeAngles;
        int trigSize = cumSize;
        int vecSize = numRopeAngles;

        Span<float> cum = scratch.Slice(0, cumSize);
        Span<float> cosTable = scratch.Slice(cumSize, trigSize);
        Span<float> sinTable = scratch.Slice(cumSize + trigSize, trigSize);
        Span<float> tanhPiVec = scratch.Slice(cumSize + trigSize * 2, vecSize);

        // Seed cum from caller.
        if (cumAnglePrev.IsEmpty)
            cum.Clear();
        else
            cumAnglePrev.Slice(0, cumSize).CopyTo(cum);

        // Strides (row-major).
        int rotaryDim = 2 * numRopeAngles;
        int bcHeadStride = dState;
        int bcRankStride = nHead * dState;
        int bcTokenStride = nRank * bcRankStride;
        const float twoPi = 2f * MathF.PI;
        const float invTwoPi = 1f / twoPi;
        int halfDState = dState >> 1;                    // only used in halved mode

        for (int t = 0; t < seqLen; t++)
        {
            // tanh(angles_raw[t]) * π  →  a slab of [numRopeAngles]
            ReadOnlySpan<float> angRow = anglesRaw.Slice(t * numRopeAngles, numRopeAngles);
            TensorPrimitives.Tanh(angRow, tanhPiVec);
            TensorPrimitives.Multiply(tanhPiVec, MathF.PI, tanhPiVec);

            // For each head: cum[h] += dt[t,h] * tanhPiVec  (positive cumsum)
            ReadOnlySpan<float> dtRow = dt.Slice(t * nHead, nHead);
            for (int h = 0; h < nHead; h++)
            {
                Span<float> cumRow = cum.Slice(h * numRopeAngles, numRopeAngles);
                float dth = dtRow[h];
                TensorPrimitives.MultiplyAdd(tanhPiVec, dth, cumRow, cumRow);
                // mod 2π lane-wise — scalar loop is fine for typical numRopeAngles (32..64).
                for (int k = 0; k < numRopeAngles; k++)
                {
                    float v = cumRow[k];
                    float floored = MathF.Floor(v * invTwoPi);
                    cumRow[k] = v - twoPi * floored;
                }
            }

            TensorPrimitives.Cos(cum, cosTable);
            TensorPrimitives.Sin(cum, sinTable);

            // Apply rotation to b and c for this token across all rank slices.
            int tokenBase = t * bcTokenStride;
            for (int r = 0; r < nRank; r++)
            {
                int rankBase = tokenBase + r * bcRankStride;
                for (int h = 0; h < nHead; h++)
                {
                    int bcBase = rankBase + h * bcHeadStride;
                    int trigBase = h * numRopeAngles;
                    Span<float> bSlice = b.Slice(bcBase, dState);
                    Span<float> cSlice = c.Slice(bcBase, dState);
                    ReadOnlySpan<float> cosSlice = cosTable.Slice(trigBase, numRopeAngles);
                    ReadOnlySpan<float> sinSlice = sinTable.Slice(trigBase, numRopeAngles);

                    if (mode == Mamba3RoPEMode.Pairwise)
                    {
                        ApplyPairRotation(bSlice.Slice(0, rotaryDim), cosSlice, sinSlice, numRopeAngles);
                        ApplyPairRotation(cSlice.Slice(0, rotaryDim), cosSlice, sinSlice, numRopeAngles);
                        // Channels [rotaryDim..dState) are pass-through (untouched).
                    }
                    else
                    {
                        ApplyHalvedRotation(bSlice, cosSlice, sinSlice, numRopeAngles, halfDState);
                        ApplyHalvedRotation(cSlice, cosSlice, sinSlice, numRopeAngles, halfDState);
                    }
                }
            }
        }

        if (!cumAngleOut.IsEmpty)
            cum.CopyTo(cumAngleOut.Slice(0, cumSize));
    }

    /// <summary>
    /// Halved rotation: pairs are <c>(x[k], x[k + dState/2])</c>. Rotates only
    /// the first <paramref name="numRopeAngles"/> lanes of each half; the
    /// remaining <c>halfDState - numRopeAngles</c> lanes of each half pass
    /// through unchanged (canonical's cos=1 / sin=0 padding). Operates in place.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    [SkipLocalsInit]
    private static void ApplyHalvedRotation(
        Span<float> vec,
        ReadOnlySpan<float> cos,
        ReadOnlySpan<float> sin,
        int numRopeAngles,
        int halfDState)
    {
        // Read both halves into locals before writing the first half, so the
        // update is alias-safe on the same buffer.
        for (int k = 0; k < numRopeAngles; k++)
        {
            float a = vec[k];
            float bval = vec[halfDState + k];
            float co = cos[k];
            float si = sin[k];
            vec[k] = co * a - si * bval;
            vec[halfDState + k] = si * a + co * bval;
        }
        // Tail lanes [numRopeAngles..halfDState) pass through unchanged.
    }
}
