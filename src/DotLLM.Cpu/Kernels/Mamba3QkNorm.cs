using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// QK-Normalization wrapper used by Mamba-3 to RMS-normalize the B and C coefficient
/// tensors post-projection and pre-RoPE. Applies <see cref="RmsNorm.Execute"/>
/// independently to each <c>[d_state]</c> slice of a <c>[T, n_group, d_state]</c>
/// tensor.
/// </summary>
/// <remarks>
/// <para>
/// In the VikramKarLex/mamba3-minimal reference, <c>B_norm</c> / <c>C_norm</c> are a
/// pair of <c>RMSNorm(bc_dim)</c> layers applied before group reshape and before the
/// learnable BC bias add. In our split shape <c>[T, n_group, d_state]</c> we apply
/// the same per-<c>d_state</c>-slice normalization independently across groups.
/// The learnable weight vector is shared across <c>(t, g)</c> slices — shape
/// <c>[d_state]</c> — matching the reference's <c>RMSNorm(bc_dim)</c> (in SISO
/// <c>bc_dim = d_state</c> and <c>n_group = 1</c>, so the two shapes coincide).
/// </para>
/// <para>
/// This kernel exists purely as a named wrapper around <see cref="RmsNorm.Execute"/>
/// to keep the Mamba-3 scan orchestrator free of a double-nested loop. No math is
/// re-implemented.
/// </para>
/// <para>
/// <b>Alias safety:</b> <c>bc</c> is normalized in place.
/// <see cref="RmsNorm.Execute"/> is already alias-safe for in-place input/output,
/// so passing the same span as both logical-input and logical-output is the normal
/// mode of use. The <c>weight</c> span must not overlap
/// <c>bc</c>.
/// </para>
/// </remarks>
public static class Mamba3QkNorm
{
    /// <summary>
    /// Applies <see cref="RmsNorm.Execute"/> to each <c>[d_state]</c> slice of
    /// <paramref name="bc"/>, shape <c>[T, n_group, d_state]</c> row-major.
    /// Normalizes in place.
    /// </summary>
    /// <param name="bc">
    /// B or C tensor, shape <c>[T, n_group, d_state]</c> row-major. Overwritten in
    /// place with its RMS-normalized value. Length must be <c>T * n_group * d_state</c>.
    /// </param>
    /// <param name="weight">
    /// Per-element RMSNorm scale, length <c>d_state</c>. Shared across all
    /// <c>(t, g)</c> slices, matching the reference's <c>RMSNorm(bc_dim)</c> layer.
    /// Must not overlap <paramref name="bc"/>.
    /// </param>
    /// <param name="epsilon">RMSNorm stabilizing constant (typically <c>1e-5</c>).</param>
    /// <param name="seqLen">Number of tokens <c>T</c>.</param>
    /// <param name="nGroup">Number of B/C groups.</param>
    /// <param name="dState">State-space width per group.</param>
    [SkipLocalsInit]
    public static void Execute(
        Span<float> bc,
        ReadOnlySpan<float> weight,
        float epsilon,
        int seqLen,
        int nGroup,
        int dState)
    {
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nGroup <= 0) throw new ArgumentOutOfRangeException(nameof(nGroup));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));

        long expected = (long)seqLen * nGroup * dState;
        if (bc.Length < expected)
            throw new ArgumentException(
                $"bc length {bc.Length} < T*n_group*d_state = {expected}.", nameof(bc));
        if (weight.Length < dState)
            throw new ArgumentException(
                $"weight length {weight.Length} < d_state = {dState}.", nameof(weight));

        if (seqLen == 0) return;

        ReadOnlySpan<float> w = weight[..dState];
        int sliceCount = seqLen * nGroup;
        for (int i = 0; i < sliceCount; i++)
        {
            Span<float> slice = bc.Slice(i * dState, dState);
            // RmsNorm.Execute is alias-safe for input == result.
            RmsNorm.Execute(slice, w, epsilon, slice);
        }
    }
}
