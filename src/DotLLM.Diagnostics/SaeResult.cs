namespace DotLLM.Diagnostics;

/// <summary>
/// Result of a single SAE encode + top-K extraction at one (layer, position) site.
/// </summary>
/// <remarks>
/// <para>
/// Produced by <see cref="SaeHook.GetResults"/>. Holds the indices and magnitudes of the K
/// strongest active features (post-ReLU encoder output) at this site, along with the L2
/// reconstruction error of decoding those K features alone back to the original activation.
/// </para>
/// <para>
/// <see cref="ActiveFeatureCount"/> reports how many post-ReLU features fired across the
/// entire dictionary (potentially much larger than K) — the difference between
/// <see cref="ActiveFeatureCount"/> and <see cref="FeatureIndices"/>.<c>Length</c> indicates
/// how much sparsity information is lost by the top-K truncation.
/// </para>
/// </remarks>
public sealed class SaeResult
{
    /// <summary>Transformer layer index this result is for (0-based).</summary>
    public int LayerIndex { get; init; }

    /// <summary>Token position within the sequence (0-based).</summary>
    public int TokenPosition { get; init; }

    /// <summary>
    /// Indices of the top-K active features in descending magnitude order. Indices are into
    /// the SAE feature dictionary (range <c>[0, FeatureCount)</c>).
    /// </summary>
    public required int[] FeatureIndices { get; init; }

    /// <summary>
    /// Post-ReLU encoder magnitudes for <see cref="FeatureIndices"/>, parallel array,
    /// descending. All entries are non-negative (ReLU output).
    /// </summary>
    public required float[] FeatureMagnitudes { get; init; }

    /// <summary>
    /// L2 norm of (original activation − decoded top-K reconstruction). Lower is better;
    /// the lower bound for a well-trained SAE with K matching its training sparsity is
    /// typically a few percent of <c>||activation||</c>.
    /// </summary>
    /// <remarks>
    /// Computed against the top-K-only reconstruction, NOT the full-dictionary reconstruction.
    /// To compare against the underlying SAE's true reconstruction quality, set
    /// <see cref="SaeConfig.TopK"/> to <see cref="ISparseAutoencoder.FeatureCount"/>.
    /// </remarks>
    public required float ReconstructionError { get; init; }

    /// <summary>
    /// Total number of features that fired (post-ReLU value &gt; 0) across the full
    /// dictionary, before the top-K truncation. Useful as a sparsity proxy.
    /// </summary>
    public required int ActiveFeatureCount { get; init; }
}
