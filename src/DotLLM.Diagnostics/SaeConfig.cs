namespace DotLLM.Diagnostics;

/// <summary>
/// Configuration for <see cref="SaeHook"/> — which layers and token positions to analyse and
/// how many top-K features to retain per (layer, position) invocation.
/// </summary>
/// <remarks>
/// <para>
/// Layer / position selection mirrors <see cref="LogitLensConfig"/>. The active-feature count
/// (<see cref="TopK"/>) is the dominant scalability lever — SAE dictionaries are typically
/// 8-64× wider than the residual stream (d_sae ≫ d_in) but trained to fire only a few features
/// at a time, so retaining only the top-K is the standard mechanistic-interpretability workflow.
/// </para>
/// </remarks>
public sealed class SaeConfig
{
    /// <summary>
    /// Layer-selection strategy. Defaults to <see cref="LogitLensLayerSelector.AllLayers"/>.
    /// </summary>
    /// <remarks>
    /// Reuses the logit-lens selector record because SAE hooks fire at the same
    /// <see cref="DotLLM.Core.Diagnostics.HookPoint.PostLayer"/> pipeline location and follow the
    /// same per-layer selection semantics. No need for a parallel SAE-specific selector type.
    /// </remarks>
    public LogitLensLayerSelector Layers { get; init; } = LogitLensLayerSelector.AllLayers;

    /// <summary>
    /// Number of top-magnitude active features to retain per invocation.
    /// Clamped to the SAE dictionary size (<c>d_sae</c>) at encode time. Defaults to 32.
    /// </summary>
    /// <remarks>
    /// "Active" means post-ReLU encoder output. Features outside the top-K are dropped from
    /// the returned <see cref="SaeResult"/>; <see cref="SaeResult.ReconstructionError"/> is
    /// always computed against the original activation (i.e. it reflects the top-K-only
    /// reconstruction loss, NOT the full-dictionary reconstruction loss).
    /// </remarks>
    public int TopK { get; init; } = 32;

    /// <summary>
    /// Optional set of token positions to analyse. When <c>null</c>, all positions seen by
    /// the hook are analysed. Restricting to the final prompt position is the typical
    /// prompt-only inspection workflow.
    /// </summary>
    public IReadOnlyCollection<int>? TokenPositions { get; init; }
}
