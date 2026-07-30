using System.Collections.ObjectModel;
using DotLLM.Core.Diagnostics;

namespace DotLLM.Diagnostics;

/// <summary>
/// Built-in <see cref="IInferenceHook"/> that captures activation snapshots at a configured
/// <see cref="HookPoint"/>, optionally filtered to a set of layers and token positions.
/// </summary>
/// <remarks>
/// <para>
/// Each captured activation is copied into a freshly-allocated <see cref="float"/>[] and stored
/// in <see cref="Captures"/>, keyed by <see cref="CaptureKey"/>. The captured array is independent
/// of the underlying inference scratch buffers, so it remains valid after the forward pass returns.
/// </para>
/// <para>
/// This is a research/diagnostic tool — capture allocates per-fire and is not appropriate for
/// production serving. Use it under explicit interpretability sessions.
/// </para>
/// </remarks>
public sealed class CaptureHook : IInferenceHook
{
    private readonly HashSet<int>? _layerFilter;
    private readonly HashSet<int>? _positionFilter;
    private readonly Dictionary<CaptureKey, float[]> _captures = new();

    // Live read-only view over _captures, cached once. ReadOnlyDictionary wraps the same backing
    // store, so it reflects later adds and Clear() without re-allocation while preventing callers
    // from mutating the captures through the public surface.
    private readonly ReadOnlyDictionary<CaptureKey, float[]> _capturesView;

    /// <summary>
    /// Creates a capture hook for the given <paramref name="point"/>.
    /// </summary>
    /// <param name="point">The pipeline location to capture at.</param>
    /// <param name="layers">
    /// Optional set of layer indices to capture. <c>null</c> means "all layers".
    /// Ignored for non-layer points (<see cref="HookPoint.PostEmbedding"/>,
    /// <see cref="HookPoint.PreLmHead"/>, <see cref="HookPoint.PostLmHead"/>).
    /// </param>
    /// <param name="tokenPositions">
    /// Optional set of token positions to capture. <c>null</c> means "all positions".
    /// </param>
    public CaptureHook(
        HookPoint point,
        IEnumerable<int>? layers = null,
        IEnumerable<int>? tokenPositions = null)
    {
        HookPoint = point;
        _layerFilter = layers is null ? null : new HashSet<int>(layers);
        _positionFilter = tokenPositions is null ? null : new HashSet<int>(tokenPositions);
        _capturesView = new ReadOnlyDictionary<CaptureKey, float[]>(_captures);
    }

    /// <inheritdoc/>
    public HookPoint HookPoint { get; }

    /// <summary>
    /// All captured activations keyed by (layer, position). Layer is -1 for non-layer points.
    /// Returned as an immutable view; callers cannot mutate the underlying capture store.
    /// </summary>
    public IReadOnlyDictionary<CaptureKey, float[]> Captures => _capturesView;

    /// <summary>Clears all previously-captured activations.</summary>
    public void Clear() => _captures.Clear();

    /// <inheritdoc/>
    public HookResult OnActivation(ReadOnlySpan<float> activation, HookContext context)
    {
        if (_layerFilter is not null && context.LayerIndex >= 0 && !_layerFilter.Contains(context.LayerIndex))
            return HookResult.Continue;

        if (_positionFilter is not null && !_positionFilter.Contains(context.TokenPosition))
            return HookResult.Continue;

        _captures[new CaptureKey(context.LayerIndex, context.TokenPosition)] = activation.ToArray();
        return HookResult.Continue;
    }

    /// <summary>
    /// Key identifying a single captured activation by transformer layer and token position.
    /// </summary>
    /// <param name="LayerIndex">Layer index (-1 for non-layer hook points).</param>
    /// <param name="TokenPosition">Token position within the sequence.</param>
    public readonly record struct CaptureKey(int LayerIndex, int TokenPosition);
}
