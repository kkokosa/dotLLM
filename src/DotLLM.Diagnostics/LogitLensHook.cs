using DotLLM.Core.Diagnostics;

namespace DotLLM.Diagnostics;

/// <summary>
/// Captures residual-stream hidden states at <see cref="HookPoint.PostLayer"/> and projects
/// each through the model's final norm + LM head to produce per-layer token predictions —
/// the "logit lens" technique introduced by nostalgebraist (2020).
/// </summary>
/// <remarks>
/// <para>
/// During the forward pass <see cref="OnActivation"/> clones the hidden state at every
/// selected (layer, position). All projection work is deferred to <see cref="GetResults"/>
/// so that the hot inference path only pays the cost of an array copy per fire — heavier
/// operations (RMSNorm, GEMV, softmax, top-K) run once on retrieval.
/// </para>
/// <para>
/// The hook depends only on the <see cref="ILogitsProjector"/> abstraction in
/// <c>DotLLM.Core.Diagnostics</c>; any model implementing that interface can be analysed.
/// On <c>TransformerModel</c> the projection reuses the same final-norm + LM-head kernel
/// path the model takes for a single-token forward, so the final-layer lens output matches
/// the model's actual logits row.
/// </para>
/// </remarks>
public sealed class LogitLensHook : IInferenceHook
{
    private readonly ILogitsProjector _projector;
    private readonly LogitLensConfig _config;
    private readonly HashSet<int>? _positionFilter;
    private readonly Dictionary<(int Layer, int Position), float[]> _captures = new();

    /// <summary>
    /// Creates a new logit-lens hook bound to <paramref name="projector"/>.
    /// </summary>
    /// <param name="projector">The model whose final norm + LM head should be applied to each captured layer.</param>
    /// <param name="config">Layer/position selection, top-K, full-distribution toggle. Optional — defaults to all layers, top-5.</param>
    public LogitLensHook(ILogitsProjector projector, LogitLensConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(projector);
        _projector = projector;
        _config = config ?? new LogitLensConfig();
        _positionFilter = _config.TokenPositions is null
            ? null
            : new HashSet<int>(_config.TokenPositions);
    }

    /// <inheritdoc/>
    public HookPoint HookPoint => HookPoint.PostLayer;

    /// <summary>
    /// Configuration in effect for this hook.
    /// </summary>
    public LogitLensConfig Config => _config;

    /// <summary>
    /// Returns the set of (layer, position) keys this hook has captured so far.
    /// Exposed primarily for tests and diagnostics.
    /// </summary>
    public IReadOnlyCollection<(int Layer, int Position)> CapturedKeys => _captures.Keys;

    /// <summary>
    /// Number of distinct (layer, position) captures collected so far.
    /// </summary>
    public int CaptureCount => _captures.Count;

    /// <summary>Discards all previously-captured hidden states.</summary>
    public void Clear() => _captures.Clear();

    /// <inheritdoc/>
    public HookResult OnActivation(ReadOnlySpan<float> activation, HookContext context)
    {
        if (context.LayerIndex < 0)
            return HookResult.Continue;

        if (!_config.Layers.ShouldAnalyze(context.LayerIndex))
            return HookResult.Continue;

        if (_positionFilter is not null && !_positionFilter.Contains(context.TokenPosition))
            return HookResult.Continue;

        // Clone the hidden state — the underlying buffer is reused across positions/layers
        // and will be overwritten by downstream computation. Projection is deferred to
        // GetResults() so the hot path pays only the copy cost.
        _captures[(context.LayerIndex, context.TokenPosition)] = activation.ToArray();
        return HookResult.Continue;
    }

    /// <summary>
    /// Projects every captured hidden state through the model's final norm + LM head,
    /// computes softmax, top-K, and Shannon entropy, and returns one
    /// <see cref="LogitLensResult"/> per (layer, position) capture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Heavyweight: allocates one logits buffer per capture, plus the result arrays.
    /// Intended to be called once after the forward pass returns, not per token.
    /// </para>
    /// </remarks>
    /// <returns>Results ordered by (TokenPosition, LayerIndex) ascending.</returns>
    public IReadOnlyList<LogitLensResult> GetResults()
    {
        if (_captures.Count == 0)
            return Array.Empty<LogitLensResult>();

        int vocabSize = _projector.VocabSize;
        int topK = Math.Clamp(_config.TopK, 1, vocabSize);

        // Sort by (position, layer) so the caller gets a stable, intuitive enumeration.
        var ordered = _captures
            .OrderBy(kv => kv.Key.Position)
            .ThenBy(kv => kv.Key.Layer)
            .ToList();

        var logits = new float[vocabSize];
        var results = new List<LogitLensResult>(ordered.Count);
        foreach (var kv in ordered)
        {
            _projector.ProjectToLogits(kv.Value, logits);

            float[]? full = _config.StoreFullProbabilities ? new float[vocabSize] : null;
            LogitLensMath.Softmax(logits, full ?? logits);
            var distribution = full ?? logits;

            float entropy = LogitLensMath.Entropy(distribution);
            LogitLensMath.TopK(distribution, topK, out int[] topTokens, out float[] topProbs);

            results.Add(new LogitLensResult
            {
                LayerIndex = kv.Key.Layer,
                TokenPosition = kv.Key.Position,
                TopKTokens = topTokens,
                TopKProbabilities = topProbs,
                Entropy = entropy,
                FullProbabilities = full,
            });
        }

        return results;
    }
}
