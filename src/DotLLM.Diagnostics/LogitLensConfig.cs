namespace DotLLM.Diagnostics;

/// <summary>
/// Configuration for <see cref="LogitLensHook"/> — which layers to analyse, how many
/// top tokens to retain, and whether to store full probability distributions.
/// </summary>
public sealed class LogitLensConfig
{
    /// <summary>
    /// Layer-selection strategy. Defaults to <see cref="LogitLensLayerSelector.AllLayers"/>.
    /// </summary>
    public LogitLensLayerSelector Layers { get; init; } = LogitLensLayerSelector.AllLayers;

    /// <summary>
    /// Number of top-probability tokens to retain per layer. Clamped to vocabulary size at
    /// projection time. Defaults to 5.
    /// </summary>
    public int TopK { get; init; } = 5;

    /// <summary>
    /// When <c>true</c>, the full <c>vocabSize</c>-length softmax distribution is retained
    /// in <see cref="LogitLensResult.FullProbabilities"/>. Enables exact rank computation
    /// for tokens outside the top-K, at the cost of one allocation per (layer, position).
    /// Defaults to <c>false</c>.
    /// </summary>
    public bool StoreFullProbabilities { get; init; }

    /// <summary>
    /// Optional set of token positions to analyse. When <c>null</c>, all positions seen by
    /// the hook are analysed. For prompt-only logit lens, callers typically restrict to the
    /// final prompt position.
    /// </summary>
    public IReadOnlyCollection<int>? TokenPositions { get; init; }
}

/// <summary>
/// Selects which transformer layers contribute logit-lens results.
/// </summary>
public abstract record LogitLensLayerSelector
{
    private LogitLensLayerSelector() { }

    /// <summary>Returns <c>true</c> when <paramref name="layer"/> should be analysed.</summary>
    /// <param name="layer">Zero-based transformer layer index.</param>
    public abstract bool ShouldAnalyze(int layer);

    /// <summary>Analyse every layer.</summary>
    public static LogitLensLayerSelector AllLayers { get; } = new AllSelector();

    /// <summary>Analyse every <paramref name="n"/>-th layer (layers 0, n, 2n, …).</summary>
    /// <param name="n">Stride; must be at least 1.</param>
    public static LogitLensLayerSelector EveryNth(int n)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(n, 1);
        return new EveryNthSelector(n);
    }

    /// <summary>Analyse only the specified layer indices.</summary>
    /// <param name="layers">Layer indices to retain.</param>
    public static LogitLensLayerSelector Specific(IEnumerable<int> layers)
    {
        ArgumentNullException.ThrowIfNull(layers);
        return new SpecificSelector(new HashSet<int>(layers));
    }

    /// <summary>Analyses every layer.</summary>
    public sealed record AllSelector : LogitLensLayerSelector
    {
        /// <inheritdoc/>
        public override bool ShouldAnalyze(int layer) => true;
    }

    /// <summary>Analyses every n-th layer.</summary>
    /// <param name="N">Stride between selected layers.</param>
    public sealed record EveryNthSelector(int N) : LogitLensLayerSelector
    {
        /// <inheritdoc/>
        public override bool ShouldAnalyze(int layer) => layer >= 0 && layer % N == 0;
    }

    /// <summary>Analyses only the specified layer indices.</summary>
    /// <param name="Layers">Set of layers to retain.</param>
    public sealed record SpecificSelector(HashSet<int> Layers) : LogitLensLayerSelector
    {
        /// <inheritdoc/>
        public override bool ShouldAnalyze(int layer) => Layers.Contains(layer);
    }
}
