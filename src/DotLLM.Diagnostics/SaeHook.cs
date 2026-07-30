using DotLLM.Core.Diagnostics;

namespace DotLLM.Diagnostics;

/// <summary>
/// Captures residual-stream hidden states at <see cref="HookPoint.PostLayer"/> and runs them
/// through a pre-trained <see cref="SparseAutoencoder"/> to surface the top-K active features
/// (post-ReLU encoder output) per (layer, position).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the <see cref="LogitLensHook"/> defer pattern: <see cref="OnActivation"/> clones the
/// hidden state on the hot path and returns <see cref="HookResult.Continue"/> unchanged; the
/// heavier encode + top-K + reconstruction-error work is deferred to <see cref="GetResults"/>.
/// </para>
/// <para>
/// Typically registered at the <see cref="HookPoint.PostLayer"/> site for the specific layer the
/// SAE was trained on. The <see cref="SaeConfig.Layers"/> selector restricts which layer indices
/// the hook accepts at <see cref="OnActivation"/> — out-of-selector layers are dropped before any
/// allocation.
/// </para>
/// <para>
/// This hook is read-only (always returns <see cref="HookResult.Continue"/>). Steering / ablation
/// via <see cref="HookResult.Replace(System.ReadOnlySpan{float})"/> is intentionally out of scope
/// for the initial integration — see the SAE integration follow-up issue.
/// </para>
/// </remarks>
public sealed class SaeHook : IInferenceHook
{
    private readonly SparseAutoencoder _sae;
    private readonly SaeConfig _config;
    private readonly HashSet<int>? _positionFilter;
    private readonly Dictionary<(int Layer, int Position), float[]> _captures = new();

    /// <summary>
    /// Creates an SAE hook bound to <paramref name="sae"/>.
    /// </summary>
    /// <param name="sae">The pre-trained sparse autoencoder to apply. The hook does not take ownership.</param>
    /// <param name="config">Layer/position selection and top-K. Optional — defaults to all layers, top-32.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="sae"/> is null.</exception>
    /// <remarks>
    /// The hook writes <see cref="SaeConfig.TopK"/> (clamped to the SAE's dictionary size) into
    /// <see cref="SparseAutoencoder.TopK"/> at construction so subsequent encode calls return
    /// the configured number of features. If the same SAE is shared across multiple hooks with
    /// different TopK values, the latest constructed hook wins — keep TopK in one place.
    /// </remarks>
    public SaeHook(SparseAutoencoder sae, SaeConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(sae);
        _sae = sae;
        _config = config ?? new SaeConfig();
        _positionFilter = _config.TokenPositions is null
            ? null
            : new HashSet<int>(_config.TokenPositions);

        _sae.TopK = Math.Clamp(_config.TopK, 1, _sae.FeatureCount);
    }

    /// <inheritdoc/>
    public HookPoint HookPoint => HookPoint.PostLayer;

    /// <summary>Configuration in effect for this hook.</summary>
    public SaeConfig Config => _config;

    /// <summary>The SAE this hook is bound to.</summary>
    public SparseAutoencoder Sae => _sae;

    /// <summary>
    /// Returns the set of (layer, position) keys this hook has captured so far. Exposed for
    /// tests and diagnostics.
    /// </summary>
    public IReadOnlyCollection<(int Layer, int Position)> CapturedKeys => _captures.Keys;

    /// <summary>Number of distinct (layer, position) captures collected so far.</summary>
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

        if (activation.Length != _sae.HiddenSize)
            throw new InvalidOperationException(
                $"SaeHook received an activation of length {activation.Length} at layer {context.LayerIndex}, " +
                $"but the bound SAE has HiddenSize={_sae.HiddenSize}. SAEs are layer-dimension-specific.");

        // Clone the hidden state — the underlying buffer is reused across positions/layers
        // and will be overwritten by downstream computation. Encode/top-K/L2 are deferred
        // to GetResults() so the hot path pays only the copy cost.
        _captures[(context.LayerIndex, context.TokenPosition)] = activation.ToArray();
        return HookResult.Continue;
    }

    /// <summary>
    /// Encodes every captured hidden state through the bound SAE, extracts the top-K active
    /// features, and computes the L2 reconstruction error of decoding those K features alone
    /// back to the original activation.
    /// </summary>
    /// <returns>One <see cref="SaeResult"/> per (layer, position) capture, ordered by (position, layer) ascending.</returns>
    public IReadOnlyList<SaeResult> GetResults()
    {
        if (_captures.Count == 0)
            return Array.Empty<SaeResult>();

        var ordered = _captures
            .OrderBy(kv => kv.Key.Position)
            .ThenBy(kv => kv.Key.Layer)
            .ToList();

        var results = new List<SaeResult>(ordered.Count);
        var reconstruction = new float[_sae.HiddenSize];

        foreach (var kv in ordered)
        {
            var (indices, values, activeCount) = _sae.EncodeWithDetails(kv.Value);
            _sae.Decode(indices, values, reconstruction);
            float l2 = SaeMath.L2Distance(kv.Value, reconstruction);

            results.Add(new SaeResult
            {
                LayerIndex = kv.Key.Layer,
                TokenPosition = kv.Key.Position,
                FeatureIndices = indices,
                FeatureMagnitudes = values,
                ReconstructionError = l2,
                ActiveFeatureCount = activeCount,
            });
        }

        return results;
    }
}
