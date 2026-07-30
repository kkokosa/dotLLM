namespace DotLLM.Diagnostics;

/// <summary>
/// Per-layer result of a logit-lens analysis at a single token position.
/// </summary>
/// <remarks>
/// <para>
/// Produced by <see cref="LogitLensHook.GetResults"/>. Holds the top-K most-probable tokens
/// the model would have predicted if decoding stopped at this layer, the entropy of the
/// full distribution, and (optionally — when configured to store the full distribution)
/// the rank of the model's actual final-layer top-1 token within this layer's predictions.
/// </para>
/// <para>
/// The <see cref="FullProbabilities"/> array is non-null only when
/// <see cref="LogitLensConfig.StoreFullProbabilities"/> is set; otherwise consumers can use
/// <see cref="TopKTokens"/> / <see cref="TopKProbabilities"/> only.
/// </para>
/// </remarks>
public sealed class LogitLensResult
{
    /// <summary>Transformer layer index this result is for (0-based).</summary>
    public int LayerIndex { get; init; }

    /// <summary>Token position within the sequence (0-based).</summary>
    public int TokenPosition { get; init; }

    /// <summary>
    /// Top-K token ids in descending probability order. Length equals
    /// <see cref="LogitLensConfig.TopK"/> capped at vocabulary size.
    /// </summary>
    public required int[] TopKTokens { get; init; }

    /// <summary>
    /// Probabilities (softmax output) for <see cref="TopKTokens"/>, parallel array.
    /// </summary>
    public required float[] TopKProbabilities { get; init; }

    /// <summary>
    /// Shannon entropy of the full probability distribution, in nats.
    /// Low entropy = confident; high entropy ≈ <c>log(vocab)</c> ≈ uniform.
    /// </summary>
    public required float Entropy { get; init; }

    /// <summary>
    /// Full softmax probability distribution over the vocabulary, or <c>null</c> when
    /// <see cref="LogitLensConfig.StoreFullProbabilities"/> is <c>false</c>.
    /// </summary>
    public float[]? FullProbabilities { get; init; }
}
