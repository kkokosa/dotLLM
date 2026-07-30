namespace DotLLM.Diagnostics;

/// <summary>
/// Sparse Autoencoder (SAE) for mechanistic interpretability.
/// Decomposes activations into sparse, interpretable features.
/// </summary>
/// <remarks>
/// <para>
/// SAEs are trained offline with an L1 / top-K sparsity penalty so that a wide overcomplete
/// dictionary (<c>d_sae</c> ≫ <c>d_in</c>) decomposes a dense residual-stream activation into
/// a small number of active features. At inference time the encoder/decoder weights are
/// loaded as fixed parameters and the typical workflow is encode → inspect top-K → optionally
/// modify → decode.
/// </para>
/// <para>
/// Implementations should hold weight tensors in unmanaged memory (per project rules) but
/// return managed arrays from <see cref="Encode"/>: the sparse output is bounded in size
/// (top-K) and is consumed by analysis code that operates in managed land. Per-call
/// allocations on the encode path are acceptable for diagnostics — the hook only fires when
/// explicitly registered.
/// </para>
/// </remarks>
public interface ISparseAutoencoder
{
    /// <summary>
    /// Encodes an activation vector into a sparse feature representation.
    /// </summary>
    /// <param name="activation">Input activation vector. Length must equal <see cref="HiddenSize"/>.</param>
    /// <returns>
    /// Top-K active feature indices (descending magnitude) and their post-ReLU activation
    /// values. K is implementation-defined (typically a configurable property on the
    /// concrete SAE).
    /// </returns>
    (int[] FeatureIndices, float[] FeatureValues) Encode(ReadOnlySpan<float> activation);

    /// <summary>
    /// Decodes sparse features back into an activation vector.
    /// </summary>
    /// <param name="featureIndices">Active feature indices.</param>
    /// <param name="featureValues">Activation values for the active features.</param>
    /// <param name="output">Output buffer for the reconstructed activation. Length must equal <see cref="HiddenSize"/>.</param>
    void Decode(ReadOnlySpan<int> featureIndices, ReadOnlySpan<float> featureValues, Span<float> output);

    /// <summary>Total number of features in the dictionary (<c>d_sae</c>).</summary>
    int FeatureCount { get; }

    /// <summary>
    /// Input/output activation dimensionality (<c>d_in</c>) — the residual-stream width the
    /// SAE was trained on. Encode inputs and Decode outputs must match this size.
    /// </summary>
    int HiddenSize { get; }
}
