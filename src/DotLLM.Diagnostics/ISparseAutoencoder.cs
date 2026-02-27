namespace DotLLM.Diagnostics;

/// <summary>
/// Sparse Autoencoder (SAE) for mechanistic interpretability.
/// Decomposes activations into sparse, interpretable features.
/// </summary>
public interface ISparseAutoencoder
{
    /// <summary>
    /// Encodes an activation vector into a sparse feature representation.
    /// </summary>
    /// <param name="activation">Input activation vector.</param>
    /// <returns>Sparse feature indices and their activation values.</returns>
    (int[] FeatureIndices, float[] FeatureValues) Encode(ReadOnlySpan<float> activation);

    /// <summary>
    /// Decodes sparse features back into an activation vector.
    /// </summary>
    /// <param name="featureIndices">Active feature indices.</param>
    /// <param name="featureValues">Activation values for the active features.</param>
    /// <param name="output">Output buffer for the reconstructed activation.</param>
    void Decode(ReadOnlySpan<int> featureIndices, ReadOnlySpan<float> featureValues, Span<float> output);

    /// <summary>Total number of features in the dictionary.</summary>
    int FeatureCount { get; }
}
