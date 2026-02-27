using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Core.PositionEncoding;

/// <summary>
/// Applies positional information to query and key projections.
/// Implementations: RoPE, ALiBi, absolute embeddings.
/// </summary>
public interface IPositionEncoding
{
    /// <summary>
    /// Applies position encoding to query and key tensors.
    /// </summary>
    /// <param name="q">Query tensor to encode.</param>
    /// <param name="k">Key tensor to encode.</param>
    /// <param name="positions">Position indices for each token in the sequence.</param>
    /// <returns>Position-encoded query and key tensors.</returns>
    (ITensor Q, ITensor K) Apply(ITensor q, ITensor k, ReadOnlySpan<int> positions);

    /// <summary>
    /// Precomputes frequency/position tables for the given sequence length.
    /// Call once after model loading; cached until <see cref="InvalidateCache"/> is called.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length to precompute for.</param>
    /// <param name="config">Model configuration for dimension sizing.</param>
    void PrecomputeTables(int maxSeqLen, ModelConfig config);

    /// <summary>
    /// Invalidates cached precomputed tables. Call when parameters change at runtime
    /// (e.g., dynamic NTK scaling adjusts theta).
    /// </summary>
    void InvalidateCache();
}
