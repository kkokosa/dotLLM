using DotLLM.Core.Tensors;

namespace DotLLM.Core.Attention;

/// <summary>
/// Compute strategy for the attention kernel (naive, flash, paged).
/// Decoupled from the attention mechanism (GQA, MLA) to allow independent selection.
/// </summary>
public interface IAttentionStrategy
{
    /// <summary>
    /// Computes scaled dot-product attention.
    /// </summary>
    /// <param name="q">Query tensor.</param>
    /// <param name="k">Key tensor.</param>
    /// <param name="v">Value tensor.</param>
    /// <param name="mask">Attention mask. Null for no masking.</param>
    /// <param name="scale">Scaling factor (typically 1/sqrt(head_dim)).</param>
    /// <returns>Attention output tensor.</returns>
    ITensor ComputeAttention(ITensor q, ITensor k, ITensor v, ITensor? mask, float scale);

    /// <summary>Whether this strategy supports paged KV-cache.</summary>
    bool SupportsPagedKvCache { get; }

    /// <summary>Minimum GPU compute capability required. Null if CPU-compatible.</summary>
    int? RequiredComputeCapability { get; }
}
