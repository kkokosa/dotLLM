using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;

namespace DotLLM.Core.Attention;

/// <summary>
/// Attention mechanism that handles the full Q/K/V → output pipeline,
/// including position encoding and KV-cache management. Separates mechanism (GQA, MLA)
/// from compute strategy (naive, flash, paged).
/// </summary>
public interface IAttentionMechanism
{
    /// <summary>
    /// Computes attention output for the given Q/K/V projections.
    /// </summary>
    /// <param name="q">Query projections.</param>
    /// <param name="k">Key projections.</param>
    /// <param name="v">Value projections.</param>
    /// <param name="positionEncoding">Position encoding to apply.</param>
    /// <param name="kvCache">KV-cache to update and read from.</param>
    /// <param name="mask">Attention mask (causal or custom).</param>
    /// <param name="hook">Optional inference hook for activation capture. Null when disabled.</param>
    /// <returns>Attention output tensor.</returns>
    ITensor Forward(ITensor q, ITensor k, ITensor v,
                    IPositionEncoding positionEncoding,
                    IKvCache kvCache, ITensor? mask,
                    IInferenceHook? hook);

    /// <summary>
    /// Creates a new KV-cache sized for the given model configuration.
    /// </summary>
    /// <param name="config">Model configuration determining cache dimensions.</param>
    /// <returns>A newly allocated, empty KV-cache.</returns>
    IKvCache CreateKvCache(ModelConfig config);
}
