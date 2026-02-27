namespace DotLLM.Core.Models;

/// <summary>
/// Configuration for Multi-head Latent Attention (MLA), used by DeepSeek models.
/// </summary>
/// <param name="LatentDim">Dimension of the latent compressed KV representation (e.g., 512).</param>
/// <param name="RopeDim">Dimension allocated for RoPE within MLA (e.g., 64).</param>
public readonly record struct MlaConfig(int LatentDim, int RopeDim);
