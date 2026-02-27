using DotLLM.Core.Configuration;

namespace DotLLM.Core.PositionEncoding;

/// <summary>
/// Configuration for Rotary Position Embeddings (RoPE).
/// </summary>
/// <param name="Theta">Base frequency. Default 10000.0, Llama 3 uses 500000.0.</param>
/// <param name="DimensionCount">Number of dimensions for the rotation.</param>
/// <param name="ScalingType">Context-length scaling strategy.</param>
/// <param name="ScalingFactor">Scaling factor for Linear/NTK methods.</param>
/// <param name="OrigMaxSeqLen">Original max sequence length before scaling.</param>
/// <param name="AttnFactor">YaRN attention factor.</param>
/// <param name="BetaFast">YaRN beta-fast parameter.</param>
/// <param name="BetaSlow">YaRN beta-slow parameter.</param>
public readonly record struct RoPEConfig(
    float Theta = 10000.0f,
    int DimensionCount = 0,
    RoPEScalingType ScalingType = RoPEScalingType.None,
    float ScalingFactor = 1.0f,
    int OrigMaxSeqLen = 0,
    float AttnFactor = 1.0f,
    float BetaFast = 32.0f,
    float BetaSlow = 1.0f);
