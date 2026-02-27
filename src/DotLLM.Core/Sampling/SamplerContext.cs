namespace DotLLM.Core.Sampling;

/// <summary>
/// Context passed to <see cref="ISamplerStep.Apply"/> providing request-level information.
/// </summary>
/// <param name="Temperature">Current temperature setting.</param>
/// <param name="TopK">Top-K value. 0 = disabled.</param>
/// <param name="TopP">Top-P (nucleus) threshold.</param>
/// <param name="MinP">Min-P threshold. 0 = disabled.</param>
/// <param name="Seed">Random seed for deterministic sampling. Null = non-deterministic.</param>
public readonly record struct SamplerContext(
    float Temperature,
    int TopK,
    float TopP,
    float MinP,
    int? Seed);
