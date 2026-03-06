using DotLLM.Core.Sampling;

namespace DotLLM.Core.Configuration;

/// <summary>
/// Options controlling inference behavior: sampling parameters, stop conditions, and limits.
/// </summary>
public record InferenceOptions
{
    /// <summary>Temperature for sampling. 0 = greedy.</summary>
    public float Temperature { get; init; } = 0.7f;

    /// <summary>Top-K sampling. 0 = disabled.</summary>
    public int TopK { get; init; }

    /// <summary>Top-P (nucleus) sampling threshold.</summary>
    public float TopP { get; init; } = 1.0f;

    /// <summary>Min-P sampling threshold. 0 = disabled.</summary>
    public float MinP { get; init; }

    /// <summary>Repetition penalty factor. 1.0 = disabled.</summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>Number of recent tokens for repetition penalty lookback. 0 = full history.</summary>
    public int RepetitionPenaltyWindow { get; init; }

    /// <summary>Maximum number of tokens to generate.</summary>
    public int MaxTokens { get; init; } = 2048;

    /// <summary>Random seed for reproducible sampling. Null = non-deterministic.</summary>
    public int? Seed { get; init; }

    /// <summary>Stop sequences that terminate generation.</summary>
    public IReadOnlyList<string> StopSequences { get; init; } = [];

    /// <summary>
    /// Explicit sampler steps composing the sampling pipeline.
    /// When set, these steps are used instead of building from the flat properties
    /// (Temperature, TopK, TopP, MinP). Steps are applied in order.
    /// </summary>
    public IReadOnlyList<ISamplerStep>? SamplerSteps { get; init; }

    /// <summary>
    /// Explicit logit processors (e.g., repetition penalty).
    /// When set, used instead of building from RepetitionPenalty.
    /// </summary>
    public IReadOnlyList<ILogitProcessor>? LogitProcessors { get; init; }

    /// <summary>
    /// Explicit stop conditions. When set, used instead of the default
    /// (EOS + MaxTokens + StopSequences). The caller controls the full set.
    /// </summary>
    public IReadOnlyList<IStopCondition>? StopConditions { get; init; }

    /// <summary>CPU threading configuration for parallel inference. Default: auto (all cores).</summary>
    public ThreadingConfig Threading { get; init; } = ThreadingConfig.Auto;
}
