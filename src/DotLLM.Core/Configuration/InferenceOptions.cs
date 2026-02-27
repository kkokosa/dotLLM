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

    /// <summary>Maximum number of tokens to generate.</summary>
    public int MaxTokens { get; init; } = 2048;

    /// <summary>Random seed for reproducible sampling. Null = non-deterministic.</summary>
    public int? Seed { get; init; }

    /// <summary>Stop sequences that terminate generation.</summary>
    public IReadOnlyList<string> StopSequences { get; init; } = [];
}
