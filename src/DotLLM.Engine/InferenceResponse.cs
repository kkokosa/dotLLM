namespace DotLLM.Engine;

/// <summary>
/// Result of a completed inference request.
/// </summary>
public record InferenceResponse
{
    /// <summary>Generated token IDs.</summary>
    public required int[] GeneratedTokenIds { get; init; }

    /// <summary>Decoded output text.</summary>
    public required string Text { get; init; }

    /// <summary>Reason generation stopped.</summary>
    public required FinishReason FinishReason { get; init; }

    /// <summary>Number of prompt tokens processed.</summary>
    public int PromptTokenCount { get; init; }

    /// <summary>Number of tokens generated.</summary>
    public int GeneratedTokenCount { get; init; }
}
