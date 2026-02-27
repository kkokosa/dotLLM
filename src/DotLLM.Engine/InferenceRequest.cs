using DotLLM.Core.Configuration;
using DotLLM.Core.Constraints;

namespace DotLLM.Engine;

/// <summary>
/// An inference request submitted to the scheduler.
/// </summary>
public record InferenceRequest
{
    /// <summary>Input token IDs (prompt).</summary>
    public required int[] TokenIds { get; init; }

    /// <summary>Inference options (temperature, top-K, etc.).</summary>
    public InferenceOptions Options { get; init; } = new();

    /// <summary>Optional decoding constraint for structured output.</summary>
    public IDecodingConstraint? Constraint { get; init; }

    /// <summary>Request priority for scheduling.</summary>
    public RequestPriority Priority { get; init; } = RequestPriority.Normal;

    /// <summary>Optional LoRA adapter ID to use for this request.</summary>
    public string? AdapterId { get; init; }
}
