using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible streaming chat completion chunk (SSE).
/// </summary>
public sealed record ChatCompletionChunk
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "chat.completion.chunk";

    [JsonPropertyName("created")]
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("choices")]
    public required ChatChunkChoiceDto[] Choices { get; init; }

    [JsonPropertyName("usage")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public UsageDto? Usage { get; init; }

    [JsonPropertyName("timings")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public TimingsDto? Timings { get; init; }

}

/// <summary>
/// Inference timing details (dotLLM extension, not in OpenAI spec).
/// </summary>
public sealed record TimingsDto
{
    [JsonPropertyName("prefill_time_ms")]
    public double PrefillTimeMs { get; init; }

    [JsonPropertyName("decode_time_ms")]
    public double DecodeTimeMs { get; init; }

    [JsonPropertyName("sampling_time_ms")]
    public double SamplingTimeMs { get; init; }

    [JsonPropertyName("prefill_tokens_per_sec")]
    public double PrefillTokensPerSec { get; init; }

    [JsonPropertyName("decode_tokens_per_sec")]
    public double DecodeTokensPerSec { get; init; }

    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    [JsonPropertyName("generated_tokens")]
    public int GeneratedTokens { get; init; }

    [JsonPropertyName("cached_tokens")]
    public int CachedTokens { get; init; }

    [JsonPropertyName("speculative_draft_tokens")]
    public int SpeculativeDraftTokens { get; init; }

    [JsonPropertyName("speculative_accepted_tokens")]
    public int SpeculativeAcceptedTokens { get; init; }

    [JsonPropertyName("speculative_acceptance_rate")]
    public float SpeculativeAcceptanceRate { get; init; }
}

/// <summary>
/// A single choice in a streaming chunk.
/// </summary>
public sealed record ChatChunkChoiceDto
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("delta")]
    public required ChatDeltaDto Delta { get; init; }

    [JsonPropertyName("logprobs")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public LogprobsDto? Logprobs { get; init; }

    [JsonPropertyName("finish_reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FinishReason { get; init; }
}

/// <summary>
/// Incremental message delta in a streaming chunk.
/// </summary>
public sealed record ChatDeltaDto
{
    [JsonPropertyName("role")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Role { get; init; }

    [JsonPropertyName("content")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Content { get; init; }

    [JsonPropertyName("tool_calls")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ToolCallDeltaDto[]? ToolCalls { get; init; }
}

/// <summary>
/// Streaming-specific <c>delta.tool_calls[]</c> element. All fields except
/// <c>index</c> are optional so partial deltas (e.g. an arguments-only fragment)
/// serialise to the OpenAI SSE shape:
/// <c>{"index":0,"function":{"arguments":"..."}}</c>.
/// </summary>
public sealed record ToolCallDeltaDto
{
    /// <summary>Zero-based parallel-call index. Required; increments for each new call within a stream.</summary>
    [JsonPropertyName("index")]
    public required int Index { get; init; }

    /// <summary>Server-generated call id. Present on the first fragment of a call only.</summary>
    [JsonPropertyName("id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Id { get; init; }

    /// <summary>Always <c>"function"</c> for first fragments; <c>null</c> on subsequent argument-only fragments.</summary>
    [JsonPropertyName("type")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Type { get; init; }

    /// <summary>Function delta. <c>null</c> only for opening fragments that carry no name/args yet.</summary>
    [JsonPropertyName("function")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ToolCallFunctionDeltaDto? Function { get; init; }
}

/// <summary>
/// Streaming-specific function-call delta. <c>Name</c> is present on the first
/// fragment only; <c>Arguments</c> carries the next slice of the arguments JSON
/// string (consumers concatenate slices to reconstruct the full arguments object).
/// </summary>
public sealed record ToolCallFunctionDeltaDto
{
    /// <summary>Function name. Present on the first fragment of a call only.</summary>
    [JsonPropertyName("name")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Name { get; init; }

    /// <summary>Incremental slice of the arguments JSON string. <c>null</c> if this fragment carries no argument text.</summary>
    [JsonPropertyName("arguments")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Arguments { get; init; }
}
