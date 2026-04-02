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
    public ToolCallDto[]? ToolCalls { get; init; }
}
