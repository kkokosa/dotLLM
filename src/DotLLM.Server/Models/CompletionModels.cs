using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible raw completion request (no chat template).
/// </summary>
public sealed record CompletionRequest
{
    [JsonPropertyName("prompt")]
    public required string Prompt { get; init; }

    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }

    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }

    [JsonPropertyName("stream")]
    public bool Stream { get; init; }

    [JsonPropertyName("stop")]
    public JsonElement? Stop { get; init; }

    [JsonPropertyName("seed")]
    public int? Seed { get; init; }

    [JsonPropertyName("repetition_penalty")]
    public float? RepetitionPenalty { get; init; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; init; }

    [JsonPropertyName("min_p")]
    public float? MinP { get; init; }

    [JsonPropertyName("response_format")]
    public JsonElement? ResponseFormat { get; init; }
}

/// <summary>
/// OpenAI-compatible raw completion response.
/// </summary>
public sealed record CompletionResponse
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "text_completion";

    [JsonPropertyName("created")]
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("choices")]
    public required CompletionChoiceDto[] Choices { get; init; }

    [JsonPropertyName("usage")]
    public required UsageDto Usage { get; init; }
}

/// <summary>
/// A single choice in a raw completion response.
/// </summary>
public sealed record CompletionChoiceDto
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("text")]
    public required string Text { get; init; }

    [JsonPropertyName("finish_reason")]
    public required string FinishReason { get; init; }
}

/// <summary>
/// Streaming raw completion chunk.
/// </summary>
public sealed record CompletionChunk
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "text_completion";

    [JsonPropertyName("created")]
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("choices")]
    public required CompletionChunkChoiceDto[] Choices { get; init; }
}

/// <summary>
/// A single choice in a streaming completion chunk.
/// </summary>
public sealed record CompletionChunkChoiceDto
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("text")]
    public required string Text { get; init; }

    [JsonPropertyName("finish_reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? FinishReason { get; init; }
}
