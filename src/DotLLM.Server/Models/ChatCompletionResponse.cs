using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible chat completion response (non-streaming).
/// </summary>
public sealed record ChatCompletionResponse
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "chat.completion";

    [JsonPropertyName("created")]
    public long Created { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("choices")]
    public required ChatChoiceDto[] Choices { get; init; }

    [JsonPropertyName("usage")]
    public required UsageDto Usage { get; init; }
}

/// <summary>
/// A single choice in a chat completion response.
/// </summary>
public sealed record ChatChoiceDto
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("message")]
    public required ChatMessageDto Message { get; init; }

    [JsonPropertyName("finish_reason")]
    public required string FinishReason { get; init; }
}

/// <summary>
/// Token usage information.
/// </summary>
public sealed record UsageDto
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get; init; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; init; }
}
