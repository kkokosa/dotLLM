using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible chat completion request.
/// </summary>
public sealed record ChatCompletionRequest
{
    [JsonPropertyName("messages")]
    public required ChatMessageDto[] Messages { get; init; }

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

    [JsonPropertyName("tools")]
    public ToolDefinitionDto[]? Tools { get; init; }

    [JsonPropertyName("tool_choice")]
    public JsonElement? ToolChoice { get; init; }

    [JsonPropertyName("response_format")]
    public JsonElement? ResponseFormat { get; init; }

    [JsonPropertyName("seed")]
    public int? Seed { get; init; }

    [JsonPropertyName("frequency_penalty")]
    public float? FrequencyPenalty { get; init; }

    [JsonPropertyName("presence_penalty")]
    public float? PresencePenalty { get; init; }

    [JsonPropertyName("repetition_penalty")]
    public float? RepetitionPenalty { get; init; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; init; }

    [JsonPropertyName("min_p")]
    public float? MinP { get; init; }

    [JsonPropertyName("n")]
    public int N { get; init; } = 1;
}

/// <summary>
/// A chat message in the OpenAI format.
/// </summary>
public sealed record ChatMessageDto
{
    [JsonPropertyName("role")]
    public required string Role { get; init; }

    [JsonPropertyName("content")]
    public string? Content { get; init; }

    [JsonPropertyName("tool_calls")]
    public ToolCallDto[]? ToolCalls { get; init; }

    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; init; }
}

/// <summary>
/// Tool definition in the OpenAI format.
/// </summary>
public sealed record ToolDefinitionDto
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "function";

    [JsonPropertyName("function")]
    public required ToolFunctionDto Function { get; init; }
}

/// <summary>
/// Function definition within a tool.
/// </summary>
public sealed record ToolFunctionDto
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("description")]
    public string? Description { get; init; }

    [JsonPropertyName("parameters")]
    public JsonElement? Parameters { get; init; }
}

/// <summary>
/// A tool call made by the assistant.
/// </summary>
public sealed record ToolCallDto
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("type")]
    public string Type { get; init; } = "function";

    [JsonPropertyName("function")]
    public required ToolCallFunctionDto Function { get; init; }
}

/// <summary>
/// Function invocation within a tool call.
/// </summary>
public sealed record ToolCallFunctionDto
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("arguments")]
    public required string Arguments { get; init; }
}
