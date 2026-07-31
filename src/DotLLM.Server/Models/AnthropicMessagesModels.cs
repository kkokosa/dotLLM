using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

// ---------------------------------------------------------------------------
// Anthropic Messages API DTOs (POST /v1/messages, /v1/messages/count_tokens).
//
// These mirror the public Anthropic Messages API request/response shapes so
// that clients written for the `anthropic` SDKs can talk to dotLLM unchanged.
// The translation onto dotLLM engine types lives in AnthropicConverter; these
// records are pure serialization contracts.
//
// Reference: https://docs.anthropic.com/en/api/messages
// ---------------------------------------------------------------------------

/// <summary>
/// Anthropic-compatible <c>POST /v1/messages</c> request. Also used (with
/// <c>max_tokens</c>/<c>stream</c> ignored) for <c>POST /v1/messages/count_tokens</c>.
/// </summary>
public sealed record AnthropicMessagesRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("messages")]
    public required AnthropicMessageDto[] Messages { get; init; }

    /// <summary>
    /// Top-level system prompt. Either a string or an array of text content
    /// blocks (<c>{"type":"text","text":"..."}</c>). Parsed by the converter.
    /// </summary>
    [JsonPropertyName("system")]
    public JsonElement? System { get; init; }

    /// <summary>Maximum tokens to generate. Required by the Anthropic spec.</summary>
    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }

    [JsonPropertyName("stream")]
    public bool Stream { get; init; }

    [JsonPropertyName("stop_sequences")]
    public string[]? StopSequences { get; init; }

    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }

    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }

    [JsonPropertyName("top_k")]
    public int? TopK { get; init; }

    [JsonPropertyName("tools")]
    public AnthropicToolDto[]? Tools { get; init; }

    /// <summary>
    /// Tool choice: <c>{"type":"auto"|"any"|"none"}</c> or
    /// <c>{"type":"tool","name":"..."}</c>. Parsed by the converter.
    /// </summary>
    [JsonPropertyName("tool_choice")]
    public JsonElement? ToolChoice { get; init; }

    /// <summary>Opaque request metadata (e.g. <c>user_id</c>). Accepted and ignored.</summary>
    [JsonPropertyName("metadata")]
    public JsonElement? Metadata { get; init; }
}

/// <summary>
/// A single Anthropic message. <c>role</c> is <c>"user"</c> or <c>"assistant"</c>;
/// <c>content</c> is either a string or an array of content blocks.
/// </summary>
public sealed record AnthropicMessageDto
{
    [JsonPropertyName("role")]
    public required string Role { get; init; }

    [JsonPropertyName("content")]
    public JsonElement Content { get; init; }
}

/// <summary>Anthropic tool definition. <c>input_schema</c> is a JSON Schema object.</summary>
public sealed record AnthropicToolDto
{
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("description")]
    public string? Description { get; init; }

    [JsonPropertyName("input_schema")]
    public JsonElement? InputSchema { get; init; }
}

// --- Response ---------------------------------------------------------------

/// <summary>Anthropic-compatible non-streaming message response.</summary>
public sealed record AnthropicMessageResponse
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("type")]
    public string Type { get; init; } = "message";

    [JsonPropertyName("role")]
    public string Role { get; init; } = "assistant";

    [JsonPropertyName("content")]
    public required AnthropicContentBlockDto[] Content { get; init; }

    [JsonPropertyName("model")]
    public required string Model { get; init; }

    // stop_reason / stop_sequence are always emitted (even when null) to match
    // the Anthropic wire format, overriding the context's WhenWritingNull default.
    [JsonPropertyName("stop_reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
    public string? StopReason { get; init; }

    [JsonPropertyName("stop_sequence")]
    [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
    public string? StopSequence { get; init; }

    [JsonPropertyName("usage")]
    public required AnthropicUsageDto Usage { get; init; }
}

/// <summary>
/// A response content block. <c>type</c> is <c>"text"</c> or <c>"tool_use"</c>;
/// only the fields relevant to the type are populated (others omitted).
/// </summary>
public sealed record AnthropicContentBlockDto
{
    [JsonPropertyName("type")]
    public required string Type { get; init; }

    [JsonPropertyName("text")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Text { get; init; }

    [JsonPropertyName("id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Id { get; init; }

    [JsonPropertyName("name")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Name { get; init; }

    [JsonPropertyName("input")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Input { get; init; }
}

/// <summary>Anthropic token usage (<c>input_tokens</c>/<c>output_tokens</c>).</summary>
public sealed record AnthropicUsageDto
{
    [JsonPropertyName("input_tokens")]
    public int InputTokens { get; init; }

    [JsonPropertyName("output_tokens")]
    public int OutputTokens { get; init; }
}

/// <summary><c>POST /v1/messages/count_tokens</c> response.</summary>
public sealed record AnthropicCountTokensResponse
{
    [JsonPropertyName("input_tokens")]
    public int InputTokens { get; init; }
}

// --- Error envelope ---------------------------------------------------------

/// <summary>Anthropic error envelope: <c>{"type":"error","error":{...}}</c>.</summary>
public sealed record AnthropicErrorResponse
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "error";

    [JsonPropertyName("error")]
    public required AnthropicErrorBody Error { get; init; }
}

/// <summary>Inner body of an Anthropic error envelope.</summary>
public sealed record AnthropicErrorBody
{
    /// <summary>Error category, e.g. <c>invalid_request_error</c>, <c>api_error</c>.</summary>
    [JsonPropertyName("type")]
    public required string Type { get; init; }

    [JsonPropertyName("message")]
    public required string Message { get; init; }
}

// --- Streaming events -------------------------------------------------------
// Each is emitted as a named SSE event: `event: <type>\ndata: <json>\n\n`.

/// <summary><c>message_start</c> streaming event.</summary>
public sealed record AnthropicMessageStartEvent
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "message_start";

    [JsonPropertyName("message")]
    public required AnthropicMessageResponse Message { get; init; }
}

/// <summary><c>content_block_start</c> streaming event.</summary>
public sealed record AnthropicContentBlockStartEvent
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "content_block_start";

    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("content_block")]
    public required AnthropicContentBlockDto ContentBlock { get; init; }
}

/// <summary><c>content_block_delta</c> streaming event.</summary>
public sealed record AnthropicContentBlockDeltaEvent
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "content_block_delta";

    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("delta")]
    public required AnthropicStreamDeltaDto Delta { get; init; }
}

/// <summary>
/// Incremental delta inside a <c>content_block_delta</c> event:
/// <c>text_delta</c> (carries <c>text</c>) or <c>input_json_delta</c>
/// (carries <c>partial_json</c>).
/// </summary>
public sealed record AnthropicStreamDeltaDto
{
    [JsonPropertyName("type")]
    public required string Type { get; init; }

    [JsonPropertyName("text")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Text { get; init; }

    [JsonPropertyName("partial_json")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? PartialJson { get; init; }
}

/// <summary><c>content_block_stop</c> streaming event.</summary>
public sealed record AnthropicContentBlockStopEvent
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "content_block_stop";

    [JsonPropertyName("index")]
    public int Index { get; init; }
}

/// <summary><c>message_delta</c> streaming event (final stop reason + usage).</summary>
public sealed record AnthropicMessageDeltaEvent
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "message_delta";

    [JsonPropertyName("delta")]
    public required AnthropicMessageDeltaBody Delta { get; init; }

    [JsonPropertyName("usage")]
    public required AnthropicUsageDto Usage { get; init; }
}

/// <summary>Delta body of a <c>message_delta</c> event.</summary>
public sealed record AnthropicMessageDeltaBody
{
    [JsonPropertyName("stop_reason")]
    [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
    public string? StopReason { get; init; }

    [JsonPropertyName("stop_sequence")]
    [JsonIgnore(Condition = JsonIgnoreCondition.Never)]
    public string? StopSequence { get; init; }
}

/// <summary><c>message_stop</c> streaming event.</summary>
public sealed record AnthropicMessageStopEvent
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "message_stop";
}

/// <summary><c>ping</c> streaming keep-alive event.</summary>
public sealed record AnthropicPingEvent
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = "ping";
}
