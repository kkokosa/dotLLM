using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// Standard error response DTO. Replaces anonymous <c>new { error = "..." }</c> types
/// for AOT-compatible source-generated serialization.
/// </summary>
public sealed record ErrorResponse
{
    [JsonPropertyName("error")]
    public required string Error { get; init; }
}

/// <summary>
/// OpenAI-shaped error response DTO: <c>{"error": {"message": "...", "type": "...", "param": "...", "code": "..."}}</c>.
/// Used when the response needs to look like an upstream OpenAI 4xx so existing client SDKs surface a typed error.
/// </summary>
public sealed record OpenAiErrorResponse
{
    [JsonPropertyName("error")]
    public required OpenAiErrorBody Error { get; init; }
}

/// <summary>Body of an <see cref="OpenAiErrorResponse"/>.</summary>
public sealed record OpenAiErrorBody
{
    [JsonPropertyName("message")]
    public required string Message { get; init; }

    [JsonPropertyName("type")]
    public required string Type { get; init; }

    [JsonPropertyName("param")]
    public string? Param { get; init; }

    [JsonPropertyName("code")]
    public string? Code { get; init; }
}

/// <summary>
/// Standard status response DTO. Replaces anonymous <c>new { status = "..." }</c> types
/// for AOT-compatible source-generated serialization.
/// </summary>
public sealed record StatusResponse
{
    [JsonPropertyName("status")]
    public required string Status { get; init; }
}
