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
/// Standard status response DTO. Replaces anonymous <c>new { status = "..." }</c> types
/// for AOT-compatible source-generated serialization.
/// </summary>
public sealed record StatusResponse
{
    [JsonPropertyName("status")]
    public required string Status { get; init; }
}
