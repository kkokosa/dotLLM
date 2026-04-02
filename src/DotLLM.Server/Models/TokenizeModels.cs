using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// Request for the /v1/tokenize extension endpoint.
/// </summary>
public sealed record TokenizeRequest
{
    [JsonPropertyName("text")]
    public required string Text { get; init; }

    [JsonPropertyName("model")]
    public string? Model { get; init; }
}

/// <summary>
/// Response for the /v1/tokenize extension endpoint.
/// </summary>
public sealed record TokenizeResponse
{
    [JsonPropertyName("tokens")]
    public required int[] Tokens { get; init; }

    [JsonPropertyName("token_strings")]
    public required string[] TokenStrings { get; init; }

    [JsonPropertyName("count")]
    public int Count { get; init; }
}

/// <summary>
/// Request for the /v1/detokenize extension endpoint.
/// </summary>
public sealed record DetokenizeRequest
{
    [JsonPropertyName("tokens")]
    public required int[] Tokens { get; init; }

    [JsonPropertyName("model")]
    public string? Model { get; init; }
}

/// <summary>
/// Response for the /v1/detokenize extension endpoint.
/// </summary>
public sealed record DetokenizeResponse
{
    [JsonPropertyName("text")]
    public required string Text { get; init; }
}
