using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible logprobs container. Wraps the per-token content array.
/// </summary>
public sealed record LogprobsDto
{
    [JsonPropertyName("content")]
    public required TokenLogprobDto[] Content { get; init; }
}

/// <summary>
/// Per-token log-probability information in the OpenAI format.
/// </summary>
public sealed record TokenLogprobDto
{
    [JsonPropertyName("token")]
    public required string Token { get; init; }

    [JsonPropertyName("logprob")]
    public required float Logprob { get; init; }

    [JsonPropertyName("bytes")]
    public byte[]? Bytes { get; init; }

    [JsonPropertyName("top_logprobs")]
    public required TopLogprobDto[] TopLogprobs { get; init; }
}

/// <summary>
/// A single alternative token entry in the top logprobs list.
/// </summary>
public sealed record TopLogprobDto
{
    [JsonPropertyName("token")]
    public required string Token { get; init; }

    [JsonPropertyName("logprob")]
    public required float Logprob { get; init; }

    [JsonPropertyName("bytes")]
    public byte[]? Bytes { get; init; }
}
