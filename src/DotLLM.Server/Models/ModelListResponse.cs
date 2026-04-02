using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// OpenAI-compatible model list response for GET /v1/models.
/// </summary>
public sealed record ModelListResponse
{
    [JsonPropertyName("object")]
    public string Object { get; init; } = "list";

    [JsonPropertyName("data")]
    public required ModelInfoDto[] Data { get; init; }
}

/// <summary>
/// Individual model information.
/// </summary>
public sealed record ModelInfoDto
{
    [JsonPropertyName("id")]
    public required string Id { get; init; }

    [JsonPropertyName("object")]
    public string Object { get; init; } = "model";

    [JsonPropertyName("created")]
    public long Created { get; init; }

    [JsonPropertyName("owned_by")]
    public string OwnedBy { get; init; } = "dotllm";
}
