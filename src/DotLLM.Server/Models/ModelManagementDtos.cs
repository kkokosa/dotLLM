using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// Response for <c>GET /v1/models/available</c> — locally downloaded models.
/// </summary>
public sealed record AvailableModelsResponse
{
    [JsonPropertyName("models")]
    public required AvailableModelDto[] Models { get; init; }
}

/// <summary>
/// A locally downloaded GGUF model file.
/// </summary>
public sealed record AvailableModelDto
{
    [JsonPropertyName("repo_id")]
    public required string RepoId { get; init; }

    [JsonPropertyName("filename")]
    public required string Filename { get; init; }

    [JsonPropertyName("full_path")]
    public required string FullPath { get; init; }

    [JsonPropertyName("size_bytes")]
    public long SizeBytes { get; init; }
}

/// <summary>
/// Request for <c>POST /v1/models/load</c> — load/swap a model.
/// </summary>
public sealed record ModelLoadRequest
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("quant")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Quant { get; init; }
}

/// <summary>
/// Response for <c>POST /v1/models/load</c>.
/// </summary>
public sealed record ModelLoadResponse
{
    [JsonPropertyName("status")]
    public required string Status { get; init; }

    [JsonPropertyName("model")]
    public required string Model { get; init; }
}
