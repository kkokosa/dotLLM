using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// Response for <c>GET /props</c> — server configuration and model info.
/// </summary>
public sealed record PropsResponse
{
    [JsonPropertyName("model_id")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ModelId { get; init; }

    [JsonPropertyName("model_path")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? ModelPath { get; init; }

    [JsonPropertyName("architecture")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Architecture { get; init; }

    [JsonPropertyName("num_layers")]
    public int NumLayers { get; init; }

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; init; }

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; init; }

    [JsonPropertyName("max_sequence_length")]
    public int MaxSequenceLength { get; init; }

    [JsonPropertyName("device")]
    public string Device { get; init; } = "cpu";

    [JsonPropertyName("gpu_layers")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? GpuLayers { get; init; }

    [JsonPropertyName("threads")]
    public int Threads { get; init; }

    [JsonPropertyName("sampling_defaults")]
    public required SamplingDefaultsDto SamplingDefaults { get; init; }

    [JsonPropertyName("draft_model_path")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? DraftModelPath { get; init; }

    [JsonPropertyName("is_ready")]
    public bool IsReady { get; init; }
}

/// <summary>
/// Sampling parameter defaults (used in both <c>/props</c> and <c>/v1/config</c>).
/// </summary>
public sealed record SamplingDefaultsDto
{
    [JsonPropertyName("temperature")]
    public float Temperature { get; init; }

    [JsonPropertyName("top_p")]
    public float TopP { get; init; }

    [JsonPropertyName("top_k")]
    public int TopK { get; init; }

    [JsonPropertyName("min_p")]
    public float MinP { get; init; }

    [JsonPropertyName("repetition_penalty")]
    public float RepetitionPenalty { get; init; }

    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; init; }

    [JsonPropertyName("seed")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public int? Seed { get; init; }
}
