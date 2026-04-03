using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// Response for <c>GET /v1/models/inspect</c> — lightweight GGUF metadata.
/// </summary>
public sealed record ModelInspectResponse
{
    [JsonPropertyName("architecture")]
    public required string Architecture { get; init; }

    [JsonPropertyName("num_layers")]
    public int NumLayers { get; init; }

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; init; }

    [JsonPropertyName("num_kv_heads")]
    public int NumKvHeads { get; init; }

    [JsonPropertyName("head_dim")]
    public int HeadDim { get; init; }

    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; init; }

    [JsonPropertyName("max_sequence_length")]
    public int MaxSequenceLength { get; init; }

    [JsonPropertyName("file_size_bytes")]
    public long FileSizeBytes { get; init; }
}
