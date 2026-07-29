using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// Request body for <c>POST /v1/lora/load</c> — register a LoRA adapter
/// from a HuggingFace PEFT directory under a logical name.
/// </summary>
public sealed record LoraLoadRequest
{
    /// <summary>Logical name to register the adapter under (used in chat requests via <c>lora_adapter</c>).</summary>
    [JsonPropertyName("name")]
    public required string Name { get; init; }

    /// <summary>
    /// Path to a HuggingFace PEFT adapter directory (containing
    /// <c>adapter_config.json</c> + <c>adapter_model.safetensors</c>).
    /// </summary>
    [JsonPropertyName("path")]
    public required string Path { get; init; }
}

/// <summary>
/// Response body for <c>POST /v1/lora/load</c>.
/// </summary>
public sealed record LoraLoadResponse
{
    [JsonPropertyName("status")]
    public required string Status { get; init; }

    [JsonPropertyName("name")]
    public required string Name { get; init; }

    [JsonPropertyName("rank")]
    public int Rank { get; init; }

    [JsonPropertyName("alpha")]
    public float Alpha { get; init; }

    [JsonPropertyName("target_modules")]
    public required string[] TargetModules { get; init; }
}

/// <summary>
/// Response body for <c>GET /v1/lora</c> — list of currently-registered adapter names.
/// </summary>
public sealed record LoraListResponse
{
    [JsonPropertyName("adapters")]
    public required string[] Adapters { get; init; }
}
