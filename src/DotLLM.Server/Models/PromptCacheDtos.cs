using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>
/// POST /v1/prompt-cache/{id} body — register a named prefix.
/// </summary>
public sealed record PromptCacheRegisterRequest
{
    /// <summary>Raw text prompt to register. Either this or <see cref="TokenIds"/> must be set.</summary>
    [JsonPropertyName("prompt")]
    public string? Prompt { get; init; }

    /// <summary>Pre-tokenised prompt (alternative to <see cref="Prompt"/>).</summary>
    [JsonPropertyName("token_ids")]
    public int[]? TokenIds { get; init; }
}

/// <summary>
/// Response describing a named prefix entry.
/// </summary>
public sealed record PromptCacheResponse
{
    [JsonPropertyName("prefix_id")]
    public required string PrefixId { get; init; }

    [JsonPropertyName("tokens")]
    public required int Tokens { get; init; }

    [JsonPropertyName("blocks")]
    public required int Blocks { get; init; }

    [JsonPropertyName("status")]
    public required string Status { get; init; }
}

/// <summary>
/// Summary of the cross-request prefix cache returned by GET /v1/prompt-cache.
/// </summary>
public sealed record PromptCacheStatsResponse
{
    [JsonPropertyName("enabled")]
    public required bool Enabled { get; init; }

    [JsonPropertyName("block_size")]
    public required int BlockSize { get; init; }

    [JsonPropertyName("nodes")]
    public required int Nodes { get; init; }

    [JsonPropertyName("hit_tokens")]
    public required long HitTokens { get; init; }

    [JsonPropertyName("miss_tokens")]
    public required long MissTokens { get; init; }

    [JsonPropertyName("lookups")]
    public required long Lookups { get; init; }

    [JsonPropertyName("hits")]
    public required long Hits { get; init; }

    [JsonPropertyName("misses")]
    public required long Misses { get; init; }

    [JsonPropertyName("eviction_refusals")]
    public required long EvictionRefusals { get; init; }

    [JsonPropertyName("free_blocks")]
    public required int FreeBlocks { get; init; }

    [JsonPropertyName("total_blocks")]
    public required int TotalBlocks { get; init; }
}
