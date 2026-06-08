using System.Text.Json;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// DTO + state-shape tests for the /v1/prompt-cache surface. We don't spin
/// up the full ASP.NET pipeline here; the endpoint handlers are simple
/// pass-throughs to <see cref="PrefixTrieManager"/>, so we cover the manager
/// (engine tests) and the DTO/JSON wiring (here).
/// </summary>
public sealed class PromptCacheEndpointTests
{
    [Fact]
    public void Request_DeserializesFromJson_PromptOnly()
    {
        const string json = """{"prompt":"system: you are helpful"}""";
        var req = JsonSerializer.Deserialize(json, ServerJsonContext.Default.PromptCacheRegisterRequest);

        Assert.NotNull(req);
        Assert.Equal("system: you are helpful", req!.Prompt);
        Assert.Null(req.TokenIds);
    }

    [Fact]
    public void Request_DeserializesFromJson_TokenIdsOnly()
    {
        const string json = """{"token_ids":[1,2,3,4]}""";
        var req = JsonSerializer.Deserialize(json, ServerJsonContext.Default.PromptCacheRegisterRequest);

        Assert.NotNull(req);
        Assert.Equal(new[] { 1, 2, 3, 4 }, req!.TokenIds);
        Assert.Null(req.Prompt);
    }

    [Fact]
    public void Response_SerializesToJson_SnakeCase()
    {
        var resp = new PromptCacheResponse
        {
            PrefixId = "sys-1",
            Tokens = 256,
            Blocks = 16,
            Status = "registered",
        };

        string json = JsonSerializer.Serialize(resp, ServerJsonContext.Default.PromptCacheResponse);

        Assert.Contains("\"prefix_id\":\"sys-1\"", json);
        Assert.Contains("\"tokens\":256", json);
        Assert.Contains("\"blocks\":16", json);
    }

    [Fact]
    public void StatsResponse_SerializesAllFields()
    {
        var resp = new PromptCacheStatsResponse
        {
            Enabled = true,
            BlockSize = 16,
            Nodes = 4,
            HitTokens = 256,
            MissTokens = 32,
            Lookups = 10,
            Hits = 7,
            Misses = 3,
            EvictionRefusals = 1,
            FreeBlocks = 60,
            TotalBlocks = 64,
        };

        string json = JsonSerializer.Serialize(resp, ServerJsonContext.Default.PromptCacheStatsResponse);

        Assert.Contains("\"enabled\":true", json);
        Assert.Contains("\"block_size\":16", json);
        Assert.Contains("\"hit_tokens\":256", json);
        Assert.Contains("\"miss_tokens\":32", json);
        Assert.Contains("\"eviction_refusals\":1", json);
        Assert.Contains("\"total_blocks\":64", json);
    }

    [Fact]
    public void ChatRequest_DeserializesPrefixId()
    {
        const string json = """
        {"messages":[{"role":"user","content":"hi"}],"prefix_id":"sys-main"}
        """;
        var req = JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest);

        Assert.NotNull(req);
        Assert.Equal("sys-main", req!.PrefixId);
    }

    [Fact]
    public void CompletionRequest_DeserializesPrefixId()
    {
        const string json = """{"prompt":"...","prefix_id":"sys-main","max_tokens":1}""";
        var req = JsonSerializer.Deserialize(json, ServerJsonContext.Default.CompletionRequest);

        Assert.NotNull(req);
        Assert.Equal("sys-main", req!.PrefixId);
    }

    [Fact]
    public void ServerState_PrefixTrieManager_Field_RoundTrips()
    {
        // Smoke test: the manager field is wired correctly into ServerState.
        using var factory = new PagedKvCacheFactory(2, 2, 4, blockSize: 8, maxTotalTokens: 64);
        using var mgr = new PrefixTrieManager(factory);

        var state = new ServerState
        {
            Options = new ServerOptions { Model = "test" },
            PrefixTrieManager = mgr,
        };

        Assert.NotNull(state.PrefixTrieManager);
        var stats = state.PrefixTrieManager!.GetStats();
        Assert.True(stats.Enabled);
        Assert.Equal(8, stats.BlockSize);
    }
}
