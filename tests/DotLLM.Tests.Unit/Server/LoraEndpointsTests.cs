using System.Text.Json;
using DotLLM.Core.Lora;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for the LoRA admin / request-resolution surface introduced in
/// Phase 4c. Covers (a) the additive <c>lora_adapter</c> request-DTO
/// field is backwards-compatible at the deserializer level, (b)
/// <see cref="LoraEndpoints.Resolve"/> returns null on no-name and
/// throws with a useful list on bad-name, (c) registry round-trip via
/// the in-memory factory.
/// </summary>
public sealed class LoraEndpointsTests
{
    private static LoraAdapter NewSyntheticAdapter(string name) =>
        new(name, rank: 4, alpha: 8f, targetModules: ["q_proj"]);

    private static ServerState NewState(ILoraAdapterRegistry? registry, bool allowAdmin = false) =>
        new()
        {
            Options = new ServerOptions { Model = "test", AllowLoraAdminApi = allowAdmin },
            LoraRegistry = registry,
        };

    // ── DTO deserialization: backwards-compat ───────────────────────────

    [Fact]
    public void ChatCompletionRequest_NoLoraAdapter_DeserializesAsNull()
    {
        const string json = """
        {"messages":[{"role":"user","content":"hi"}],"max_tokens":4}
        """;
        var req = JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest);
        Assert.NotNull(req);
        Assert.Null(req!.LoraAdapter);
    }

    [Fact]
    public void ChatCompletionRequest_WithLoraAdapter_DeserializesField()
    {
        const string json = """
        {"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"lora_adapter":"my-adapter"}
        """;
        var req = JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest);
        Assert.NotNull(req);
        Assert.Equal("my-adapter", req!.LoraAdapter);
    }

    [Fact]
    public void CompletionRequest_NoLoraAdapter_DeserializesAsNull()
    {
        const string json = """
        {"prompt":"hello","max_tokens":4}
        """;
        var req = JsonSerializer.Deserialize(json, ServerJsonContext.Default.CompletionRequest);
        Assert.NotNull(req);
        Assert.Null(req!.LoraAdapter);
    }

    [Fact]
    public void CompletionRequest_WithLoraAdapter_DeserializesField()
    {
        const string json = """
        {"prompt":"hello","lora_adapter":"adapter-2"}
        """;
        var req = JsonSerializer.Deserialize(json, ServerJsonContext.Default.CompletionRequest);
        Assert.NotNull(req);
        Assert.Equal("adapter-2", req!.LoraAdapter);
    }

    // ── Resolve(): null/empty/missing/found ─────────────────────────────

    [Fact]
    public void Resolve_NullName_ReturnsNullAdapter()
    {
        using var registry = new LoraAdapterRegistry((n, p) => NewSyntheticAdapter(n));
        var state = NewState(registry);
        var result = LoraEndpoints.Resolve(null, state);
        Assert.Null(result);
    }

    [Fact]
    public void Resolve_EmptyName_ReturnsNullAdapter()
    {
        using var registry = new LoraAdapterRegistry((n, p) => NewSyntheticAdapter(n));
        var state = NewState(registry);
        var result = LoraEndpoints.Resolve("", state);
        Assert.Null(result);
    }

    [Fact]
    public void Resolve_UnknownName_ThrowsWithAvailableList()
    {
        using var registry = new LoraAdapterRegistry((n, p) => NewSyntheticAdapter(n));
        registry.Load("alpha", "p");
        registry.Load("beta", "p");

        var state = NewState(registry);
        var ex = Assert.Throws<LoraAdapterNotFoundException>(
            () => LoraEndpoints.Resolve("does-not-exist", state));
        Assert.Contains("does-not-exist", ex.Message);
        // Available adapters are listed for diagnostic purposes
        Assert.Contains("alpha", ex.Message);
        Assert.Contains("beta", ex.Message);
    }

    [Fact]
    public void Resolve_UnknownName_NoneLoaded_ReportsNoneLoaded()
    {
        using var registry = new LoraAdapterRegistry((n, p) => NewSyntheticAdapter(n));
        var state = NewState(registry);
        var ex = Assert.Throws<LoraAdapterNotFoundException>(
            () => LoraEndpoints.Resolve("missing", state));
        Assert.Contains("none loaded", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Resolve_KnownName_ReturnsAdapter()
    {
        using var registry = new LoraAdapterRegistry((n, p) => NewSyntheticAdapter(n));
        registry.Load("present", "p");
        var state = NewState(registry);
        var adapter = LoraEndpoints.Resolve("present", state);
        Assert.NotNull(adapter);
        Assert.Equal("present", adapter!.Name);
    }

    [Fact]
    public void Resolve_NoRegistry_ThrowsForNonNullName()
    {
        var state = NewState(registry: null);
        Assert.Throws<LoraAdapterNotFoundException>(
            () => LoraEndpoints.Resolve("anything", state));
    }

    // ── Registry round-trip semantics ───────────────────────────────────

    [Fact]
    public void Registry_LoadListUnload_RoundTrip()
    {
        using var registry = new LoraAdapterRegistry((n, p) => NewSyntheticAdapter(n));
        Assert.Empty(registry.List());

        registry.Load("a", "p");
        registry.Load("b", "p");
        var listed = registry.List();
        Assert.Contains("a", listed);
        Assert.Contains("b", listed);
        Assert.Equal(2, listed.Count);

        registry.Unload("a");
        listed = registry.List();
        Assert.DoesNotContain("a", listed);
        Assert.Contains("b", listed);
    }

    [Fact]
    public void Registry_DuplicateLoad_Throws()
    {
        using var registry = new LoraAdapterRegistry((n, p) => NewSyntheticAdapter(n));
        registry.Load("dup", "p");
        Assert.Throws<InvalidOperationException>(() => registry.Load("dup", "p"));
    }

    // ── Admin gating ────────────────────────────────────────────────────

    [Fact]
    public void AdminFlag_DefaultsToFalse()
    {
        var opts = new ServerOptions { Model = "x" };
        Assert.False(opts.AllowLoraAdminApi);
    }

    [Fact]
    public void AdminFlag_HonoursOptInTrue()
    {
        var opts = new ServerOptions { Model = "x", AllowLoraAdminApi = true };
        Assert.True(opts.AllowLoraAdminApi);
    }
}
