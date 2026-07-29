using DotLLM.Core.Lora;
using DotLLM.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Tests for <see cref="MultiAdapterBatcher.Group{T}"/>: the Phase 4c
/// first-cut partition contract for multi-adapter dispatch.
/// </summary>
/// <remarks>
/// <para>
/// The current <see cref="DotLLM.Server.ServerState"/> request gate
/// serialises chat-completion requests strictly (a single
/// <c>SemaphoreSlim(1, 1)</c>), so cross-request batching is not
/// performed by the server today. The batcher exists so the partition
/// contract is testable independently and the future scheduler
/// (Phase 4d / Wave 9) can plug in without re-shaping the API.
/// </para>
/// <para>
/// The end-to-end "two concurrent requests with different adapters"
/// test is therefore expressed as a deliberate skip — see
/// <see cref="ConcurrentMixedAdapter_Note"/> — calling out the
/// limitation rather than half-implementing parallel dispatch.
/// </para>
/// </remarks>
public sealed class MultiAdapterBatcherTests
{
    private static DotLLM.Core.Lora.LoraAdapter NewAdapter(string name) =>
        new(name, rank: 4, alpha: 8f, targetModules: ["q_proj"]);

    [Fact]
    public void Group_EmptyBatch_ReturnsEmpty()
    {
        var groups = MultiAdapterBatcher.Group(
            Array.Empty<int>(),
            _ => (ILoraAdapter?)null);
        Assert.Empty(groups);
    }

    [Fact]
    public void Group_AllBaseModel_OneNullGroup()
    {
        int[] reqs = [1, 2, 3, 4];
        var groups = MultiAdapterBatcher.Group(reqs, _ => (ILoraAdapter?)null);
        Assert.Single(groups);
        Assert.Null(groups[0].Adapter);
        Assert.Equal(new[] { 1, 2, 3, 4 }, groups[0].Requests);
    }

    [Fact]
    public void Group_AllSameAdapter_OneGroup()
    {
        using var a = NewAdapter("a");
        int[] reqs = [10, 20, 30];
        var groups = MultiAdapterBatcher.Group(reqs, _ => a);
        Assert.Single(groups);
        Assert.Same(a, groups[0].Adapter);
        Assert.Equal(new[] { 10, 20, 30 }, groups[0].Requests);
    }

    [Fact]
    public void Group_MixedAdapters_PartitionsByReference()
    {
        using var a = NewAdapter("a");
        using var b = NewAdapter("b");
        // Order: a, b, null, a, b, null, b
        var sel = new ILoraAdapter?[] { a, b, null, a, b, null, b };
        int[] reqs = [0, 1, 2, 3, 4, 5, 6];

        var groups = MultiAdapterBatcher.Group(reqs, i => sel[i]);

        // Null group must be yielded first when present.
        Assert.Equal(3, groups.Count);
        Assert.Null(groups[0].Adapter);
        Assert.Equal(new[] { 2, 5 }, groups[0].Requests);

        // Adapter groups follow in first-seen order: a then b.
        Assert.Same(a, groups[1].Adapter);
        Assert.Equal(new[] { 0, 3 }, groups[1].Requests);

        Assert.Same(b, groups[2].Adapter);
        Assert.Equal(new[] { 1, 4, 6 }, groups[2].Requests);
    }

    [Fact]
    public void Group_PreservesIntraGroupOrder()
    {
        using var a = NewAdapter("a");
        // 5 requests, all with adapter a — order must be preserved.
        int[] reqs = [9, 8, 7, 6, 5];
        var groups = MultiAdapterBatcher.Group(reqs, _ => a);
        Assert.Single(groups);
        Assert.Equal(reqs, groups[0].Requests);
    }

    [Fact]
    public void Group_NullAdapterFirstWhenPresent()
    {
        using var a = NewAdapter("a");
        // First request uses a, then a base request — null group must still come first.
        int[] reqs = [0, 1];
        var sel = new ILoraAdapter?[] { a, null };
        var groups = MultiAdapterBatcher.Group(reqs, i => sel[i]);

        Assert.Equal(2, groups.Count);
        Assert.Null(groups[0].Adapter);
        Assert.Same(a, groups[1].Adapter);
    }

    /// <summary>
    /// Skipped placeholder for true concurrent multi-adapter dispatch.
    /// </summary>
    /// <remarks>
    /// The current server's <c>ServerState.ExecuteAsync</c> serialises all
    /// inference requests through a <c>SemaphoreSlim(1, 1)</c>. Two
    /// concurrent /v1/chat/completions calls are therefore processed
    /// strictly sequentially today, so a "submit two concurrent requests
    /// with different adapters in the same engine batch" test would not
    /// exercise the partition logic — the requests would never co-exist
    /// in the same batch.
    /// <para>
    /// Phase 4d / Wave 9 will introduce continuous batching with a
    /// scheduler that can hold multiple in-flight requests; at that point
    /// this skipped test should be unskipped and reformulated to assert
    /// per-request output equivalence between batched and per-request
    /// dispatch.
    /// </para>
    /// </remarks>
    [Fact(Skip = "Continuous batching with mixed adapters in a single forward pass " +
                 "lands in Phase 4d / Wave 9. The server currently serialises requests " +
                 "via SemaphoreSlim(1,1); see MultiAdapterBatcher for the partition contract.")]
    public void ConcurrentMixedAdapter_Note() { }
}
