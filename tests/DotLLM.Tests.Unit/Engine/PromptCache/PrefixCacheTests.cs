using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.PromptCache;

public sealed class PrefixCacheTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int MaxSeqLen = 64;

    private static SimpleKvCache CreateCache() =>
        new(NumLayers, NumKvHeads, HeadDim, MaxSeqLen);

    [Fact]
    public void FindMatch_EmptyCache_ReturnsNull()
    {
        using var cache = new PrefixCache(4);
        var (entry, matched) = cache.FindMatch([1, 2, 3]);

        Assert.Null(entry);
        Assert.Equal(0, matched);
    }

    [Fact]
    public void FindMatch_ExactMatch_ReturnsFullLength()
    {
        using var cache = new PrefixCache(4);
        var kvCache = CreateCache();
        int[] tokens = [1, 2, 3, 4, 5];
        cache.Store(tokens, kvCache);

        var (entry, matched) = cache.FindMatch([1, 2, 3, 4, 5]);

        Assert.NotNull(entry);
        Assert.Equal(5, matched);
    }

    [Fact]
    public void FindMatch_PartialMatch_ReturnsMatchedLength()
    {
        using var cache = new PrefixCache(4);
        var kvCache = CreateCache();
        cache.Store([1, 2, 3, 4, 5], kvCache);

        // Prompt shares first 3 tokens with stored entry
        var (entry, matched) = cache.FindMatch([1, 2, 3, 10, 11]);

        Assert.NotNull(entry);
        Assert.Equal(3, matched);
    }

    [Fact]
    public void FindMatch_NoMatch_ReturnsZero()
    {
        using var cache = new PrefixCache(4);
        var kvCache = CreateCache();
        cache.Store([1, 2, 3], kvCache);

        var (entry, matched) = cache.FindMatch([10, 20, 30]);

        Assert.Null(entry); // matchedTokens=0, so entry should be null
        Assert.Equal(0, matched);
    }

    [Fact]
    public void FindMatch_PromptLongerThanStored_MatchesUpToStoredLength()
    {
        using var cache = new PrefixCache(4);
        var kvCache = CreateCache();
        cache.Store([1, 2, 3], kvCache);

        // Prompt starts with [1,2,3] then continues
        var (entry, matched) = cache.FindMatch([1, 2, 3, 4, 5, 6]);

        Assert.NotNull(entry);
        Assert.Equal(3, matched);
    }

    [Fact]
    public void FindMatch_MultpleEntries_ReturnsBestMatch()
    {
        using var cache = new PrefixCache(4);

        var kv1 = CreateCache();
        cache.Store([1, 2], kv1);

        var kv2 = CreateCache();
        cache.Store([1, 2, 3, 4], kv2);

        // Should match the longer prefix
        var (entry, matched) = cache.FindMatch([1, 2, 3, 4, 5]);

        Assert.NotNull(entry);
        Assert.Equal(4, matched);
    }

    [Fact]
    public void Store_SameKvCache_UpdatesTokenSequence()
    {
        using var cache = new PrefixCache(4);
        var kvCache = CreateCache();

        cache.Store([1, 2, 3], kvCache);
        Assert.Equal(1, cache.EntryCount);

        // Store again with same KV-cache but longer sequence
        cache.Store([1, 2, 3, 4, 5], kvCache);
        Assert.Equal(1, cache.EntryCount);

        var (entry, matched) = cache.FindMatch([1, 2, 3, 4, 5]);
        Assert.Equal(5, matched);
    }

    [Fact]
    public void Store_EvictsLRU_WhenAtCapacity()
    {
        using var cache = new PrefixCache(2);

        var kv1 = CreateCache();
        cache.Store([10, 20], kv1);

        var kv2 = CreateCache();
        cache.Store([30, 40], kv2);
        Assert.Equal(2, cache.EntryCount);

        // Access kv2 to make kv1 the LRU
        cache.FindMatch([30, 40]);

        // Adding a third should evict kv1 (LRU)
        var kv3 = CreateCache();
        cache.Store([50, 60], kv3);
        Assert.Equal(2, cache.EntryCount);

        // kv1 should be gone
        var (entry1, matched1) = cache.FindMatch([10, 20]);
        Assert.Equal(0, matched1);

        // kv2 should still be there
        var (entry2, matched2) = cache.FindMatch([30, 40]);
        Assert.Equal(2, matched2);
    }

    [Fact]
    public void Clear_RemovesAllEntries()
    {
        using var cache = new PrefixCache(4);
        cache.Store([1, 2], CreateCache());
        cache.Store([3, 4], CreateCache());

        cache.Clear();

        Assert.Equal(0, cache.EntryCount);
        var (entry, matched) = cache.FindMatch([1, 2]);
        Assert.Equal(0, matched);
    }

    [Fact]
    public void Dispose_ClearsAllEntries()
    {
        var cache = new PrefixCache(4);
        cache.Store([1, 2], CreateCache());
        cache.Store([3, 4], CreateCache());

        cache.Dispose();

        Assert.Equal(0, cache.EntryCount);
    }

    [Fact]
    public void Constructor_ThrowsOnZeroMaxEntries()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PrefixCache(0));
    }
}
