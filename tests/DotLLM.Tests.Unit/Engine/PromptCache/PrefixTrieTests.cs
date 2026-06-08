using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.PromptCache;

public sealed class PrefixTrieTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int BlockSize = 4;
    private const int TotalBlocks = 16;

    private static KvBlockPool CreatePool() =>
        new(NumLayers, NumKvHeads, HeadDim, BlockSize, TotalBlocks);

    // Build a prompt of length tokenCount with predictable token IDs.
    private static int[] MakePrompt(int seed, int tokenCount)
    {
        var p = new int[tokenCount];
        for (int i = 0; i < tokenCount; i++) p[i] = seed * 1000 + i + 1;
        return p;
    }

    // Simulate "fresh prefill": allocate one block per chunk and insert into the trie.
    private static List<int> AllocateAndInsert(PrefixTrie trie, KvBlockPool pool, int[] tokens, int startTokenIndex = 0)
    {
        int blockSize = trie.BlockSize;
        int blocksNeeded = (tokens.Length - startTokenIndex) / blockSize;
        var blocks = new List<int>(blocksNeeded);
        for (int b = 0; b < blocksNeeded; b++)
            blocks.Add(pool.Allocate());
        trie.Insert(tokens, startTokenIndex, blocks);
        return blocks;
    }

    [Fact]
    public void Lookup_EmptyTrie_ReturnsZero()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);
        var matched = new List<int>();

        int n = trie.Lookup(MakePrompt(1, 16), matched);

        Assert.Equal(0, n);
        Assert.Empty(matched);
    }

    [Fact]
    public void Lookup_ExactFullPrefix_ReturnsAllBlocks()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);
        var tokens = MakePrompt(1, 16); // 4 blocks of 4 tokens each
        var inserted = AllocateAndInsert(trie, pool, tokens);

        var matched = new List<int>();
        int n = trie.Lookup(tokens, matched);

        Assert.Equal(16, n);
        Assert.Equal(4, matched.Count);
        Assert.Equal(inserted, matched);

        // Lookup acquired one extra ref per block on top of the trie's own ref.
        foreach (int b in matched)
            Assert.Equal(2, pool.RefCount(b));

        trie.Release(matched);
    }

    [Fact]
    public void Lookup_PartialPrefix_ReturnsCommonBlocksOnly()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);

        var a = MakePrompt(1, 16);                       // 4 blocks
        var b = (int[])a.Clone();
        for (int i = 8; i < 16; i++) b[i] = 9000 + i;    // diverge after block 2
        AllocateAndInsert(trie, pool, a);

        var matched = new List<int>();
        int n = trie.Lookup(b, matched);

        Assert.Equal(8, n);
        Assert.Equal(2, matched.Count);
        trie.Release(matched);
    }

    [Fact]
    public void Lookup_DifferentPrompt_ReturnsZero()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);
        AllocateAndInsert(trie, pool, MakePrompt(1, 16));

        var matched = new List<int>();
        int n = trie.Lookup(MakePrompt(2, 16), matched);

        Assert.Equal(0, n);
        Assert.Empty(matched);
    }

    [Fact]
    public void Lookup_PartialBlock_DoesNotMatch()
    {
        // Last block of the prompt is not full → cannot be matched against the trie.
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);
        AllocateAndInsert(trie, pool, MakePrompt(1, 16));

        // 6 tokens = 1 full block + 2 leftover
        var matched = new List<int>();
        var prompt = new int[6];
        Array.Copy(MakePrompt(1, 16), prompt, 6);
        int n = trie.Lookup(prompt, matched);

        Assert.Equal(4, n);
        Assert.Single(matched);
        trie.Release(matched);
    }

    [Fact]
    public void Insert_ExistingPath_IsIdempotent()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);
        var tokens = MakePrompt(1, 16);
        AllocateAndInsert(trie, pool, tokens);

        int countBefore = trie.NodeCount;
        int freeBefore = pool.FreeBlocks;

        // Re-insert identical tokens with freshly allocated (duplicate-content) blocks.
        // Trie should release them rather than adding duplicate nodes.
        var dupBlocks = new List<int>
        {
            pool.Allocate(), pool.Allocate(), pool.Allocate(), pool.Allocate(),
        };
        int linked = trie.Insert(tokens, 0, dupBlocks);

        Assert.Equal(0, linked);
        Assert.Equal(countBefore, trie.NodeCount);
        Assert.Equal(freeBefore, pool.FreeBlocks); // duplicates returned to pool
    }

    [Fact]
    public void Refcount_TrackedAcrossLookupAndRelease()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);
        var tokens = MakePrompt(1, 16);
        var inserted = AllocateAndInsert(trie, pool, tokens);

        int firstBlock = inserted[0];
        Assert.Equal(1, pool.RefCount(firstBlock));     // trie holds 1 ref

        var matched = new List<int>();
        trie.Lookup(tokens, matched);
        Assert.Equal(2, pool.RefCount(firstBlock));     // trie + lookup

        trie.Release(matched);
        Assert.Equal(1, pool.RefCount(firstBlock));     // back to trie-only
    }

    [Fact]
    public void EvictOneLru_RemovesUnreferencedLeaf()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);

        var promptA = MakePrompt(1, 16);
        var blocksA = AllocateAndInsert(trie, pool, promptA);
        int leafBlockA = blocksA[^1];

        // No external refs on any of A's blocks → leaf is evictable.
        int freeBefore = pool.FreeBlocks;
        int evicted = trie.EvictOneLru();

        Assert.Equal(leafBlockA, evicted);
        Assert.Equal(freeBefore + 1, pool.FreeBlocks);
    }

    [Fact]
    public void EvictOneLru_PrefersOlderLeaf()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);

        var promptA = MakePrompt(1, 16);
        var blocksA = AllocateAndInsert(trie, pool, promptA);
        int oldestLeaf = blocksA[^1];

        Thread.Sleep(5); // ensure distinct ticks

        var promptB = MakePrompt(2, 16);
        AllocateAndInsert(trie, pool, promptB);

        int evicted = trie.EvictOneLru();
        Assert.Equal(oldestLeaf, evicted);
    }

    [Fact]
    public void EvictOneLru_SkipsReferencedNodes()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);

        var tokens = MakePrompt(1, 16);
        AllocateAndInsert(trie, pool, tokens);

        // Pin every block via a Lookup (acquires external refs).
        var matched = new List<int>();
        trie.Lookup(tokens, matched);

        int evicted = trie.EvictOneLru();
        Assert.Equal(-1, evicted);

        trie.Release(matched);
    }

    [Fact]
    public void RegisterNamedPrefix_PinsPath_PreventsEviction()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);

        var tokens = MakePrompt(1, 16);
        AllocateAndInsert(trie, pool, tokens);

        int matched = trie.RegisterNamedPrefix("sys-1", tokens);
        Assert.Equal(16, matched);

        // Cannot evict — every block is pinned.
        Assert.Equal(-1, trie.EvictOneLru());

        bool unpinned = trie.UnpinNamedPrefix("sys-1");
        Assert.True(unpinned);

        // After unpin, leaf is evictable again.
        Assert.NotEqual(-1, trie.EvictOneLru());
    }

    [Fact]
    public void RegisterNamedPrefix_Duplicate_Throws()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);
        var tokens = MakePrompt(1, 16);
        AllocateAndInsert(trie, pool, tokens);

        trie.RegisterNamedPrefix("a", tokens);
        Assert.Throws<InvalidOperationException>(() => trie.RegisterNamedPrefix("a", tokens));

        trie.UnpinNamedPrefix("a");
    }

    [Fact]
    public void InspectNamedPrefix_ReturnsTokenDepth()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);
        var tokens = MakePrompt(1, 16);
        AllocateAndInsert(trie, pool, tokens);
        trie.RegisterNamedPrefix("sys", tokens);

        var info = trie.InspectNamedPrefix("sys");

        Assert.NotNull(info);
        Assert.Equal(16, info!.Value.Tokens);
        Assert.Equal(4, info.Value.Blocks);

        trie.UnpinNamedPrefix("sys");
    }

    [Fact]
    public void MaxPrefixDepth_BoundsMatch()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool, maxPrefixDepth: 8); // cap at 2 blocks
        AllocateAndInsert(trie, pool, MakePrompt(1, 16));

        var matched = new List<int>();
        int n = trie.Lookup(MakePrompt(1, 16), matched);

        Assert.Equal(8, n);
        Assert.Equal(2, matched.Count);
        trie.Release(matched);
    }

    [Fact]
    public void Clear_FreesAllNodes()
    {
        using var pool = CreatePool();
        var trie = new PrefixTrie(pool);
        AllocateAndInsert(trie, pool, MakePrompt(1, 16));
        AllocateAndInsert(trie, pool, MakePrompt(2, 16));

        Assert.True(trie.NodeCount > 0);
        int freeBefore = pool.FreeBlocks;

        trie.Clear();

        Assert.Equal(0, trie.NodeCount);
        Assert.True(pool.FreeBlocks > freeBefore);
    }
}
