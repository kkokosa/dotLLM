using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.PromptCache;

public sealed class PrefixTrieManagerTests
{
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int BlockSize = 4;
    private const int TotalBlocks = 32;
    private const int MaxSeqLen = 64;

    private static PagedKvCacheFactory CreateFactory() =>
        new(NumLayers, NumKvHeads, HeadDim, BlockSize, maxTotalTokens: TotalBlocks * BlockSize);

    private static int[] MakePrompt(int seed, int tokenCount)
    {
        var p = new int[tokenCount];
        for (int i = 0; i < tokenCount; i++) p[i] = seed * 1000 + i + 1;
        return p;
    }

    // Simulate a sequence: admit, fill any new blocks with fake K/V, complete.
    private static unsafe void SimulateSequence(PrefixTrieManager mgr, int[] tokens, int generatedTokens = 0)
    {
        int promptLen = tokens.Length;
        int totalLen = promptLen + generatedTokens;
        var admission = mgr.Admit(tokens, MaxSeqLen);
        var cache = admission.Cache;

        // Write fake KV data for positions [cachedTokens, totalLen).
        int kvStride = NumKvHeads * HeadDim;
        int writeStart = admission.CachedTokens;
        int writeCount = totalLen - writeStart;
        if (writeCount > 0)
        {
            // Construct the full sequence buffer used for the trie key (prompt + dummy gen tokens).
            var positions = new int[writeCount];
            for (int i = 0; i < writeCount; i++) positions[i] = writeStart + i;

            int floatCount = writeCount * kvStride;
            float[] k = new float[floatCount];
            float[] v = new float[floatCount];
            fixed (float* kp = k)
            fixed (float* vp = v)
            {
                var kRef = new DotLLM.Core.Tensors.TensorRef(writeCount, kvStride,
                    DotLLM.Core.Tensors.DType.Float32, -1, (nint)kp);
                var vRef = new DotLLM.Core.Tensors.TensorRef(writeCount, kvStride,
                    DotLLM.Core.Tensors.DType.Float32, -1, (nint)vp);
                for (int layer = 0; layer < NumLayers; layer++)
                    cache.Update(kRef, vRef, positions, layer);
            }
        }

        // Full token sequence (prompt + dummy generated for the trie key).
        var fullTokens = new int[totalLen];
        Array.Copy(tokens, fullTokens, promptLen);
        for (int i = 0; i < generatedTokens; i++) fullTokens[promptLen + i] = 99000 + i;

        mgr.RecordCompletion(cache, fullTokens);
        cache.Dispose();
    }

    [Fact]
    public void Admit_FirstSequence_NoCacheHit()
    {
        using var factory = CreateFactory();
        using var mgr = new PrefixTrieManager(factory);

        var admission = mgr.Admit(MakePrompt(1, 16), MaxSeqLen);

        Assert.Equal(0, admission.CachedTokens);
        admission.Cache.Dispose();
    }

    [Fact]
    public void SecondSequence_SharedPrefix_ReusesBlocks()
    {
        using var factory = CreateFactory();
        using var mgr = new PrefixTrieManager(factory);

        var prompt = MakePrompt(1, 16);
        SimulateSequence(mgr, prompt);

        // Second call with same prompt — full prefix should be reused.
        var admission = mgr.Admit(prompt, MaxSeqLen);
        Assert.Equal(16, admission.CachedTokens);
        admission.Cache.Dispose();
    }

    [Fact]
    public void TwoSequences_DifferentSuffixes_ShareCommonPrefix()
    {
        using var factory = CreateFactory();
        using var mgr = new PrefixTrieManager(factory);

        // Sequence A: 32 tokens, all unique
        var a = MakePrompt(1, 32);
        SimulateSequence(mgr, a);

        // Sequence B: shares first 12 tokens (= 3 blocks) with A, diverges after.
        var b = new int[32];
        Array.Copy(a, b, 12);
        for (int i = 12; i < 32; i++) b[i] = 7000 + i;

        var admission = mgr.Admit(b, MaxSeqLen);
        Assert.Equal(12, admission.CachedTokens); // 3 blocks × 4 tokens
        admission.Cache.Dispose();
    }

    [Fact]
    public void FourConcurrentSequences_SharePrefill_OnlyOnce()
    {
        // Simulates the acceptance criterion: 4 concurrent requests sharing a
        // long prefix should only run prefill compute once.
        using var factory = CreateFactory();
        using var mgr = new PrefixTrieManager(factory);

        // Shared "system prompt" of 32 tokens (8 blocks).
        var sysPrompt = MakePrompt(42, 32);

        // First request fills the trie.
        SimulateSequence(mgr, sysPrompt, generatedTokens: 0);

        long firstHitTokens = mgr.Trie.HitTokens;
        Assert.Equal(0, firstHitTokens);

        // Next three requests should all be full hits.
        for (int i = 0; i < 3; i++)
        {
            var admission = mgr.Admit(sysPrompt, MaxSeqLen);
            Assert.Equal(32, admission.CachedTokens);
            admission.Cache.Dispose();
        }

        Assert.Equal(32L * 3, mgr.Trie.HitTokens);
    }

    [Fact]
    public void RecordCompletion_InsertsNewSuffixBlocks()
    {
        using var factory = CreateFactory();
        using var mgr = new PrefixTrieManager(factory);

        var prompt = MakePrompt(1, 8);
        SimulateSequence(mgr, prompt);

        // Extended prompt: shares first 8 tokens with the stored sequence.
        var extended = new int[16];
        Array.Copy(prompt, extended, 8);
        for (int i = 8; i < 16; i++) extended[i] = 5000 + i;

        SimulateSequence(mgr, extended);

        // Now request the extended prompt again — should be a 16-token hit.
        var admission = mgr.Admit(extended, MaxSeqLen);
        Assert.Equal(16, admission.CachedTokens);
        admission.Cache.Dispose();
    }

    [Fact]
    public void Disabled_BehavesLikeFreshAllocation()
    {
        using var factory = CreateFactory();
        var cfg = new PrefixCacheConfig { Enabled = false };
        using var mgr = new PrefixTrieManager(factory, cfg);

        var prompt = MakePrompt(1, 16);
        SimulateSequence(mgr, prompt);

        var admission = mgr.Admit(prompt, MaxSeqLen);
        Assert.Equal(0, admission.CachedTokens);
        admission.Cache.Dispose();
    }

    [Fact]
    public void TryEvict_FreesUnreferencedBlocks()
    {
        using var factory = CreateFactory();
        using var mgr = new PrefixTrieManager(factory);

        SimulateSequence(mgr, MakePrompt(1, 16));

        int freeBefore = factory.Pool.FreeBlocks;
        int evicted = mgr.TryEvict(2);

        Assert.True(evicted >= 1, $"expected to evict at least one block, got {evicted}");
        Assert.True(factory.Pool.FreeBlocks > freeBefore);
    }

    [Fact]
    public void TryEvict_NoEligibleBlocks_IncrementsRefusal()
    {
        using var factory = CreateFactory();
        using var mgr = new PrefixTrieManager(factory);

        // Empty trie — no eviction possible.
        Assert.Equal(0, mgr.TryEvict(1));
        Assert.Equal(1, mgr.EvictionRefusals);
    }

    [Fact]
    public void NamedPrefix_PinAndUnpin()
    {
        using var factory = CreateFactory();
        using var mgr = new PrefixTrieManager(factory);

        var prompt = MakePrompt(1, 16);
        SimulateSequence(mgr, prompt);

        int matched = mgr.RegisterNamedPrefix("sys-A", prompt);
        Assert.Equal(16, matched);

        var info = mgr.InspectNamedPrefix("sys-A");
        Assert.NotNull(info);
        Assert.Equal(16, info!.Value.Tokens);

        Assert.True(mgr.UnpinNamedPrefix("sys-A"));
    }

    [Fact]
    public void Stats_ReflectActivity()
    {
        using var factory = CreateFactory();
        using var mgr = new PrefixTrieManager(factory);

        SimulateSequence(mgr, MakePrompt(1, 16));
        var admission = mgr.Admit(MakePrompt(1, 16), MaxSeqLen);
        admission.Cache.Dispose();

        var stats = mgr.GetStats();
        Assert.True(stats.Enabled);
        Assert.Equal(BlockSize, stats.BlockSize);
        Assert.True(stats.NodeCount > 0);
        Assert.Equal(1L, stats.Hits);
        Assert.True(stats.HitTokens >= 16);
    }

    [Fact]
    public void Refcounts_BalancedAcrossManyAdmissions()
    {
        // Regression: pool refcount math must balance — after N admit-complete
        // cycles, the pool's free-block count should match the trie's footprint.
        using var factory = CreateFactory();
        using var mgr = new PrefixTrieManager(factory);

        var prompt = MakePrompt(1, 16);
        for (int i = 0; i < 5; i++)
            SimulateSequence(mgr, prompt);

        var stats = mgr.GetStats();
        // 4 blocks in trie + free = total
        Assert.Equal(stats.TotalBlocks, stats.FreeBlocks + stats.NodeCount);
    }
}
