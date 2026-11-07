using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Tokenizers;
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

    [Fact]
    public void TextGenerator_WhenCachedKvCapacitySmallerThanRequiredSize_ReallocatesInsteadOfOverflowing()
    {
        using var cache = new PrefixCache(4);

        // Pre-populate prefix cache with a small KV cache (MaxLength = 8) that fits
        // the 4-token prompt [1, 2, 3, 4] but cannot fit promptLen (4) + maxTokens (10) = 14.
        var smallKv = new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, maxSeqLen: 8);
        smallKv.SetCurrentLength(3);
        cache.Store([1, 2, 3], smallKv);

        using var model = new StubModel();
        var tokenizer = new StubTokenizer([1, 2, 3, 4]);
        var generator = new TextGenerator(model, tokenizer, prefixCache: cache);

        var response = generator.Generate("test", new InferenceOptions
        {
            Temperature = 0f,
            MaxTokens = 10,
        });

        Assert.Equal(10, response.GeneratedTokenIds.Length);
        // The small KV cache's CurrentLength must not have been mutated before rejecting it
        Assert.Equal(3, smallKv.CurrentLength);
    }

    [Fact]
    public async Task TextGenerator_PrefillRequestsLastLogitOnly_InSyncAndStreamingPaths()
    {
        int[] prompt = [1, 10, 11, 12, 13];
        var model = new StubModel();
        var tokenizer = new StubTokenizer(prompt);
        var generator = new TextGenerator(model, tokenizer);
        var options = new DotLLM.Core.Configuration.InferenceOptions { MaxTokens = 2, Temperature = 0f };

        var syncRes = generator.Generate("prompt", options);
        Assert.Equal(2, syncRes.GeneratedTokenIds.Length);
        Assert.Contains(model.Calls, c => c.BatchSize == prompt.Length && c.LastLogitOnly);

        model.Calls.Clear();
        int streamedTokens = 0;
        await foreach (var _ in generator.GenerateStreamingTokensAsync("prompt", options))
            streamedTokens++;

        Assert.Equal(2, streamedTokens);
        Assert.Contains(model.Calls, c => c.BatchSize == prompt.Length && c.LastLogitOnly);
    }

    private sealed class StubModel : IModel
    {
        private const int VocabSize = 8;

        public List<(int BatchSize, bool LastLogitOnly)> Calls { get; } = [];

        public ModelConfig Config => new()
        {
            VocabSize = VocabSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumKvHeads,
            NumKvHeads = NumKvHeads,
            HiddenSize = HeadDim * NumKvHeads,
            IntermediateSize = HeadDim * 4,
            HeadDim = HeadDim,
            MaxSequenceLength = 64,
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null, lastLogitOnly: false);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache)
            => Forward(tokenIds, positions, deviceId, kvCache, lastLogitOnly: false);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache, bool lastLogitOnly)
        {
            int batchSize = tokenIds.Length;
            int logitRows = lastLogitOnly ? 1 : batchSize;
            Calls.Add((batchSize, lastLogitOnly));

            long totalFloats = (long)logitRows * VocabSize;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
            var logits = new Span<float>((void*)ptr, (int)totalFloats);
            logits.Fill(-10f);
            for (int b = 0; b < logitRows; b++)
                logits[b * VocabSize + 5] = 10f; // always pick token 5 (non-EOS)

            if (kvCache != null)
            {
                int kvStride = NumKvHeads * HeadDim;
                for (int layer = 0; layer < NumLayers; layer++)
                {
                    nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)(batchSize * kvStride * sizeof(float)), 64);
                    nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)(batchSize * kvStride * sizeof(float)), 64);
                    NativeMemory.Clear((void*)kPtr, (nuint)(batchSize * kvStride * sizeof(float)));
                    NativeMemory.Clear((void*)vPtr, (nuint)(batchSize * kvStride * sizeof(float)));

                    var kRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, kPtr);
                    var vRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, vPtr);
                    kvCache.Update(kRef, vRef, positions, layer);

                    NativeMemory.AlignedFree((void*)kPtr);
                    NativeMemory.AlignedFree((void*)vPtr);
                }
            }

            return new UnmanagedTensor(new TensorShape(logitRows, VocabSize), DType.Float32, deviceId, ptr);
        }

        public void Dispose() { }
    }

    private sealed class StubTokenizer(int[] encodedPrompt) : ITokenizer
    {
        public int VocabSize => 8;
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public int[] Encode(string text) => encodedPrompt;
        public string Decode(ReadOnlySpan<int> tokenIds) => new('x', tokenIds.Length);
        public string DecodeToken(int tokenId) => "x";
        public int CountTokens(string text) => encodedPrompt.Length;
    }
}
