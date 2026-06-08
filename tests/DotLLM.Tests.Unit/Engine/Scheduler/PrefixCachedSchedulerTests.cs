using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Engine.Scheduler;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// Tests for the cross-request prefix cache wired into <see cref="ContinuousBatchScheduler"/>.
/// Uses a deterministic mock model so the test can assert exactly how much prefill compute
/// happened per request.
/// </summary>
public sealed class PrefixCachedSchedulerTests
{
    private const int VocabSize = 32;
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int BlockSize = 16;
    private const int MaxSeqLen = 512;
    private const int EosTokenId = 0;

    [Fact]
    public async Task FourConcurrentSequences_SharedPrompt_PrefillCounts()
    {
        // Acceptance criterion: 4 admissions of a shared 256-token system prompt
        // should drive PREFILL compute only on the FIRST admission; the next 3
        // should see the prompt reused from the trie (zero prefilled-tokens-of-prompt).
        using var fix = new Fixture(totalBlocks: 256);
        var prompt = MakePrompt(seed: 7, count: 256);

        // 1st request — populates the trie with 256/16 = 16 blocks of prompt KV.
        var h1 = fix.Scheduler.Submit(MakeRequest(prompt, maxTokens: 1));
        DriveUntilIdle(fix.Scheduler);
        await h1.Completion;

        long prefilledAfterFirst = fix.Scheduler.PrefilledPromptTokens;
        long cachedAfterFirst = fix.Scheduler.CachedPromptTokens;
        Assert.Equal(256, prefilledAfterFirst);
        Assert.Equal(0, cachedAfterFirst);

        // Next 3 requests — should hit the trie and prefill no prompt tokens.
        for (int i = 0; i < 3; i++)
        {
            var h = fix.Scheduler.Submit(MakeRequest(prompt, maxTokens: 1));
            DriveUntilIdle(fix.Scheduler);
            await h.Completion;
        }

        // Total prefilled prompt tokens did NOT grow beyond the first request.
        Assert.Equal(prefilledAfterFirst, fix.Scheduler.PrefilledPromptTokens);
        Assert.Equal(256 * 3, fix.Scheduler.CachedPromptTokens);
    }

    [Fact]
    public async Task BitIdenticalOutput_CachedVsFreshPath()
    {
        // Same prompt, same scripted token emission → output tokens must match
        // whether or not the prefix cache served the prefill.

        // Run A: fresh path (no manager).
        using var fixA = new Fixture(usePrefixCache: false);
        var prompt = MakePrompt(seed: 13, count: 32);
        var hA = fixA.Scheduler.Submit(MakeRequest(prompt, maxTokens: 4));
        DriveUntilIdle(fixA.Scheduler);
        var rA = await hA.Completion;

        // Run B: cached path — same prompt twice.
        using var fixB = new Fixture(usePrefixCache: true);
        var hB1 = fixB.Scheduler.Submit(MakeRequest(prompt, maxTokens: 4));
        DriveUntilIdle(fixB.Scheduler);
        await hB1.Completion;
        var hB2 = fixB.Scheduler.Submit(MakeRequest(prompt, maxTokens: 4));
        DriveUntilIdle(fixB.Scheduler);
        var rB2 = await hB2.Completion;

        Assert.Equal(rA.GeneratedTokenIds, rB2.GeneratedTokenIds);
    }

    [Fact]
    public async Task EvictionEnabled_RecoversBlocksUnderPressure()
    {
        // Configure a tiny pool: 32 blocks total. After 1st sequence stores 16 blocks
        // and completes, the next sequence with a 16-block prompt should fit only
        // if the trie evicts (or backpressure waits).
        using var fix = new Fixture(
            totalBlocks: 32,
            options: new ContinuousBatchSchedulerOptions
            {
                MaxActiveSequences = 1,
                ReserveBlocksPerSequence = 24,
            });

        var promptA = MakePrompt(1, 256);
        var promptB = MakePrompt(2, 256);

        var hA = fix.Scheduler.Submit(MakeRequest(promptA, maxTokens: 1));
        DriveUntilIdle(fix.Scheduler);
        await hA.Completion;

        // After A completes, A's blocks are in the trie (refcount=0, evictable).
        // B has a different prompt — admission should evict A's blocks to make room.
        int beforeFree = fix.PagedPool.FreeBlocks;
        var hB = fix.Scheduler.Submit(MakeRequest(promptB, maxTokens: 1));
        DriveUntilIdle(fix.Scheduler);
        await hB.Completion;

        Assert.True(fix.PagedPool.FreeBlocks > 0,
            $"Pool should have free blocks after B completes (was {beforeFree} before B).");
    }

    [Fact]
    public async Task Stats_TrackCachedTokens()
    {
        using var fix = new Fixture();
        var prompt = MakePrompt(1, 32);

        var h1 = fix.Scheduler.Submit(MakeRequest(prompt, maxTokens: 1));
        DriveUntilIdle(fix.Scheduler);
        await h1.Completion;

        Assert.NotNull(fix.PrefixCache);
        var stats = fix.PrefixCache!.GetStats();
        Assert.True(stats.NodeCount >= 2);
        Assert.Equal(0L, stats.Hits); // first call is a miss
        Assert.Equal(1L, stats.Misses);

        var h2 = fix.Scheduler.Submit(MakeRequest(prompt, maxTokens: 1));
        DriveUntilIdle(fix.Scheduler);
        await h2.Completion;

        stats = fix.PrefixCache.GetStats();
        Assert.Equal(1L, stats.Hits);
        Assert.True(stats.HitTokens >= 32);
    }

    // ── Helpers ──

    private static void DriveUntilIdle(IBatchScheduler scheduler, int maxIterations = 2000)
    {
        for (int i = 0; i < maxIterations; i++)
        {
            if (scheduler.IsIdle) return;
            scheduler.Step();
        }
        Assert.Fail("Scheduler did not reach idle within iteration cap.");
    }

    private static int[] MakePrompt(int seed, int count)
    {
        var p = new int[count];
        for (int i = 0; i < count; i++) p[i] = seed * 1000 + i + 1;
        return p;
    }

    private static InferenceRequest MakeRequest(int[] prompt, int maxTokens) => new()
    {
        TokenIds = prompt,
        Options = new InferenceOptions { Temperature = 0f, MaxTokens = maxTokens },
    };

    private sealed class Fixture : IDisposable
    {
        public PagedKvCacheFactory PagedFactory { get; }
        public KvBlockPool PagedPool => PagedFactory.Pool;
        public PrefixTrieManager? PrefixCache { get; }
        public MockModel Model { get; }
        public MockTokenizer Tokenizer { get; }
        public ContinuousBatchScheduler Scheduler { get; }

        public Fixture(int totalBlocks = 64, bool usePrefixCache = true,
            ContinuousBatchSchedulerOptions? options = null)
        {
            PagedFactory = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize,
                maxTotalTokens: totalBlocks * BlockSize);
            PrefixCache = usePrefixCache ? new PrefixTrieManager(PagedFactory) : null;
            Model = new MockModel();
            Tokenizer = new MockTokenizer();
            Scheduler = new ContinuousBatchScheduler(
                Model,
                Tokenizer,
                (_, maxSeq) => PagedFactory.Create(maxSeq),
                options,
                pagedPool: PagedFactory.Pool,
                prefixCache: PrefixCache);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            PrefixCache?.Dispose();
            PagedFactory.Dispose();
            Model.Dispose();
        }
    }

    private sealed class MockModel : IModel
    {
        public ModelConfig Config => new()
        {
            VocabSize = VocabSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumKvHeads,
            NumKvHeads = NumKvHeads,
            HiddenSize = HeadDim * NumKvHeads,
            IntermediateSize = HeadDim * 4,
            HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen,
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache)
        {
            int batchSize = tokenIds.Length;
            int lastPos = positions.Length > 0 ? positions[^1] : 0;
            int emitToken = 1 + (lastPos % (VocabSize - 2)); // deterministic, never EOS

            long totalFloats = (long)batchSize * VocabSize;
            nint logitsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
            NativeMemory.Clear((void*)logitsPtr, (nuint)(totalFloats * sizeof(float)));

            float* dst = (float*)logitsPtr;
            for (int b = 0; b < batchSize; b++)
            {
                float* row = dst + (long)b * VocabSize;
                for (int v = 0; v < VocabSize; v++) row[v] = -10f;
                row[emitToken] = 10f;
            }

            if (kvCache is not null)
            {
                int kvStride = NumKvHeads * HeadDim;
                long kvBytes = (long)batchSize * kvStride * sizeof(float);
                nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
                NativeMemory.Clear((void*)kPtr, (nuint)kvBytes);
                NativeMemory.Clear((void*)vPtr, (nuint)kvBytes);
                try
                {
                    for (int layer = 0; layer < NumLayers; layer++)
                    {
                        var kRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, kPtr);
                        var vRef = new TensorRef(batchSize, kvStride, DType.Float32, -1, vPtr);
                        kvCache.Update(kRef, vRef, positions, layer);
                    }
                }
                finally
                {
                    NativeMemory.AlignedFree((void*)kPtr);
                    NativeMemory.AlignedFree((void*)vPtr);
                }
            }

            return new UnmanagedTensor(new TensorShape(batchSize, VocabSize), DType.Float32, deviceId, logitsPtr);
        }

        public void Dispose() { }
    }

    private sealed class MockTokenizer : ITokenizer
    {
        public int VocabSize => PrefixCachedSchedulerTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => PrefixCachedSchedulerTests.EosTokenId;

        public int[] Encode(string text) => Array.Empty<int>();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString();
        public int CountTokens(string text) => 0;
    }
}
