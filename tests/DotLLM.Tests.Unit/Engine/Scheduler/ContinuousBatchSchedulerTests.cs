using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Engine.Scheduler;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// Unit tests for <see cref="ContinuousBatchScheduler"/>. Uses a deterministic mock model that
/// emits a scripted token sequence per (prompt-last-token, step) so the test can verify
/// admission, decode iteration, eviction, and KV-cache release in isolation from any real model.
/// </summary>
public sealed class ContinuousBatchSchedulerTests
{
    private const int VocabSize = 32;
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int KvStride = NumKvHeads * HeadDim;
    private const int BlockSize = 4;
    private const int MaxSeqLen = 64;
    private const int EosTokenId = 0;

    [Fact]
    public void Submit_NewRequest_QueuedNotActive()
    {
        using var fix = new TestFixture();
        var req = MakeRequest(promptLen: 3, maxTokens: 4);

        var handle = fix.Scheduler.Submit(req);

        Assert.Equal(SequenceState.Queued, handle.State);
        Assert.False(fix.Scheduler.IsIdle);
        Assert.Equal(1, fix.Scheduler.QueueDepth);
        Assert.Equal(0, fix.Scheduler.ActiveCount);
    }

    [Fact]
    public async Task SingleSequence_RunsToCompletionOnEos()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 3)); // 3 normal tokens then EOS
        var req = MakeRequest(promptLen: 4, maxTokens: 16);

        var handle = fix.Scheduler.Submit(req);
        DriveUntilIdle(fix.Scheduler);

        var response = await handle.Completion;
        Assert.Equal(FinishReason.Stop, response.FinishReason);
        // 3 emitted tokens (EOS excluded per Stop semantics)
        Assert.Equal(3, response.GeneratedTokenCount);
        Assert.True(fix.Scheduler.IsIdle);
    }

    [Fact]
    public async Task FourSequences_DifferentPromptLengths_AllFinish()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 2));

        int[] promptLens = [2, 5, 7, 3];
        var handles = new ISchedulerRequest[promptLens.Length];

        for (int i = 0; i < promptLens.Length; i++)
            handles[i] = fix.Scheduler.Submit(MakeRequest(promptLens[i], maxTokens: 16));

        DriveUntilIdle(fix.Scheduler);

        for (int i = 0; i < handles.Length; i++)
        {
            var r = await handles[i].Completion;
            Assert.Equal(FinishReason.Stop, r.FinishReason);
            Assert.Equal(2, r.GeneratedTokenCount);
            Assert.Equal(promptLens[i], r.PromptTokenCount);
        }
        Assert.True(fix.Scheduler.IsIdle);
        Assert.Equal(0, fix.Scheduler.ActiveCount);
    }

    [Fact]
    public async Task Sequence_HitsMaxTokens_FinishReasonLength()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 7)); // emit 7 forever
        var req = MakeRequest(promptLen: 2, maxTokens: 5);

        var handle = fix.Scheduler.Submit(req);
        DriveUntilIdle(fix.Scheduler);

        var r = await handle.Completion;
        Assert.Equal(FinishReason.Length, r.FinishReason);
        Assert.Equal(5, r.GeneratedTokenCount);
    }

    [Fact]
    public async Task KvCacheBlocks_ReleasedAfterCompletion()
    {
        // Need 1 prompt token + 2 generated tokens = 3 positions → ceil(3/4) = 1 block
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 2));

        int totalBlocksBefore = fix.PagedPool.FreeBlocks;
        Assert.Equal(fix.PagedPool.TotalBlocks, totalBlocksBefore);

        for (int i = 0; i < 3; i++)
        {
            var handle = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 8));
            DriveUntilIdle(fix.Scheduler);
            await handle.Completion;
        }

        Assert.Equal(totalBlocksBefore, fix.PagedPool.FreeBlocks);
        Assert.True(fix.Scheduler.IsIdle);
    }

    [Fact]
    public void Backpressure_WhenCapacityExhausted_RequestStaysQueued()
    {
        // Cap to 1 active sequence. Submit 3 and verify only one runs at a time.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 5), // long-running, never EOS
            options: new ContinuousBatchSchedulerOptions { MaxActiveSequences = 1 });

        var h1 = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 4));
        var h2 = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 4));
        var h3 = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 4));

        // First Step: admit h1, prefill, no further admission (active==1 cap).
        fix.Scheduler.Step();
        Assert.Equal(1, fix.Scheduler.ActiveCount);
        Assert.Equal(2, fix.Scheduler.QueueDepth);

        // Drain h1 by stepping decode until it completes (max-tokens=4 → 4 decode steps).
        DriveUntilCompleted(fix.Scheduler, h1);
        Assert.True(h1.Completion.IsCompletedSuccessfully);

        // Now h2 can be admitted. Verify backpressure progresses.
        fix.Scheduler.Step();
        Assert.Equal(1, fix.Scheduler.ActiveCount);
        Assert.Equal(1, fix.Scheduler.QueueDepth);

        DriveUntilCompleted(fix.Scheduler, h2);
        DriveUntilCompleted(fix.Scheduler, h3);
        Assert.True(fix.Scheduler.IsIdle);
    }

    [Fact]
    public async Task Cancellation_BeforeAdmission_PropagatesToCompletion()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 5));

        using var cts = new CancellationTokenSource();
        var h = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 8), cts.Token);

        cts.Cancel();
        fix.Scheduler.Step(); // sweeps the cancelled queued request

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await h.Completion);
    }

    [Fact]
    public async Task Cancellation_DuringDecode_ReleasesKvBlocks()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 5));

        using var cts = new CancellationTokenSource();
        var h = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 100), cts.Token);

        // Admit + first decode.
        fix.Scheduler.Step();
        Assert.Equal(SequenceState.Decoding, h.State);
        int freeBeforeCancel = fix.PagedPool.FreeBlocks;

        cts.Cancel();
        fix.Scheduler.Step(); // sweeps cancellation

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await h.Completion);
        Assert.True(fix.Scheduler.IsIdle);
        Assert.True(fix.PagedPool.FreeBlocks >= freeBeforeCancel);
    }

    [Fact]
    public async Task ExplicitStopCondition_MaxTokensZeroBlock_Honored()
    {
        // Provide an explicit stop-condition list including MaxTokens(2) — should override
        // the request's MaxTokens.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Constant(tokenId: 9));

        var opts = new InferenceOptions
        {
            Temperature = 0f,
            MaxTokens = 100,
            StopConditions = [new MaxTokensStopCondition(2)],
        };
        var req = new InferenceRequest
        {
            TokenIds = new int[] { 1, 2 },
            Options = opts,
        };

        var h = fix.Scheduler.Submit(req);
        DriveUntilIdle(fix.Scheduler);

        var r = await h.Completion;
        Assert.Equal(2, r.GeneratedTokenCount);
        Assert.Equal(FinishReason.Length, r.FinishReason);
    }

    // ── Helpers ──

    private static void DriveUntilIdle(IBatchScheduler scheduler, int maxIterations = 1000)
    {
        for (int i = 0; i < maxIterations; i++)
        {
            if (scheduler.IsIdle) return;
            scheduler.Step();
        }
        Assert.Fail("Scheduler did not reach idle within iteration cap.");
    }

    private static void DriveUntilCompleted(IBatchScheduler scheduler, ISchedulerRequest handle, int maxIterations = 1000)
    {
        for (int i = 0; i < maxIterations; i++)
        {
            if (handle.Completion.IsCompleted) return;
            scheduler.Step();
        }
        Assert.Fail("Sequence did not complete within iteration cap.");
    }

    private static InferenceRequest MakeRequest(int promptLen, int maxTokens)
    {
        // Build prompt: avoid 0 (EOS) to keep things clean. Tokens 1..promptLen.
        var tokens = new int[promptLen];
        for (int i = 0; i < promptLen; i++) tokens[i] = i + 1;

        return new InferenceRequest
        {
            TokenIds = tokens,
            Options = new InferenceOptions { Temperature = 0f, MaxTokens = maxTokens },
        };
    }

    // Scripted token emission. A script returns the token to emit given the prefill step counter.
    private sealed class TokenScript
    {
        private readonly Func<int, int> _emit;
        public TokenScript(Func<int, int> emit) => _emit = emit;
        public int Emit(int step) => _emit(step);

        /// <summary>Emit <paramref name="tokenId"/> on every step.</summary>
        public static TokenScript Constant(int tokenId) => new(_ => tokenId);

        /// <summary>Emit <paramref name="afterToken"/> on the first <paramref name="afterNTokens"/> steps,
        /// then emit <paramref name="tokenId"/>.</summary>
        public static TokenScript Constant(int tokenId, int afterNTokens, int afterToken = 9)
            => new(step => step < afterNTokens ? afterToken : tokenId);
    }

    private sealed class TestFixture : IDisposable
    {
        public PagedKvCacheFactory PagedFactory { get; }
        public KvBlockPool PagedPool => PagedFactory.Pool;
        public MockModel Model { get; }
        public MockTokenizer Tokenizer { get; }
        public ContinuousBatchScheduler Scheduler { get; }

        public TestFixture(
            TokenScript? tokenScript = null,
            ContinuousBatchSchedulerOptions? options = null,
            int totalBlocks = 64)
        {
            tokenScript ??= TokenScript.Constant(EosTokenId, afterNTokens: 1);
            PagedFactory = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize,
                maxTotalTokens: totalBlocks * BlockSize);
            Model = new MockModel(tokenScript);
            Tokenizer = new MockTokenizer();
            Scheduler = new ContinuousBatchScheduler(
                Model,
                Tokenizer,
                (_, maxSeq) => PagedFactory.Create(maxSeq),
                options,
                pagedPool: PagedFactory.Pool);
        }

        public void Dispose()
        {
            Scheduler.Dispose();
            PagedFactory.Dispose();
            Model.Dispose();
        }
    }

    /// <summary>
    /// Deterministic model. Tracks a per-sequence decode-step counter using the KV-cache instance
    /// identity (each PagedKvCache is unique). Forward emits scripted logits so argmax = scripted token.
    /// </summary>
    private sealed class MockModel : IModel
    {
        private readonly TokenScript _script;
        private readonly Dictionary<IKvCache, int> _stepCounters = new(ReferenceEqualityComparer.Instance);

        public MockModel(TokenScript script) => _script = script;

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

            // Allocate logits [batchSize, VocabSize]
            long totalFloats = (long)batchSize * VocabSize;
            nint logitsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
            NativeMemory.Clear((void*)logitsPtr, (nuint)(totalFloats * sizeof(float)));

            // Determine current step for this sequence (prefill is step 0; decode increments).
            int step;
            if (kvCache is not null)
            {
                if (!_stepCounters.TryGetValue(kvCache, out step)) step = 0;
                _stepCounters[kvCache] = step + 1;
            }
            else
            {
                step = 0;
            }

            int emitToken = _script.Emit(step);
            if ((uint)emitToken >= VocabSize) emitToken = 1;

            float* dst = (float*)logitsPtr;
            for (int b = 0; b < batchSize; b++)
            {
                // Set argmax for the last position only; for prefill batches the scheduler reads
                // logitRows-1 anyway, so set the same argmax across the batch for safety.
                float* row = dst + (long)b * VocabSize;
                for (int v = 0; v < VocabSize; v++) row[v] = -10f;
                row[emitToken] = 10f;
            }

            // Update KV-cache (write zeros — the scheduler doesn't inspect content).
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
        public int VocabSize => ContinuousBatchSchedulerTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => ContinuousBatchSchedulerTests.EosTokenId;

        public int[] Encode(string text) => Array.Empty<int>();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString();
        public int CountTokens(string text) => 0;
    }
}
