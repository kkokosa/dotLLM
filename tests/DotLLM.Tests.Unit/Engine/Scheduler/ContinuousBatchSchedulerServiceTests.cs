using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Scheduler;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// Smoke tests for the async <see cref="ContinuousBatchSchedulerService"/> wrapper. Verifies that
/// the run-loop drives the inner step-based scheduler, that <see cref="ContinuousBatchSchedulerService.EnqueueAsync"/>
/// wakes the loop from idle, and that multiple concurrent enqueues all complete.
/// </summary>
public sealed class ContinuousBatchSchedulerServiceTests
{
    private const int VocabSize = 32;
    private const int NumLayers = 2;
    private const int NumKvHeads = 2;
    private const int HeadDim = 4;
    private const int BlockSize = 4;
    private const int MaxSeqLen = 64;
    private const int EosTokenId = 0;

    [Fact]
    public async Task EnqueueAsync_RunLoopDrivesCompletion()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 2);
        using var loopCts = new CancellationTokenSource();

        var loopTask = Task.Run(() => fix.Service.RunLoopAsync(loopCts.Token));

        var response = await fix.Service.EnqueueAsync(MakeRequest(promptLen: 3, maxTokens: 16));

        Assert.Equal(FinishReason.Stop, response.FinishReason);
        Assert.Equal(2, response.GeneratedTokenCount);

        loopCts.Cancel();
        await loopTask;
    }

    [Fact]
    public async Task ConcurrentEnqueue_AllComplete()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 3);
        using var loopCts = new CancellationTokenSource();
        var loopTask = Task.Run(() => fix.Service.RunLoopAsync(loopCts.Token));

        const int N = 6;
        var tasks = new Task<InferenceResponse>[N];
        for (int i = 0; i < N; i++)
            tasks[i] = fix.Service.EnqueueAsync(MakeRequest(promptLen: 2 + i, maxTokens: 16));

        var responses = await Task.WhenAll(tasks);
        Assert.All(responses, r =>
        {
            Assert.Equal(FinishReason.Stop, r.FinishReason);
            Assert.Equal(3, r.GeneratedTokenCount);
        });

        loopCts.Cancel();
        await loopTask;
    }

    [Fact]
    public async Task RunLoop_ParksWhenIdle_WokenByEnqueue()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 1);
        using var loopCts = new CancellationTokenSource();
        var loopTask = Task.Run(() => fix.Service.RunLoopAsync(loopCts.Token));

        // Give the loop a moment to reach the idle wait.
        await Task.Delay(50);

        var response = await fix.Service.EnqueueAsync(MakeRequest(promptLen: 3, maxTokens: 4));
        Assert.Equal(1, response.GeneratedTokenCount);

        loopCts.Cancel();
        await loopTask;
    }

    private static InferenceRequest MakeRequest(int promptLen, int maxTokens)
    {
        var tokens = new int[promptLen];
        for (int i = 0; i < promptLen; i++) tokens[i] = i + 1;
        return new InferenceRequest
        {
            TokenIds = tokens,
            Options = new InferenceOptions { Temperature = 0f, MaxTokens = maxTokens },
        };
    }

    private sealed class ServiceFixture : IDisposable
    {
        public PagedKvCacheFactory PagedFactory { get; }
        public MockModel Model { get; }
        public MockTokenizer Tokenizer { get; }
        public ContinuousBatchSchedulerService Service { get; }

        public ServiceFixture(int emitToken, int afterNTokens)
        {
            PagedFactory = new PagedKvCacheFactory(NumLayers, NumKvHeads, HeadDim, BlockSize, maxTotalTokens: 256);
            Model = new MockModel(emitToken, afterNTokens);
            Tokenizer = new MockTokenizer();
            Service = new ContinuousBatchSchedulerService(
                Model, Tokenizer,
                (_, maxSeq) => PagedFactory.Create(maxSeq),
                pagedPool: PagedFactory.Pool);
        }

        public void Dispose()
        {
            Service.Dispose();
            PagedFactory.Dispose();
            Model.Dispose();
        }
    }

    private sealed class MockModel : IModel
    {
        private readonly int _emitToken;
        private readonly int _afterNTokens;
        private readonly Dictionary<IKvCache, int> _steps = new(ReferenceEqualityComparer.Instance);
        private readonly object _stepLock = new();

        public MockModel(int emitToken, int afterNTokens)
        {
            _emitToken = emitToken;
            _afterNTokens = afterNTokens;
        }

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

            int step;
            lock (_stepLock)
            {
                if (kvCache is null) step = 0;
                else
                {
                    if (!_steps.TryGetValue(kvCache, out step)) step = 0;
                    _steps[kvCache] = step + 1;
                }
            }

            int emit = step < _afterNTokens ? 9 : _emitToken;
            if ((uint)emit >= VocabSize) emit = 1;

            long totalFloats = (long)batchSize * VocabSize;
            nint logitsPtr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
            float* dst = (float*)logitsPtr;
            for (int b = 0; b < batchSize; b++)
            {
                float* row = dst + (long)b * VocabSize;
                for (int v = 0; v < VocabSize; v++) row[v] = -10f;
                row[emit] = 10f;
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
        public int VocabSize => ContinuousBatchSchedulerServiceTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => ContinuousBatchSchedulerServiceTests.EosTokenId;
        public int[] Encode(string text) => Array.Empty<int>();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString();
        public int CountTokens(string text) => 0;
    }
}
