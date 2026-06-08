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

/// <summary>
/// End-to-end regression tests for the prefix-cache fixes in <see cref="TextGenerator"/>
/// (issue #121, items 2 and 6). Drives <c>Generate</c> against a mock model+tokenizer so the
/// resolve/allocate/store path is exercised without requiring real weights.
/// </summary>
public sealed class PrefixCacheIntegrationTests
{
    private const int VocabSize = 8;
    private const int NumLayers = 1;
    private const int NumKvHeads = 1;
    private const int HeadDim = 4;
    private const int MaxSeqLen = 64;

    /// <summary>
    /// Issue #121, item 2: a cache that fits the prompt but not the new max_tokens used to be
    /// returned as a hit, capping generation silently at the cached capacity. Now it must fall
    /// through to allocation, and the second call must generate up to its full max_tokens.
    /// </summary>
    [Fact]
    public void Resolve_LargerMaxTokens_ReallocatesWhenCacheTooSmall()
    {
        var model = new MockModel(MakeArgmaxLogits(token: 3));
        var tokenizer = new IdentityTokenizer();
        using var prefixCache = new PrefixCache(maxEntries: 4);

        // Honour the requested size — the cache cap comes from the per-request max_tokens, not the
        // factory. This way the second allocation is large enough; if the fix were missing, the
        // second request would reuse the smaller cache and stop early.
        int allocCount = 0;
        Func<ModelConfig, int, IKvCache> trackingFactory = (_, requested) =>
        {
            allocCount++;
            return new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, requested);
        };

        var generator = new TextGenerator(model, tokenizer,
            kvCacheFactory: trackingFactory, prefixCache: prefixCache);

        // First request: small max_tokens — cache sized to 5 prompt + 2 = 7. Stored in prefix cache.
        var first = generator.Generate("hello", new InferenceOptions { Temperature = 0f, MaxTokens = 2 });
        Assert.Equal(2, first.GeneratedTokenIds.Length);

        // Second request: same prompt, much larger max_tokens. Required cache = 5 + 20 = 25,
        // far exceeds the stored cache's MaxLength of 7. Before the fix, the
        // `entry.KvCache.MaxLength >= promptLen` clause accepted the small cache and decode ran out
        // at position 7, producing only 2 tokens (cacheSize - promptLen). After the fix, the
        // resolver falls through to allocation and the request completes fully.
        var second = generator.Generate("hello", new InferenceOptions { Temperature = 0f, MaxTokens = 20 });
        Assert.Equal(20, second.GeneratedTokenIds.Length);
        Assert.True(allocCount >= 2, $"expected reallocation on second call, allocCount={allocCount}");
    }

    /// <summary>
    /// Issue #121, item 6: the cache-miss branch used to return <c>ownsKvCache=false</c>, so an
    /// exception thrown between allocation and the eventual <c>StoreInPrefixCache</c> would leak
    /// the cache. Fix returns <c>true</c> and flips to false only after a successful store.
    /// Here we drive an exception via a throwing model and assert the freshly-allocated cache
    /// was disposed.
    /// </summary>
    [Fact]
    public void Resolve_GenerationThrows_DisposesFreshlyAllocatedCache()
    {
        int disposeCount = 0;
        Func<ModelConfig, int, IKvCache> trackingFactory = (_, size) =>
            new DisposeTrackingKvCache(
                new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, size),
                () => disposeCount++);

        var tokenizer = new IdentityTokenizer();
        using var prefixCache = new PrefixCache(maxEntries: 4);

        // Model throws on the very first forward pass — this happens after ResolveKvCache has
        // allocated but before StoreInPrefixCache transfers ownership, so the outer finally must
        // dispose the cache.
        var model = new ThrowingMockModel();
        var generator = new TextGenerator(model, tokenizer,
            kvCacheFactory: trackingFactory, prefixCache: prefixCache);

        Assert.Throws<InvalidOperationException>(() =>
            generator.Generate("hello", new InferenceOptions { Temperature = 0f, MaxTokens = 3 }));

        Assert.Equal(1, disposeCount);
    }

    // ── Helpers ──

    private static float[] MakeArgmaxLogits(int token)
    {
        var logits = new float[VocabSize];
        for (int i = 0; i < VocabSize; i++) logits[i] = -10f;
        logits[token] = 10f;
        return logits;
    }

    /// <summary>Trivial tokenizer: every character maps to its ordinal mod vocab.</summary>
    private sealed class IdentityTokenizer : ITokenizer
    {
        public int VocabSize => PrefixCacheIntegrationTests.VocabSize;
        public int BosTokenId => 0;
        public int EosTokenId => 7;

        public int[] Encode(string text)
        {
            var ids = new int[text.Length];
            for (int i = 0; i < text.Length; i++)
                ids[i] = text[i] % VocabSize;
            return ids;
        }

        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            Span<char> buf = stackalloc char[tokenIds.Length];
            for (int i = 0; i < tokenIds.Length; i++) buf[i] = (char)('a' + tokenIds[i]);
            return new string(buf);
        }

        public string DecodeToken(int tokenId) => ((char)('a' + tokenId)).ToString();
        public int CountTokens(string text) => text.Length;
    }

    /// <summary>Constant-logits mock model with KV-cache append.</summary>
    private sealed class MockModel : IModel
    {
        private readonly float[] _logits;

        public MockModel(float[] logits) => _logits = logits;

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
            long totalFloats = (long)batchSize * VocabSize;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);
            float* dst = (float*)ptr;
            for (int b = 0; b < batchSize; b++)
                _logits.AsSpan().CopyTo(new Span<float>(dst + b * VocabSize, VocabSize));

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

            return new UnmanagedTensor(new TensorShape(batchSize, VocabSize), DType.Float32, deviceId, ptr);
        }

        public void Dispose() { }
    }

    /// <summary>Model that always throws on Forward — to drive the cache-leak scenario.</summary>
    private sealed class ThrowingMockModel : IModel
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
            => throw new InvalidOperationException("forward failed");
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
            => throw new InvalidOperationException("forward failed");
        public void Dispose() { }
    }

    /// <summary>
    /// Delegating IKvCache that increments a counter on Dispose for leak detection.
    /// Not a <see cref="SimpleKvCache"/>, so <see cref="TextGenerator.SupportsPrefixReuse"/>
    /// returns false — the type-gate path means a successful generation would never reach
    /// `Store`, leaving `ownsKvCache=true` and ensuring the outer finally disposes either way.
    /// For the leak test (Forward throws), the same finally path runs and the counter ticks.
    /// </summary>
    private sealed class DisposeTrackingKvCache(IKvCache inner, Action onDispose) : IKvCache
    {
        private readonly IKvCache _inner = inner;
        private readonly Action _onDispose = onDispose;
        private bool _disposed;

        public int CurrentLength => _inner.CurrentLength;
        public int MaxLength => _inner.MaxLength;
        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
            => _inner.Update(keys, values, positions, layerIndex);
        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
            => _inner.Update(keys, values, positions, layerIndex);
        public ITensor GetKeys(int layerIndex) => _inner.GetKeys(layerIndex);
        public ITensor GetValues(int layerIndex) => _inner.GetValues(layerIndex);
        public TensorRef GetKeysRef(int layerIndex) => _inner.GetKeysRef(layerIndex);
        public TensorRef GetValuesRef(int layerIndex) => _inner.GetValuesRef(layerIndex);
        public void Rollback(int length) => _inner.Rollback(length);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _onDispose();
            _inner.Dispose();
        }
    }
}
