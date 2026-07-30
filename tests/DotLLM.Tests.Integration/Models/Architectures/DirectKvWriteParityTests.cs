using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Parity tests for the direct-to-cache K/V write path (issue #25 item 4).
///
/// <para>
/// The optimisation lets the K and V projection GEMMs (and the subsequent in-place
/// AddBias / QK-norm / RoPE pipeline) write straight into the KV-cache slot via
/// <see cref="IKvCache.TryReserveSlot"/> / <see cref="IKvCache.CommitSlot"/>, skipping
/// the scratch buffer and the <c>Update</c> memcpy. This test exercises both paths
/// against the SmolLM-135M Q8_0 model and asserts byte-identical logits and KV-cache
/// state — proving the optimisation is a pure copy elimination with no math change.
/// </para>
///
/// <para>
/// The legacy path is forced via <see cref="LegacyUpdateOnlyCache"/>, a decorator
/// that intercepts <see cref="IKvCache.TryReserveSlot"/> and returns <c>false</c>,
/// pushing the caller back onto the <c>Update</c> branch in
/// <see cref="TransformerModel"/>.
/// </para>
/// </summary>
[Collection("SmallModel")]
public class DirectKvWriteParityTests
{
    private readonly SmallModelFixture _fixture;

    public DirectKvWriteParityTests(SmallModelFixture fixture)
    {
        _fixture = fixture;
    }

    private (TransformerModel model, GgufFile gguf, BpeTokenizer tokenizer) LoadModel()
    {
        var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        return (model, gguf, tokenizer);
    }

    /// <summary>
    /// Prefill parity: a single forward over an N-token prompt with both caches must
    /// produce byte-identical logits and byte-identical KV-cache buffers.
    /// </summary>
    [Fact]
    public void Prefill_DirectToCache_MatchesLegacyUpdate_BitExact()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int[] tokenIds = tokenizer.Encode("The capital of France is");
        int[] positions = new int[tokenIds.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        int cacheSize = tokenIds.Length + 8;

        using var directCache = new SimpleKvCache(
            model.Config.NumLayers, model.Config.NumKvHeads, model.Config.HeadDim, cacheSize);
        using var legacyInner = new SimpleKvCache(
            model.Config.NumLayers, model.Config.NumKvHeads, model.Config.HeadDim, cacheSize);
        using var legacyCache = new LegacyUpdateOnlyCache(legacyInner);

        using ITensor directLogits = model.Forward(tokenIds, positions, -1, directCache);
        using ITensor legacyLogits = model.Forward(tokenIds, positions, -1, legacyCache);

        AssertLogitsByteEqual(directLogits, legacyLogits);
        AssertKvCacheByteEqual(directCache, legacyInner, model.Config.NumLayers);
    }

    /// <summary>
    /// Decode parity: prefill + several single-token decode steps under both caches
    /// must produce byte-identical decode-step logits and byte-identical KV state.
    /// This is the case the optimisation primarily targets — every decode step would
    /// otherwise pay a kvStride * 4-byte memcpy per layer per token.
    /// </summary>
    [Fact]
    public void Decode_DirectToCache_MatchesLegacyUpdate_BitExact()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int[] promptIds = tokenizer.Encode("The capital of France is");
        int numDecodeSteps = 4;
        int cacheSize = promptIds.Length + numDecodeSteps;

        int[] positions = new int[cacheSize];
        for (int i = 0; i < cacheSize; i++) positions[i] = i;

        using var directCache = new SimpleKvCache(
            model.Config.NumLayers, model.Config.NumKvHeads, model.Config.HeadDim, cacheSize);
        using var legacyInner = new SimpleKvCache(
            model.Config.NumLayers, model.Config.NumKvHeads, model.Config.HeadDim, cacheSize);
        using var legacyCache = new LegacyUpdateOnlyCache(legacyInner);

        int vocabSize = model.Config.VocabSize;

        // Prefill both caches.
        int firstDirect, firstLegacy;
        using (ITensor d = model.Forward(promptIds, positions.AsSpan(0, promptIds.Length), -1, directCache))
        using (ITensor l = model.Forward(promptIds, positions.AsSpan(0, promptIds.Length), -1, legacyCache))
        {
            AssertLogitsByteEqual(d, l);
            firstDirect = ArgMaxLast(d, promptIds.Length, vocabSize);
            firstLegacy = ArgMaxLast(l, promptIds.Length, vocabSize);
            Assert.Equal(firstLegacy, firstDirect);
        }
        AssertKvCacheByteEqual(directCache, legacyInner, model.Config.NumLayers);

        int nextDirect = firstDirect;
        int nextLegacy = firstLegacy;

        // Decode steps: each step is a single-token forward at position prompt + step.
        // After every step both caches must be byte-identical and both logits buffers
        // must match exactly.
        for (int step = 0; step < numDecodeSteps - 1; step++)
        {
            int pos = promptIds.Length + step;
            using ITensor d = model.Forward([nextDirect], positions.AsSpan(pos, 1), -1, directCache);
            using ITensor l = model.Forward([nextLegacy], positions.AsSpan(pos, 1), -1, legacyCache);

            AssertLogitsByteEqual(d, l);
            AssertKvCacheByteEqual(directCache, legacyInner, model.Config.NumLayers);

            unsafe
            {
                nextDirect = ArgMax(new ReadOnlySpan<float>((void*)d.DataPointer, vocabSize));
                nextLegacy = ArgMax(new ReadOnlySpan<float>((void*)l.DataPointer, vocabSize));
            }
            Assert.Equal(nextLegacy, nextDirect);
        }
    }

    /// <summary>
    /// Confirms TryReserveSlot is actually exercised end-to-end. If wiring regresses
    /// and the model never calls TryReserveSlot, the parity test would still pass
    /// trivially (Update would run on both paths). This counter ensures we actually
    /// took the direct-to-cache branch.
    /// </summary>
    [Fact]
    public void TryReserveSlot_IsActuallyCalled_FromTransformerModel()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int[] promptIds = tokenizer.Encode("Hello");
        int[] positions = new int[promptIds.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        using var inner = new SimpleKvCache(
            model.Config.NumLayers, model.Config.NumKvHeads, model.Config.HeadDim, promptIds.Length + 1);
        using var counting = new CountingCache(inner);

        using var _logits = model.Forward(promptIds, positions, -1, counting);

        // SimpleKvCache reserves contiguous positions starting at 0 — must succeed
        // for every layer of the prefill.
        Assert.Equal(model.Config.NumLayers, counting.TryReserveSucceededCount);
        Assert.Equal(0, counting.UpdateCallCount);
        Assert.Equal(model.Config.NumLayers, counting.CommitSlotCount);
    }

    // ── Helpers ────────────────────────────────────────────────────────

    private static unsafe void AssertLogitsByteEqual(ITensor a, ITensor b)
    {
        Assert.Equal(a.ElementCount, b.ElementCount);
        int bytes = (int)a.ElementCount * sizeof(float);
        var sa = new ReadOnlySpan<byte>((void*)a.DataPointer, bytes);
        var sb = new ReadOnlySpan<byte>((void*)b.DataPointer, bytes);
        Assert.True(sa.SequenceEqual(sb), "Logits must be byte-identical between direct-to-cache and legacy paths.");
    }

    private static unsafe void AssertKvCacheByteEqual(SimpleKvCache a, SimpleKvCache b, int numLayers)
    {
        Assert.Equal(a.CurrentLength, b.CurrentLength);
        // GetKeysRef returns a TensorRef of shape [CurrentLength, kvStride] — use Dim1
        // for the per-row width rather than referencing internal fields.
        var probe = a.GetKeysRef(0);
        int floatsPerLayer = probe.Dim0 * probe.Dim1;
        int bytesPerLayer = floatsPerLayer * sizeof(float);
        for (int layer = 0; layer < numLayers; layer++)
        {
            var refA = a.GetKeysRef(layer);
            var refB = b.GetKeysRef(layer);
            var refAv = a.GetValuesRef(layer);
            var refBv = b.GetValuesRef(layer);

            var ka = new ReadOnlySpan<byte>((void*)refA.DataPointer, bytesPerLayer);
            var kb = new ReadOnlySpan<byte>((void*)refB.DataPointer, bytesPerLayer);
            var va = new ReadOnlySpan<byte>((void*)refAv.DataPointer, bytesPerLayer);
            var vb = new ReadOnlySpan<byte>((void*)refBv.DataPointer, bytesPerLayer);

            Assert.True(ka.SequenceEqual(kb), $"Layer {layer} K buffer must be byte-identical.");
            Assert.True(va.SequenceEqual(vb), $"Layer {layer} V buffer must be byte-identical.");
        }
    }

    private static unsafe int ArgMaxLast(ITensor logits, int seqLen, int vocabSize)
    {
        float* ptr = (float*)(logits.DataPointer + (long)(seqLen - 1) * vocabSize * sizeof(float));
        return ArgMax(new ReadOnlySpan<float>(ptr, vocabSize));
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        float bestVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > bestVal)
            {
                bestVal = values[i];
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// IKvCache decorator that forces the legacy <see cref="IKvCache.Update"/> path by
    /// short-circuiting <see cref="IKvCache.TryReserveSlot"/> to <c>false</c>. All other
    /// operations delegate to the wrapped cache. Used to compare the direct-to-cache
    /// optimisation against the pre-#278 baseline behaviour on identical state.
    /// </summary>
    private sealed class LegacyUpdateOnlyCache : IKvCache
    {
        private readonly IKvCache _inner;
        public LegacyUpdateOnlyCache(IKvCache inner) => _inner = inner;
        public int CurrentLength => _inner.CurrentLength;
        public int MaxLength => _inner.MaxLength;
        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex) =>
            _inner.Update(keys, values, positions, layerIndex);
        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex) =>
            _inner.Update(keys, values, positions, layerIndex);
        public ITensor GetKeys(int layerIndex) => _inner.GetKeys(layerIndex);
        public ITensor GetValues(int layerIndex) => _inner.GetValues(layerIndex);
        public TensorRef GetKeysRef(int layerIndex) => _inner.GetKeysRef(layerIndex);
        public TensorRef GetValuesRef(int layerIndex) => _inner.GetValuesRef(layerIndex);
        public void Rollback(int length) => _inner.Rollback(length);
        public bool TryReserveSlot(int layerIndex, ReadOnlySpan<int> positions, out Span<float> kDst, out Span<float> vDst)
        {
            kDst = default;
            vDst = default;
            return false; // force the legacy Update branch
        }
        public void CommitSlot(int layerIndex, ReadOnlySpan<int> positions) { /* never called when TryReserveSlot returns false */ }
        public void Dispose() { /* outer test owns inner */ }
    }

    /// <summary>
    /// IKvCache decorator that counts TryReserveSlot success vs Update fallback, used
    /// to assert that the direct-to-cache branch is actually being exercised end-to-end
    /// (so a parity test passing trivially can't mask a wiring regression).
    /// </summary>
    private sealed class CountingCache : IKvCache
    {
        private readonly IKvCache _inner;
        public int TryReserveSucceededCount { get; private set; }
        public int UpdateCallCount { get; private set; }
        public int CommitSlotCount { get; private set; }
        public CountingCache(IKvCache inner) => _inner = inner;
        public int CurrentLength => _inner.CurrentLength;
        public int MaxLength => _inner.MaxLength;
        public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        {
            UpdateCallCount++;
            _inner.Update(keys, values, positions, layerIndex);
        }
        public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        {
            UpdateCallCount++;
            _inner.Update(keys, values, positions, layerIndex);
        }
        public ITensor GetKeys(int layerIndex) => _inner.GetKeys(layerIndex);
        public ITensor GetValues(int layerIndex) => _inner.GetValues(layerIndex);
        public TensorRef GetKeysRef(int layerIndex) => _inner.GetKeysRef(layerIndex);
        public TensorRef GetValuesRef(int layerIndex) => _inner.GetValuesRef(layerIndex);
        public void Rollback(int length) => _inner.Rollback(length);
        public bool TryReserveSlot(int layerIndex, ReadOnlySpan<int> positions, out Span<float> kDst, out Span<float> vDst)
        {
            bool ok = _inner.TryReserveSlot(layerIndex, positions, out kDst, out vDst);
            if (ok) TryReserveSucceededCount++;
            return ok;
        }
        public void CommitSlot(int layerIndex, ReadOnlySpan<int> positions)
        {
            CommitSlotCount++;
            _inner.CommitSlot(layerIndex, positions);
        }
        public void Dispose() { /* outer test owns inner */ }
    }
}
