using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Constraints;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Tests for <see cref="SpeculativeDecoder"/>.
/// Uses mock models that return predetermined logits to verify the draft-verify-accept algorithm.
/// </summary>
public sealed class SpeculativeDecoderTests
{
    private const int VocabSize = 8;
    private const int NumLayers = 1;
    private const int NumKvHeads = 1;
    private const int HeadDim = 4;

    /// <summary>
    /// Non-greedy speculative decoding is supported once <c>q</c> and <c>p</c> are computed
    /// from the post-transform distribution (Wave 8 / issue #121). The constructor accepts
    /// <c>greedy: false</c>; behaviour is driven by the pipeline's mode at call time.
    /// </summary>
    [Fact]
    public void Constructor_AcceptsNonGreedy()
    {
        var decoder = new SpeculativeDecoder(greedy: false, seed: 42);
        Assert.NotNull(decoder);
    }

    /// <summary>
    /// When draft and target models agree on argmax at temperature=0 (greedy),
    /// all K tokens should be accepted + 1 bonus = K+1 tokens.
    /// </summary>
    [Fact]
    public void Greedy_AllAccepted_ReturnsKPlusOne()
    {
        // Both models always produce token 3 as argmax
        float[] logits = MakeLogits(argmaxToken: 3);
        var target = new MockModel(logits, VocabSize);
        var draft = new MockModel(logits, VocabSize);

        int k = 3;
        var decoder = new SpeculativeDecoder(greedy: true, seed: 42);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var generatedIds = new List<int> { 1 }; // seed token

        using var targetCache = CreateCache();
        using var draftCache = CreateCache();

        // Prefill one position for the seed token
        PrefillCache(targetCache, target);
        PrefillCache(draftCache, draft);

        Span<int> outputBuffer = stackalloc int[k + 1];
        var result = decoder.DraftAndVerify(
            target, draft, targetCache, draftCache,
            pipeline, generatedIds, constraint: null,
            position: 1, targetVocabSize: VocabSize, draftVocabSize: VocabSize, numCandidates: k,
            outputBuffer: outputBuffer);

        Assert.Equal(k + 1, result.AcceptedCount);
        for (int i = 0; i < result.AcceptedCount; i++)
            Assert.Equal(3, outputBuffer[i]);
        Assert.Equal(k, result.DraftedCount);
    }

    /// <summary>
    /// When draft proposes a different token than target argmax (greedy),
    /// the first token should be rejected and replaced with the target's argmax.
    /// Result: exactly 1 corrected token.
    /// </summary>
    [Fact]
    public void Greedy_FirstRejected_ReturnsCorrectedToken()
    {
        // Draft always picks token 2, target always picks token 5
        float[] draftLogits = MakeLogits(argmaxToken: 2);
        float[] targetLogits = MakeLogits(argmaxToken: 5);
        var target = new MockModel(targetLogits, VocabSize);
        var draft = new MockModel(draftLogits, VocabSize);

        int k = 3;
        var decoder = new SpeculativeDecoder(greedy: true, seed: 42);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var generatedIds = new List<int> { 1 };

        using var targetCache = CreateCache();
        using var draftCache = CreateCache();
        PrefillCache(targetCache, target);
        PrefillCache(draftCache, draft);

        Span<int> outputBuffer = stackalloc int[k + 1];
        var result = decoder.DraftAndVerify(
            target, draft, targetCache, draftCache,
            pipeline, generatedIds, constraint: null,
            position: 1, targetVocabSize: VocabSize, draftVocabSize: VocabSize, numCandidates: k,
            outputBuffer: outputBuffer);

        // First draft token is rejected, corrected with target argmax
        Assert.Equal(1, result.AcceptedCount);
        Assert.Equal(5, outputBuffer[0]);
        Assert.Equal(k, result.DraftedCount);
    }

    /// <summary>
    /// KV-cache should be rolled back to position + acceptedCount after rejection.
    /// </summary>
    [Fact]
    public void KvCache_RolledBackOnRejection()
    {
        float[] draftLogits = MakeLogits(argmaxToken: 2);
        float[] targetLogits = MakeLogits(argmaxToken: 5);
        var target = new MockModel(targetLogits, VocabSize);
        var draft = new MockModel(draftLogits, VocabSize);

        int k = 3;
        var decoder = new SpeculativeDecoder(greedy: true, seed: 42);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var generatedIds = new List<int> { 1 };

        using var targetCache = CreateCache();
        using var draftCache = CreateCache();
        PrefillCache(targetCache, target);
        PrefillCache(draftCache, draft);

        int positionBefore = 1;
        Span<int> outputBuffer = stackalloc int[k + 1];
        var result = decoder.DraftAndVerify(
            target, draft, targetCache, draftCache,
            pipeline, generatedIds, constraint: null,
            position: positionBefore, targetVocabSize: VocabSize, draftVocabSize: VocabSize, numCandidates: k,
            outputBuffer: outputBuffer);

        // After rejection with 1 accepted token, caches should be at position + 1
        Assert.Equal(positionBefore + result.AcceptedCount, targetCache.CurrentLength);
        Assert.Equal(positionBefore + result.AcceptedCount, draftCache.CurrentLength);
    }

    /// <summary>
    /// Speculative result records correct draft and verify timing ticks.
    /// </summary>
    [Fact]
    public void TimingTicks_AreRecorded()
    {
        float[] logits = MakeLogits(argmaxToken: 3);
        var target = new MockModel(logits, VocabSize);
        var draft = new MockModel(logits, VocabSize);

        var decoder = new SpeculativeDecoder(greedy: true, seed: 42);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var generatedIds = new List<int> { 1 };

        using var targetCache = CreateCache();
        using var draftCache = CreateCache();
        PrefillCache(targetCache, target);
        PrefillCache(draftCache, draft);

        Span<int> outputBuffer = stackalloc int[4];
        var result = decoder.DraftAndVerify(
            target, draft, targetCache, draftCache,
            pipeline, generatedIds, constraint: null,
            position: 1, targetVocabSize: VocabSize, draftVocabSize: VocabSize, numCandidates: 3,
            outputBuffer: outputBuffer);

        Assert.True(result.DraftTicks > 0);
        Assert.True(result.VerifyTicks > 0);
    }

    /// <summary>
    /// When K=0 (clamped due to cache limits), returns default empty result.
    /// </summary>
    [Fact]
    public void ZeroCandidates_ReturnsEmptyResult()
    {
        float[] logits = MakeLogits(argmaxToken: 3);
        var target = new MockModel(logits, VocabSize);
        var draft = new MockModel(logits, VocabSize);

        var decoder = new SpeculativeDecoder(greedy: true, seed: 42);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var generatedIds = new List<int> { 1 };

        using var targetCache = CreateCache();
        using var draftCache = CreateCache();
        PrefillCache(targetCache, target);
        PrefillCache(draftCache, draft);

        Span<int> outputBuffer = stackalloc int[1];
        var result = decoder.DraftAndVerify(
            target, draft, targetCache, draftCache,
            pipeline, generatedIds, constraint: null,
            position: 1, targetVocabSize: VocabSize, draftVocabSize: VocabSize, numCandidates: 0,
            outputBuffer: outputBuffer);

        Assert.Equal(0, result.AcceptedCount);
    }

    /// <summary>
    /// Non-greedy distributional correctness (Wave 8 / issue #121).
    ///
    /// The modified rejection sampling scheme produces samples from the target distribution only
    /// when <c>q</c> and <c>p</c> are computed from the post-transform distribution the pipeline
    /// actually samples from. The previous implementation used raw softmax for both, while sampling
    /// the draft token via the full pipeline — so the empirical output distribution diverged from
    /// the target whenever the pipeline transforms (here: temperature ≠ 1) shifted probability mass.
    ///
    /// Test design — draft and target have reversed logits so the buggy path frequently rejects
    /// and corrects to the wrong token. Across N trials, the speculative output token frequencies
    /// must match what the target-pipeline-alone produces (Pearson chi-squared, 99.9% CI).
    /// Verified to FAIL on the unfixed implementation when only the constructor guard is removed
    /// — see PR notes.
    /// </summary>
    [Fact]
    public void NonGreedy_OutputDistributionMatchesTargetPipeline()
    {
        // Strongly skewed, reversed distributions amplify the bug. Vocab=6 so TopK=3 actually
        // prunes tokens — exercising the top-k branch of the transform chain, not just temperature.
        // RepetitionPenalty != 1 with generatedIds = {1} ensures the processor chain runs too.
        float[] targetLogits = [5f, 3f, 1f, -1f, -3f, -5f];
        float[] draftLogits = [-5f, -3f, -1f, 1f, 3f, 5f];
        int vocabSize = targetLogits.Length;
        const int N = 4000;
        const float temperature = 0.5f;
        const int topK = 3;
        const float repetitionPenalty = 1.3f;

        InferenceOptions OptsFor(int seed) => new()
        {
            Temperature = temperature,
            TopK = topK,
            RepetitionPenalty = repetitionPenalty,
            Seed = seed,
        };

        // ── Reference: sample directly from the target pipeline N times ──
        var referenceCounts = new int[vocabSize];
        for (int trial = 0; trial < N; trial++)
        {
            var refPipeline = new SamplerPipeline(OptsFor(trial));
            var logits = (float[])targetLogits.Clone();
            int sampled = refPipeline.Sample(logits, new List<int> { 1 });
            referenceCounts[sampled]++;
        }

        // ── Spec decoder: take the first accepted/corrected token from each trial ──
        var specCounts = new int[vocabSize];
        var outputBufferArr = new int[3];
        for (int trial = 0; trial < N; trial++)
        {
            var target = new MockModel(targetLogits, vocabSize);
            var draft = new MockModel(draftLogits, vocabSize);
            // Distinct seeds for pipeline and decoder so rejection-sampling RNG is decoupled
            // from draft-sampling RNG — mirrors production where both share the InferenceOptions
            // seed lineage but draw from separate Random instances.
            var pipeline = new SamplerPipeline(OptsFor(trial));
            var decoder = new SpeculativeDecoder(greedy: false, seed: trial ^ 0x5A5A_5A5A);
            var generatedIds = new List<int> { 1 };

            using var targetCache = CreateCache();
            using var draftCache = CreateCache();
            PrefillCache(targetCache, target);
            PrefillCache(draftCache, draft);

            var result = decoder.DraftAndVerify(
                target, draft, targetCache, draftCache,
                pipeline, generatedIds, constraint: null,
                position: 1, targetVocabSize: vocabSize, draftVocabSize: vocabSize, numCandidates: 2,
                outputBuffer: outputBufferArr);

            Assert.True(result.AcceptedCount > 0, "spec decoder must emit at least one token");
            int firstToken = outputBufferArr[0];
            specCounts[firstToken]++;
        }

        // ── Pearson chi-squared on observed (spec) vs expected (reference) ──
        double chiSq = 0.0;
        int dof = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            double expected = referenceCounts[i];
            if (expected < 5) continue; // standard chi-squared validity cutoff
            double diff = specCounts[i] - expected;
            chiSq += diff * diff / expected;
            dof++;
        }
        dof = Math.Max(1, dof - 1);

        // Critical value at α = 0.001 covering up to 6 dof (22.46). The bug produces chi-squared
        // in the hundreds on this fixture; well-separated.
        const double criticalChiSq = 22.46;
        Assert.True(chiSq < criticalChiSq,
            $"empirical distribution diverges from target pipeline: chiSq={chiSq:F2} dof={dof} " +
            $"ref=[{string.Join(",", referenceCounts)}] spec=[{string.Join(",", specCounts)}]");
    }

    /// <summary>
    /// Cross-position regression guard for the repetition-penalty context rebuild in the verify
    /// pass (issue #121). With identical draft/target models, <c>q == p</c> at every position;
    /// non-greedy acceptance therefore must accept ALL K draft tokens plus the bonus token, every
    /// seed. A bug that fed the wrong rep-penalty context to the verify pass (e.g. always
    /// `generatedIds` without the draft prefix, like the original code did at lines 142-143
    /// of `SpeculativeDecoder` before the fix) would shift <c>p</c> away from <c>q</c> for
    /// positions ≥ 1 and reduce the accepted count below k+1.
    /// </summary>
    [Fact]
    public void NonGreedy_IdenticalModelsWithRepetitionPenalty_AcceptAllKPlusOne()
    {
        // Distribution skewed enough that the rep-penalty-favoured token can shift under context
        // changes — exercising the cross-position context invariant.
        float[] logits = [4f, 2f, 0f, -2f];
        int vocabSize = logits.Length;
        const int N = 50;
        const int k = 3;
        var outputBufferArr = new int[k + 1];

        for (int trial = 0; trial < N; trial++)
        {
            var target = new MockModel(logits, vocabSize);
            var draft = new MockModel(logits, vocabSize);
            var pipeline = new SamplerPipeline(new InferenceOptions
            {
                Temperature = 0.7f,
                RepetitionPenalty = 1.3f,
                Seed = trial,
            });
            var decoder = new SpeculativeDecoder(greedy: false, seed: trial ^ 0x5A5A_5A5A);
            var generatedIds = new List<int> { 1 };

            using var targetCache = CreateCache();
            using var draftCache = CreateCache();
            PrefillCache(targetCache, target);
            PrefillCache(draftCache, draft);

            var result = decoder.DraftAndVerify(
                target, draft, targetCache, draftCache,
                pipeline, generatedIds, constraint: null,
                position: 1, targetVocabSize: vocabSize, draftVocabSize: vocabSize, numCandidates: k,
                outputBuffer: outputBufferArr);

            Assert.Equal(k + 1, result.AcceptedCount);
        }
    }

    // ── Helpers ──

    private static float[] MakeLogits(int argmaxToken)
    {
        var logits = new float[VocabSize];
        for (int i = 0; i < VocabSize; i++)
            logits[i] = -10f;
        logits[argmaxToken] = 10f;
        return logits;
    }

    private static SimpleKvCache CreateCache() =>
        new(NumLayers, NumKvHeads, HeadDim, maxSeqLen: 64);

    private static void PrefillCache(SimpleKvCache cache, IModel model)
    {
        // Do a forward pass at position 0 to populate the cache
        using var _ = model.Forward([1], [0], deviceId: -1, cache);
    }

    /// <summary>
    /// Mock model that always returns the same logits regardless of input.
    /// Produces an <see cref="UnmanagedTensor"/> with the predetermined logits
    /// for each token position in the batch.
    /// </summary>
    private sealed class MockModel : IModel
    {
        private readonly float[] _logits;
        private readonly int _vocabSize;

        public MockModel(float[] logits, int vocabSize)
        {
            _logits = logits;
            _vocabSize = vocabSize;
        }

        public ModelConfig Config => new()
        {
            VocabSize = _vocabSize,
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
            => Forward(tokenIds, positions, deviceId, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache)
        {
            int batchSize = tokenIds.Length;

            // Allocate output tensor: [batchSize, vocabSize]
            long totalFloats = (long)batchSize * _vocabSize;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);

            float* dst = (float*)ptr;
            for (int b = 0; b < batchSize; b++)
            {
                _logits.AsSpan().CopyTo(new Span<float>(dst + b * _vocabSize, _vocabSize));
            }

            var shape = new TensorShape(batchSize, _vocabSize);

            // Update KV-cache if provided
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

            return new UnmanagedTensor(shape, DType.Float32, deviceId, ptr);
        }

        public void Dispose() { }
    }
}
