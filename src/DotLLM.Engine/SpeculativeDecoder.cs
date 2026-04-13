using System.Buffers;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Constraints;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Constraints;
using DotLLM.Engine.Samplers;

namespace DotLLM.Engine;

/// <summary>
/// Implements speculative decoding with draft-verify-accept.
/// Draft model proposes K tokens autoregressively; target model verifies all K tokens
/// in a single batched forward pass.
/// </summary>
/// <remarks>
/// <para>
/// Currently supports <b>greedy</b> acceptance only (<c>greedy: true</c>). In greedy mode the
/// accepted sequence is identical to what argmax-decoding the target model would produce,
/// provided the downstream sampler is also effectively greedy (temperature 0 and no
/// repetition penalty — see the gate in <c>TextGenerator</c>).
/// </para>
/// <para>
/// Probabilistic (modified rejection sampling) acceptance is <b>not yet distributionally
/// correct</b> under the full <see cref="SamplerPipeline"/>: the probabilities <c>q</c> and
/// <c>p</c> used in acceptance must be drawn from the same post-transform distribution the
/// pipeline actually samples from (temperature / top-k / top-p / min-p / repetition penalty),
/// not raw softmax over constraint-masked logits. Tracked in Wave 8 (issue #121). The
/// constructor rejects <c>greedy: false</c> until that work lands.
/// </para>
/// <para>
/// Supports draft models with slightly different vocab sizes (up to 128 token difference,
/// matching llama.cpp's tolerance). Probability comparison uses the shared vocab range;
/// tokens beyond the draft's vocab can only be produced by the target (as corrected/bonus tokens).
/// Zero-allocation on the hot path: all buffers are caller-owned or pool-rented, no per-call arrays.
/// </para>
/// </remarks>
public sealed class SpeculativeDecoder : ISpeculativeDecoder
{
    private readonly Random _rng;
    private readonly bool _greedy;

    /// <summary>
    /// Creates a new speculative decoder.
    /// </summary>
    /// <param name="greedy">Must be <c>true</c>. Probabilistic acceptance is not yet
    /// distributionally correct under the sampler pipeline; see Wave 8 (issue #121).</param>
    /// <param name="seed">Random seed for rejection sampling. Null = non-deterministic.</param>
    /// <exception cref="NotSupportedException">Thrown when <paramref name="greedy"/> is <c>false</c>.</exception>
    public SpeculativeDecoder(bool greedy, int? seed = null)
    {
        if (!greedy)
        {
            throw new NotSupportedException(
                "Probabilistic speculative decoding is not yet distributionally correct " +
                "under the sampler pipeline (temperature / top-k / top-p / min-p / repetition penalty). " +
                "Use greedy mode. See Wave 8 (issue #121) for the planned fix.");
        }
        _greedy = greedy;
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc/>
    public SpeculativeResult DraftAndVerify(
        IModel targetModel,
        IModel draftModel,
        IKvCache kvCacheTarget,
        IKvCache kvCacheDraft,
        SamplerPipeline pipeline,
        List<int> generatedIds,
        IDecodingConstraint? constraint,
        int position,
        int targetVocabSize,
        int draftVocabSize,
        int numCandidates,
        Span<int> outputBuffer)
    {
        // Clamp K to remaining cache capacity
        int maxPos = Math.Min(kvCacheTarget.MaxLength, kvCacheDraft.MaxLength);
        int k = Math.Min(numCandidates, maxPos - position - 1);
        if (k <= 0)
            return default;

        // Shared vocab range for probability comparison
        int sharedVocab = Math.Min(targetVocabSize, draftVocabSize);

        int lastToken = generatedIds[^1];

        // Flat buffer for draft probabilities: k rows × sharedVocab columns
        int[] draftTokens = ArrayPool<int>.Shared.Rent(k);
        float[] draftProbsFlat = ArrayPool<float>.Shared.Rent(k * sharedVocab);

        // Clone constraint for draft phase
        IDecodingConstraint? draftConstraint = constraint?.Clone();

        long draftTicks = 0;
        long verifyTicks = 0;

        try
        {
            // ── Draft Phase ──
            // Guard generatedIds against exceptions during draft forwards
            int originalGenCount = generatedIds.Count;
            int draftToken = lastToken;
            try
            {
                for (int i = 0; i < k; i++)
                {
                    int pos = position + i;

                    long fwdStart = Stopwatch.GetTimestamp();
                    using ITensor draftLogits = draftModel.Forward([draftToken], [pos], deviceId: -1, kvCacheDraft);
                    draftTicks += Stopwatch.GetTimestamp() - fwdStart;

                    unsafe
                    {
                        var logitSpan = new Span<float>((void*)draftLogits.DataPointer, draftVocabSize);

                        if (draftConstraint != null)
                            TokenMaskApplier.Apply(logitSpan, draftConstraint.GetAllowedTokens());

                        // Store draft probabilities in flat buffer (shared range only)
                        var probSlice = draftProbsFlat.AsSpan(i * sharedVocab, sharedVocab);
                        TensorPrimitives.SoftMax(logitSpan.Slice(0, sharedVocab), probSlice);

                        draftToken = pipeline.Sample(logitSpan, generatedIds);
                    }

                    draftTokens[i] = draftToken;
                    draftConstraint?.Advance(draftToken);

                    // Temporarily add to generatedIds for repetition penalty context
                    generatedIds.Add(draftToken);
                }
            }
            finally
            {
                // Restore generatedIds even if Forward threw mid-loop
                if (generatedIds.Count > originalGenCount)
                    generatedIds.RemoveRange(originalGenCount, generatedIds.Count - originalGenCount);
            }

            // ── Verify Phase (single batched forward pass) ──
            int verifyLen = k + 1;

            // Stackalloc for small buffers (K is typically 3-10)
            Span<int> verifyTokens = verifyLen <= 16 ? stackalloc int[verifyLen] : new int[verifyLen];
            Span<int> verifyPositions = verifyLen <= 16 ? stackalloc int[verifyLen] : new int[verifyLen];

            verifyTokens[0] = lastToken;
            verifyPositions[0] = position;
            for (int i = 0; i < k; i++)
            {
                verifyTokens[i + 1] = draftTokens[i];
                verifyPositions[i + 1] = position + i + 1;
            }

            int actualVerifyLen = Math.Min(verifyLen, maxPos - position);
            if (actualVerifyLen < 1)
                return default;

            long verifyStart = Stopwatch.GetTimestamp();
            using ITensor targetLogits = targetModel.Forward(
                verifyTokens.Slice(0, actualVerifyLen),
                verifyPositions.Slice(0, actualVerifyLen),
                deviceId: -1, kvCacheTarget);
            verifyTicks = Stopwatch.GetTimestamp() - verifyStart;

            // ── Accept/Reject Phase ──
            int acceptedCount = 0;

            unsafe
            {
                nint basePtr = targetLogits.DataPointer;

                for (int i = 0; i < Math.Min(k, actualVerifyLen); i++)
                {
                    int draftTok = draftTokens[i];
                    var targetLogitSpan = new Span<float>(
                        (void*)(basePtr + (long)i * targetVocabSize * sizeof(float)), targetVocabSize);

                    if (constraint != null)
                        TokenMaskApplier.Apply(targetLogitSpan, constraint.GetAllowedTokens());

                    // If draft token is beyond shared range, auto-reject
                    if (draftTok >= sharedVocab)
                    {
                        int corrected = _greedy
                            ? TensorPrimitives.IndexOfMax(targetLogitSpan)
                            : SampleFromLogits(targetLogitSpan);
                        outputBuffer[acceptedCount++] = corrected;
                        constraint?.Advance(corrected);
                        RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
                        return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
                    }

                    var draftProbSlice = draftProbsFlat.AsSpan(i * sharedVocab, sharedVocab);

                    if (_greedy)
                    {
                        int targetArgmax = TensorPrimitives.IndexOfMax(targetLogitSpan);
                        if (draftTok == targetArgmax)
                        {
                            outputBuffer[acceptedCount++] = draftTok;
                            constraint?.Advance(draftTok);
                        }
                        else
                        {
                            outputBuffer[acceptedCount++] = targetArgmax;
                            constraint?.Advance(targetArgmax);
                            RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
                            return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
                        }
                    }
                    else
                    {
                        float[] targetProbs = ArrayPool<float>.Shared.Rent(targetVocabSize);
                        try
                        {
                            TensorPrimitives.SoftMax(targetLogitSpan, targetProbs.AsSpan(0, targetVocabSize));

                            float p = targetProbs[draftTok];
                            float q = draftProbSlice[draftTok];
                            float acceptanceProb = q > 0 ? Math.Min(1.0f, p / q) : 0f;

                            if ((float)_rng.NextDouble() < acceptanceProb)
                            {
                                outputBuffer[acceptedCount++] = draftTok;
                                constraint?.Advance(draftTok);
                            }
                            else
                            {
                                int corrected = SampleCorrected(
                                    targetProbs.AsSpan(0, targetVocabSize),
                                    draftProbSlice,
                                    targetVocabSize, sharedVocab);
                                outputBuffer[acceptedCount++] = corrected;
                                constraint?.Advance(corrected);
                                RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
                                return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
                            }
                        }
                        finally
                        {
                            ArrayPool<float>.Shared.Return(targetProbs);
                        }
                    }
                }

                // All K accepted — sample bonus token from full target vocab
                if (actualVerifyLen > k)
                {
                    var bonusLogitSpan = new Span<float>(
                        (void*)(basePtr + (long)k * targetVocabSize * sizeof(float)), targetVocabSize);

                    if (constraint != null)
                        TokenMaskApplier.Apply(bonusLogitSpan, constraint.GetAllowedTokens());

                    int bonusToken = _greedy
                        ? TensorPrimitives.IndexOfMax(bonusLogitSpan)
                        : SampleFromLogits(bonusLogitSpan);

                    outputBuffer[acceptedCount++] = bonusToken;
                }
            }

            RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
            return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(draftTokens);
            ArrayPool<float>.Shared.Return(draftProbsFlat);
        }
    }

    /// <summary>
    /// Samples from the corrected distribution: normalize(max(0, p[i] - q[i])).
    /// Handles different vocab sizes: draft probs cover sharedVocab, target probs cover full targetVocab.
    /// For tokens beyond sharedVocab, q[i] = 0 so corrected[i] = p[i] (target probability passes through).
    /// </summary>
    private int SampleCorrected(ReadOnlySpan<float> targetProbs, ReadOnlySpan<float> draftProbs,
        int targetVocabSize, int sharedVocab)
    {
        float[] corrected = ArrayPool<float>.Shared.Rent(targetVocabSize);
        try
        {
            float sum = 0;
            for (int i = 0; i < sharedVocab; i++)
            {
                corrected[i] = Math.Max(0, targetProbs[i] - draftProbs[i]);
                sum += corrected[i];
            }
            for (int i = sharedVocab; i < targetVocabSize; i++)
            {
                corrected[i] = Math.Max(0, targetProbs[i]);
                sum += corrected[i];
            }

            if (sum <= 0)
                return SampleFromProbs(targetProbs, targetVocabSize);

            float invSum = 1.0f / sum;
            double r = _rng.NextDouble();
            double cumulative = 0.0;

            for (int i = 0; i < targetVocabSize; i++)
            {
                cumulative += corrected[i] * invSum;
                if (r < cumulative)
                    return i;
            }

            return targetVocabSize - 1;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(corrected);
        }
    }

    private int SampleFromProbs(ReadOnlySpan<float> probs, int vocabSize)
    {
        double r = _rng.NextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < vocabSize; i++)
        {
            cumulative += probs[i];
            if (r < cumulative)
                return i;
        }

        return vocabSize - 1;
    }

    private int SampleFromLogits(Span<float> logits)
    {
        int vocabSize = logits.Length;
        float[]? rented = null;
        Span<float> probs = vocabSize <= 4096
            ? stackalloc float[vocabSize]
            : (rented = ArrayPool<float>.Shared.Rent(vocabSize)).AsSpan(0, vocabSize);

        try
        {
            TensorPrimitives.SoftMax(logits, probs);

            double r = _rng.NextDouble();
            double cumulative = 0.0;

            for (int i = 0; i < vocabSize; i++)
            {
                cumulative += probs[i];
                if (r < cumulative)
                    return i;
            }

            return vocabSize - 1;
        }
        finally
        {
            if (rented is not null)
                ArrayPool<float>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RollbackCaches(IKvCache target, IKvCache draft, int acceptedEnd, int k)
    {
        if (acceptedEnd <= target.CurrentLength)
            target.Rollback(acceptedEnd);

        int draftEnd = Math.Min(acceptedEnd, draft.CurrentLength);
        if (draftEnd <= draft.CurrentLength)
            draft.Rollback(draftEnd);
    }
}
