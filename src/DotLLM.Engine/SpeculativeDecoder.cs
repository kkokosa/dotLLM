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
/// Implements speculative decoding with modified rejection sampling.
/// Draft model proposes K tokens autoregressively; target model verifies all K tokens
/// in a single batched forward pass. Output distribution is provably identical to
/// the target model's distribution.
/// </summary>
/// <remarks>
/// Supports draft models with slightly different vocab sizes (up to 128 token difference,
/// matching llama.cpp's tolerance). Probability comparison uses the shared vocab range;
/// tokens beyond the draft's vocab can only be produced by the target (as corrected/bonus tokens).
/// </remarks>
public sealed class SpeculativeDecoder : ISpeculativeDecoder
{
    private readonly Random _rng;
    private readonly bool _greedy;

    /// <summary>
    /// Creates a new speculative decoder.
    /// </summary>
    /// <param name="greedy">When true, uses argmax matching instead of probabilistic acceptance.</param>
    /// <param name="seed">Random seed for rejection sampling. Null = non-deterministic.</param>
    public SpeculativeDecoder(bool greedy, int? seed = null)
    {
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
        int numCandidates)
    {
        // Clamp K to remaining cache capacity
        int maxPos = Math.Min(kvCacheTarget.MaxLength, kvCacheDraft.MaxLength);
        int k = Math.Min(numCandidates, maxPos - position - 1);
        if (k <= 0)
            return default;

        // Shared vocab range for probability comparison
        int sharedVocab = Math.Min(targetVocabSize, draftVocabSize);

        int lastToken = generatedIds[^1];

        // Allocate storage for draft tokens and their probability distributions
        int[] draftTokens = ArrayPool<int>.Shared.Rent(k);
        float[][] draftProbs = new float[k][];
        for (int i = 0; i < k; i++)
            draftProbs[i] = ArrayPool<float>.Shared.Rent(sharedVocab);

        // Clone constraint for draft phase
        IDecodingConstraint? draftConstraint = constraint?.Clone();

        long draftTicks = 0;
        long verifyTicks = 0;

        try
        {
            // ── Draft Phase ──
            // Run draft model K times autoregressively
            int draftToken = lastToken;
            for (int i = 0; i < k; i++)
            {
                int pos = position + i;

                long fwdStart = Stopwatch.GetTimestamp();
                using ITensor draftLogits = draftModel.Forward([draftToken], [pos], deviceId: -1, kvCacheDraft);
                draftTicks += Stopwatch.GetTimestamp() - fwdStart;

                unsafe
                {
                    var logitSpan = new Span<float>((void*)draftLogits.DataPointer, draftVocabSize);

                    // Apply constraint mask to draft
                    if (draftConstraint != null)
                        TokenMaskApplier.Apply(logitSpan, draftConstraint.GetAllowedTokens());

                    // Store draft probabilities over shared range only
                    TensorPrimitives.SoftMax(logitSpan.Slice(0, sharedVocab),
                        draftProbs[i].AsSpan(0, sharedVocab));

                    // Sample draft token (from full draft vocab)
                    draftToken = pipeline.Sample(logitSpan, generatedIds);
                }

                draftTokens[i] = draftToken;
                draftConstraint?.Advance(draftToken);

                // Temporarily add to generatedIds for repetition penalty context
                generatedIds.Add(draftToken);
            }

            // Remove the draft tokens from generatedIds (they were temporary)
            generatedIds.RemoveRange(generatedIds.Count - k, k);

            // ── Verify Phase (single batched forward pass) ──
            int verifyLen = k + 1;
            int[] verifyTokens = ArrayPool<int>.Shared.Rent(verifyLen);
            int[] verifyPositions = ArrayPool<int>.Shared.Rent(verifyLen);

            try
            {
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
                    verifyTokens.AsSpan(0, actualVerifyLen),
                    verifyPositions.AsSpan(0, actualVerifyLen),
                    deviceId: -1, kvCacheTarget);
                verifyTicks = Stopwatch.GetTimestamp() - verifyStart;

                // ── Accept/Reject Phase ──
                int[] accepted = new int[k + 1];
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

                        // If draft token is beyond shared range, it can't match — auto-reject
                        if (draftTok >= sharedVocab)
                        {
                            int corrected = _greedy
                                ? TensorPrimitives.IndexOfMax(targetLogitSpan)
                                : SampleFromLogits(targetLogitSpan);
                            accepted[acceptedCount++] = corrected;
                            constraint?.Advance(corrected);
                            RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
                            return BuildResult(accepted, acceptedCount, draftTicks, verifyTicks, k);
                        }

                        if (_greedy)
                        {
                            int targetArgmax = TensorPrimitives.IndexOfMax(targetLogitSpan);
                            if (draftTok == targetArgmax)
                            {
                                accepted[acceptedCount++] = draftTok;
                                constraint?.Advance(draftTok);
                            }
                            else
                            {
                                accepted[acceptedCount++] = targetArgmax;
                                constraint?.Advance(targetArgmax);
                                RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
                                return BuildResult(accepted, acceptedCount, draftTicks, verifyTicks, k);
                            }
                        }
                        else
                        {
                            // Probabilistic acceptance over shared vocab range
                            float[] targetProbs = ArrayPool<float>.Shared.Rent(targetVocabSize);
                            try
                            {
                                TensorPrimitives.SoftMax(targetLogitSpan, targetProbs.AsSpan(0, targetVocabSize));

                                float p = targetProbs[draftTok];
                                float q = draftProbs[i][draftTok]; // draft probs are over sharedVocab
                                float acceptanceProb = q > 0 ? Math.Min(1.0f, p / q) : 0f;

                                if ((float)_rng.NextDouble() < acceptanceProb)
                                {
                                    accepted[acceptedCount++] = draftTok;
                                    constraint?.Advance(draftTok);
                                }
                                else
                                {
                                    // Sample corrected token from full target vocab
                                    int corrected = SampleCorrected(
                                        targetProbs.AsSpan(0, targetVocabSize),
                                        draftProbs[i].AsSpan(0, sharedVocab),
                                        targetVocabSize, sharedVocab);
                                    accepted[acceptedCount++] = corrected;
                                    constraint?.Advance(corrected);
                                    RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
                                    return BuildResult(accepted, acceptedCount, draftTicks, verifyTicks, k);
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

                        accepted[acceptedCount++] = bonusToken;
                    }
                }

                RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
                return BuildResult(accepted, acceptedCount, draftTicks, verifyTicks, k);
            }
            finally
            {
                ArrayPool<int>.Shared.Return(verifyTokens);
                ArrayPool<int>.Shared.Return(verifyPositions);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(draftTokens);
            for (int i = 0; i < k; i++)
                ArrayPool<float>.Shared.Return(draftProbs[i]);
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
            // Shared range: max(0, p - q)
            for (int i = 0; i < sharedVocab; i++)
            {
                corrected[i] = Math.Max(0, targetProbs[i] - draftProbs[i]);
                sum += corrected[i];
            }
            // Target-only range: p passes through (q = 0)
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static SpeculativeResult BuildResult(int[] accepted, int count, long draftTicks, long verifyTicks, int k) =>
        new(accepted[..count], count, draftTicks, verifyTicks, k);
}
