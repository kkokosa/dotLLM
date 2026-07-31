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
/// Both greedy (argmax) and probabilistic acceptance are supported. The pipeline's mode
/// (<see cref="SamplerPipeline.IsGreedy"/>) determines which path runs; the decoder mirrors
/// it so the accept/reject scheme matches what the downstream sampler actually does.
/// </para>
/// <para>
/// For probabilistic acceptance, <c>q</c> (draft probability) and <c>p</c> (target probability)
/// are drawn from the same post-transform distribution the <see cref="SamplerPipeline"/>
/// samples from (temperature / top-k / top-p / min-p / repetition penalty). This makes modified
/// rejection sampling produce samples from the target distribution exactly (Leviathan et al. 2023;
/// Chen et al. 2023). The repetition-penalty context for token <c>i</c> in the verify pass is
/// rebuilt to match the draft pass — <c>original generatedIds + draft_0 … draft_{i-1}</c> —
/// since both penalty applications must see the same history.
/// </para>
/// <para>
/// Supports draft models with slightly different vocab sizes (up to 128 token difference,
/// matching llama.cpp's tolerance). Probability comparison uses the shared vocab range;
/// tokens beyond the draft's vocab can only be produced by the target (as corrected/bonus tokens).
/// When the draft vocab is the wider one, <c>q</c> is still softmaxed over the draft's full support so
/// it is the true proposal marginal, and a draft token landing outside the shared range is rejected
/// through the same residual distribution as any other reject — so exactness holds for unequal vocabs
/// too, not merely when the tail mass happens to be zero.
/// Zero-allocation on the hot path: all buffers are caller-owned or pool-rented, no per-call arrays.
/// </para>
/// </remarks>
public sealed class SpeculativeDecoder : ISpeculativeDecoder
{
    private readonly Random _rng;

    /// <summary>
    /// Creates a new speculative decoder. The <paramref name="greedy"/> flag is retained for
    /// API compatibility but has no behavioural effect — acceptance follows the
    /// <see cref="SamplerPipeline.IsGreedy"/> of the pipeline passed to
    /// <see cref="DraftAndVerify"/>.
    /// </summary>
    /// <param name="greedy">Ignored; pipeline mode governs acceptance. Kept for API stability.</param>
    /// <param name="seed">
    /// Random seed for rejection sampling. Null = non-deterministic. The value is deterministically
    /// mixed (see <see cref="DeriveAcceptanceSeed"/>) so callers can pass the same seed they gave the
    /// <see cref="SamplerPipeline"/> without the two RNG streams coinciding.
    /// </param>
    public SpeculativeDecoder(bool greedy, int? seed = null)
    {
        _ = greedy;
        _rng = seed.HasValue ? new Random(DeriveAcceptanceSeed(seed.Value)) : new Random();
    }

    /// <summary>
    /// Derives the accept/reject RNG seed from the pipeline seed.
    /// </summary>
    /// <remarks>
    /// Modified rejection sampling requires the acceptance draw <c>u ~ U(0,1)</c> to be independent of
    /// the draw that produced the proposal token. <see cref="SamplerPipeline"/> seeds its own
    /// <see cref="Random"/> from the same <c>options.Seed</c>; without mixing, both streams emit the
    /// identical sequence, so the <c>i</c>-th acceptance test would reuse the very uniform that
    /// inverse-CDF-selected draft token <c>i</c> — correlating acceptance with the token's position in
    /// the sorted CDF and breaking the distributional guarantee. A SplitMix64-style avalanche keeps the
    /// two streams reproducible but statistically independent.
    /// </remarks>
    private static int DeriveAcceptanceSeed(int pipelineSeed)
    {
        ulong z = (ulong)(uint)pipelineSeed + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (int)(uint)z;
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

        bool greedy = pipeline.IsGreedy;

        // Shared vocab range for probability comparison
        int sharedVocab = Math.Min(targetVocabSize, draftVocabSize);

        int lastToken = generatedIds[^1];

        // Flat buffer for draft probabilities: k rows × sharedVocab columns
        int[] draftTokens = ArrayPool<int>.Shared.Rent(k);
        float[] draftProbsFlat = greedy ? [] : ArrayPool<float>.Shared.Rent(k * sharedVocab);

        // One scratch buffer for the whole verify pass — renting per accepted token churned the pool
        // K times per speculative step. Only the non-greedy path softmaxes the target logits.
        float[] targetProbs = greedy ? [] : ArrayPool<float>.Shared.Rent(targetVocabSize);

        // Only needed when the draft vocab is wider than the target's — see the draft-loop softmax.
        bool needsFullDraftProbs = !greedy && draftVocabSize > sharedVocab;
        float[] draftFullProbs = needsFullDraftProbs ? ArrayPool<float>.Shared.Rent(draftVocabSize) : [];

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

                        // Apply the pipeline's transform chain in-place, then sample. The transformed
                        // logits are what the pipeline actually draws from — so both q (here) and p
                        // (verify pass) must come from this distribution for modified rejection
                        // sampling to be exact.
                        pipeline.ApplyTransforms(logitSpan, generatedIds);

                        if (!greedy)
                        {
                            var probSlice = draftProbsFlat.AsSpan(i * sharedVocab, sharedVocab);
                            if (draftVocabSize == sharedVocab)
                            {
                                TensorPrimitives.SoftMax(logitSpan, probSlice);
                            }
                            else
                            {
                                // Draft vocab is wider than the target's. Softmax over the *full* draft
                                // support and keep the shared prefix, so q is the true proposal marginal.
                                // Softmaxing the truncated slice would renormalise away the tail mass and
                                // inflate q, biasing min(1, p/q) — the rejection scheme would no longer be
                                // exact. The tail mass stays accounted for: a draft token landing outside
                                // the shared range is rejected below with q taken from this same marginal.
                                var fullSlice = draftFullProbs.AsSpan(0, draftVocabSize);
                                TensorPrimitives.SoftMax(logitSpan, fullSlice);
                                fullSlice.Slice(0, sharedVocab).CopyTo(probSlice);
                            }
                        }

                        draftToken = pipeline.SampleFromTransformed(logitSpan);
                    }

                    draftTokens[i] = draftToken;
                    draftConstraint?.Advance(draftToken);

                    // Append so the next draft's repetition-penalty context matches what the verify
                    // pass will reconstruct (original + draft_0 … draft_{i-1}).
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

            // Rebuild the same repetition-penalty context the draft loop used so transforms at
            // position i see (original + draft_0 … draft_{i-1}). Restored in finally.
            int verifyGenCount = generatedIds.Count;

            try
            {
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

                        // Apply the same transforms the draft pass applied. generatedIds currently
                        // holds (original + draft_0 … draft_{i-1}) — identical to the draft-loop context.
                        pipeline.ApplyTransforms(targetLogitSpan, generatedIds);

                        // Draft token beyond the shared range: the target cannot emit it, so p = 0 and
                        // min(1, p/q) = 0 — an unconditional reject. The replacement must still come from
                        // the residual normalize(max(0, p - q)), exactly as an ordinary reject does;
                        // sampling raw p here would over-weight tokens the draft already covered.
                        if (draftTok >= sharedVocab)
                        {
                            int corrected;
                            if (greedy)
                            {
                                corrected = pipeline.SampleFromTransformed(targetLogitSpan);
                            }
                            else
                            {
                                var targetProbSpanOut = targetProbs.AsSpan(0, targetVocabSize);
                                TensorPrimitives.SoftMax(targetLogitSpan, targetProbSpanOut);
                                corrected = SampleCorrected(
                                    targetProbSpanOut,
                                    draftProbsFlat.AsSpan(i * sharedVocab, sharedVocab),
                                    targetVocabSize, sharedVocab);
                            }
                            outputBuffer[acceptedCount++] = corrected;
                            constraint?.Advance(corrected);
                            RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
                            return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
                        }

                        if (greedy)
                        {
                            int targetArgmax = TensorPrimitives.IndexOfMax(targetLogitSpan);
                            if (draftTok == targetArgmax)
                            {
                                outputBuffer[acceptedCount++] = draftTok;
                                constraint?.Advance(draftTok);
                                generatedIds.Add(draftTok);
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
                            var draftProbSlice = draftProbsFlat.AsSpan(i * sharedVocab, sharedVocab);
                            var targetProbSpan = targetProbs.AsSpan(0, targetVocabSize);
                            TensorPrimitives.SoftMax(targetLogitSpan, targetProbSpan);

                            float p = targetProbSpan[draftTok];
                            float q = draftProbSlice[draftTok];
                            float acceptanceProb = q > 0 ? Math.Min(1.0f, p / q) : 0f;

                            if ((float)_rng.NextDouble() < acceptanceProb)
                            {
                                outputBuffer[acceptedCount++] = draftTok;
                                constraint?.Advance(draftTok);
                                generatedIds.Add(draftTok);
                            }
                            else
                            {
                                int corrected = SampleCorrected(
                                    targetProbSpan,
                                    draftProbSlice,
                                    targetVocabSize, sharedVocab);
                                outputBuffer[acceptedCount++] = corrected;
                                constraint?.Advance(corrected);
                                RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
                                return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
                            }
                        }
                    }

                    // All K accepted — sample bonus token from the full transformed target vocab.
                    // Context now holds (original + draft_0 … draft_{k-1}) which matches what the
                    // model's forward already conditioned on for position k.
                    if (actualVerifyLen > k)
                    {
                        var bonusLogitSpan = new Span<float>(
                            (void*)(basePtr + (long)k * targetVocabSize * sizeof(float)), targetVocabSize);

                        if (constraint != null)
                            TokenMaskApplier.Apply(bonusLogitSpan, constraint.GetAllowedTokens());

                        pipeline.ApplyTransforms(bonusLogitSpan, generatedIds);
                        int bonusToken = pipeline.SampleFromTransformed(bonusLogitSpan);

                        outputBuffer[acceptedCount++] = bonusToken;
                    }
                }
            }
            finally
            {
                if (generatedIds.Count > verifyGenCount)
                    generatedIds.RemoveRange(verifyGenCount, generatedIds.Count - verifyGenCount);
            }

            RollbackCaches(kvCacheTarget, kvCacheDraft, position + acceptedCount, k);
            return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(draftTokens);
            if (!greedy)
            {
                ArrayPool<float>.Shared.Return(draftProbsFlat);
                ArrayPool<float>.Shared.Return(targetProbs);
                if (needsFullDraftProbs)
                    ArrayPool<float>.Shared.Return(draftFullProbs);
            }
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
