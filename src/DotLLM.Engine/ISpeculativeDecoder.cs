using DotLLM.Core.Attention;
using DotLLM.Core.Constraints;
using DotLLM.Core.Models;
using DotLLM.Engine.Samplers;

namespace DotLLM.Engine;

/// <summary>
/// Speculative decoding: a draft model proposes K candidate tokens and the target model verifies
/// them in a single batched forward pass. Achieves 2-3x decode speedup while preserving the
/// target model's exact output distribution via modified rejection sampling.
/// </summary>
public interface ISpeculativeDecoder
{
    /// <summary>
    /// Drafts candidate tokens with the draft model and verifies them with the target model.
    /// On rejection, rolls back KV-caches and constraint state to the last accepted position.
    /// </summary>
    /// <param name="targetModel">The full (target) model for verification.</param>
    /// <param name="draftModel">The smaller (draft) model for fast token proposals.</param>
    /// <param name="kvCacheTarget">KV-cache for the target model.</param>
    /// <param name="kvCacheDraft">KV-cache for the draft model.</param>
    /// <param name="pipeline">Sampling pipeline for token selection.</param>
    /// <param name="generatedIds">All previously generated token IDs (for repetition penalty).</param>
    /// <param name="constraint">Optional decoding constraint (cloned before drafting, rolled back on rejection).</param>
    /// <param name="position">Current sequence position (prompt length + decoded so far).</param>
    /// <param name="targetVocabSize">Vocabulary size of the target model.</param>
    /// <param name="draftVocabSize">Vocabulary size of the draft model (may differ by up to 128 tokens).</param>
    /// <param name="numCandidates">Number of draft tokens to propose (K).</param>
    /// <param name="outputBuffer">Caller-owned buffer for accepted token IDs (must be at least K+1 elements).
    /// Avoids per-call allocation — the caller reuses this buffer across speculation rounds.</param>
    /// <returns>Result containing accepted count and timing information. Accepted tokens are in <paramref name="outputBuffer"/>.</returns>
    SpeculativeResult DraftAndVerify(
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
        Span<int> outputBuffer);
}

/// <summary>
/// Maximum allowed vocabulary size difference between target and draft models.
/// Matches llama.cpp's SPEC_VOCAB_MAX_SIZE_DIFFERENCE.
/// Models within this tolerance share the same base tokenizer — the extra tokens
/// are typically padding/reserved IDs that never appear in normal generation.
/// </summary>
public static class SpeculativeConstants
{
    /// <summary>Maximum allowed vocab size difference (128 tokens, same as llama.cpp).</summary>
    public const int MaxVocabSizeDifference = 128;

    /// <summary>
    /// Checks whether two vocab sizes are compatible for speculative decoding.
    /// </summary>
    public static bool AreVocabsCompatible(int targetVocab, int draftVocab) =>
        Math.Abs(targetVocab - draftVocab) <= MaxVocabSizeDifference;
}

/// <summary>
/// Result of a single speculative decoding step. Accepted token IDs are written
/// to the caller-provided output buffer passed to <see cref="ISpeculativeDecoder.DraftAndVerify"/>.
/// </summary>
/// <param name="AcceptedCount">Number of accepted tokens written to the output buffer.</param>
/// <param name="DraftTicks">Stopwatch ticks spent on draft model forward passes.</param>
/// <param name="VerifyTicks">Stopwatch ticks spent on target model verification forward pass.</param>
/// <param name="DraftedCount">Total number of draft tokens proposed (K).</param>
public readonly record struct SpeculativeResult(
    int AcceptedCount,
    long DraftTicks,
    long VerifyTicks,
    int DraftedCount);
