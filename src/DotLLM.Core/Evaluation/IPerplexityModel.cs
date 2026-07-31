using DotLLM.Core.Tensors;

namespace DotLLM.Core.Evaluation;

/// <summary>
/// The minimal model surface perplexity scoring needs: a teacher-forced forward pass over a
/// token window, plus enough metadata to interpret the logits it returns.
/// </summary>
/// <remarks>
/// <para>Deliberately narrower than <c>IModel</c>. Perplexity needs neither sampling, KV-cache
/// lifetime management, nor streaming, and binding the evaluator to the full model interface
/// would prevent scoring a bare backend or a test double.</para>
/// <para><b>The evaluator never loads weights.</b> Implementations are handed an
/// already-constructed model, so a caller that has the weights resident on a device is never
/// forced into a second host-side copy. This matters on unified-memory parts (e.g. Strix Halo)
/// where a large VRAM carve-out leaves host RAM scarce, and perplexity — a long sequence of
/// full-context prefills rather than a single load — is the workload most punished by paying
/// for the model twice.</para>
/// </remarks>
public interface IPerplexityModel
{
    /// <summary>Vocabulary size; the row length of the returned logits.</summary>
    int VocabSize { get; }

    /// <summary>
    /// Maximum token window a single <see cref="Forward"/> call accepts. Sliding-window scoring
    /// never requests a window larger than this.
    /// </summary>
    int MaxContextLength { get; }

    /// <summary>
    /// <see langword="true"/> when <see cref="Forward"/> returns logits for every position
    /// (shape <c>[seqLen, VocabSize]</c>); <see langword="false"/> when only the final row is
    /// returned (shape <c>[1, VocabSize]</c> or <c>[VocabSize]</c>).
    /// </summary>
    /// <remarks>
    /// This is the single axis that decides scoring cost, and the reason the existing per-test
    /// helpers diverged into two shapes. All-rows backends score a window in one forward pass
    /// (O(n)); last-row-only backends must re-prefill each growing prefix (O(n^2)). The evaluator
    /// selects the strategy from this flag rather than the caller hard-coding one.
    /// </remarks>
    bool ReturnsAllRows { get; }

    /// <summary>
    /// Runs a teacher-forced forward pass over <paramref name="tokens"/>.
    /// </summary>
    /// <param name="tokens">Input token ids for this window.</param>
    /// <param name="positions">
    /// Position ids, one per token, passed explicitly rather than derived so the caller controls
    /// them.
    /// <para><b>Do not assume they are absolute corpus offsets.</b> Sliding-window scoring passes
    /// window-relative positions restarting at <c>0</c> for every window, because llama.cpp
    /// evaluates each chunk as an independent sequence and matching it is the point of that mode.
    /// It is also what lets a corpus longer than <see cref="MaxContextLength"/> be scored at all —
    /// absolute positions would run past the model's limit on the second window.</para>
    /// </param>
    /// <returns>
    /// Logits, owned by the caller. Row layout is governed by <see cref="ReturnsAllRows"/>.
    /// </returns>
    ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions);
}
