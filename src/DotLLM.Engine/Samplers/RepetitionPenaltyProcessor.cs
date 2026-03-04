using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Applies repetition penalty to logits for tokens that appeared in recent history.
/// Positive logits are divided by the penalty factor, negative logits are multiplied,
/// effectively reducing the probability of repeated tokens in both cases.
/// </summary>
public sealed class RepetitionPenaltyProcessor : ILogitProcessor
{
    /// <inheritdoc/>
    public void Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext context)
    {
        float penalty = context.RepetitionPenalty;
        if (penalty == 1.0f || previousTokens.Count == 0)
            return;

        int window = context.RepetitionPenaltyWindow;
        int startIndex = window > 0 ? Math.Max(0, previousTokens.Count - window) : 0;

        // Collect unique token IDs in the window
        var penalizedTokens = new HashSet<int>();
        for (int i = startIndex; i < previousTokens.Count; i++)
            penalizedTokens.Add(previousTokens[i]);

        foreach (int tokenId in penalizedTokens)
        {
            if ((uint)tokenId >= (uint)logits.Length)
                continue;

            if (logits[tokenId] > 0f)
                logits[tokenId] /= penalty;
            else
                logits[tokenId] *= penalty;
        }
    }
}
