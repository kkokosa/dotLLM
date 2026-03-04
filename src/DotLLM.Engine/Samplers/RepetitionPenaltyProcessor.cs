using System.Buffers;
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
        int windowLength = previousTokens.Count - startIndex;

        // Rent array, copy window tokens, sort for dedup without HashSet allocation
        int[] rented = ArrayPool<int>.Shared.Rent(windowLength);
        try
        {
            for (int i = 0; i < windowLength; i++)
                rented[i] = previousTokens[startIndex + i];

            Array.Sort(rented, 0, windowLength);

            // Iterate sorted array, skip duplicates
            int prev = -1;
            for (int i = 0; i < windowLength; i++)
            {
                int tokenId = rented[i];
                if (tokenId == prev)
                    continue;
                prev = tokenId;

                if ((uint)tokenId >= (uint)logits.Length)
                    continue;

                if (logits[tokenId] > 0f)
                    logits[tokenId] /= penalty;
                else
                    logits[tokenId] *= penalty;
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }
    }
}
