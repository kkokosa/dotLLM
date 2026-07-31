using System.Buffers;
using System.Numerics.Tensors;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Top-P (nucleus) sampling: keeps the smallest set of tokens whose cumulative probability
/// exceeds P, masking the rest to -infinity.
/// </summary>
public sealed class TopPSampler : ISamplerStep
{
    private readonly float? _topP;

    /// <summary>Creates a top-P step that reads from <see cref="SamplerContext"/>.</summary>
    public TopPSampler() { }

    /// <summary>Creates a self-configured top-P step.</summary>
    /// <param name="topP">Cumulative probability threshold (ignores context).</param>
    public TopPSampler(float topP) => _topP = topP;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, SamplerContext context)
    {
        float topP = _topP ?? context.TopP;
        if (topP >= 1.0f)
            return;

        int vocabSize = logits.Length;
        float[] rentedProbs = ArrayPool<float>.Shared.Rent(vocabSize);
        int[] rentedIndices = ArrayPool<int>.Shared.Rent(vocabSize);
        bool[] rentedKeep = ArrayPool<bool>.Shared.Rent(vocabSize);
        try
        {
            var probs = rentedProbs.AsSpan(0, vocabSize);

            // Softmax to get probabilities (full vocab — needed for the cutoff computation
            // and the mask write-back).
            TensorPrimitives.SoftMax(logits, probs);

            // Pre-filter cutoff (Karpathy llama2.c / run.c:sample_topp): any token with
            // probability strictly less than `cutoff = (1 - topP) / (n - 1)` cannot be
            // part of the nucleus. Proof: the dropped tokens collectively contribute at
            // most `(n - 1) * cutoff = 1 - topP`, so the surviving (filtered) set holds
            // at least `topP` of the mass. Walking it in descending order will hit `topP`
            // within the filtered set, so dropping the rest is safe.
            //
            // At typical vocabSize=32K–128K and topP=0.9, this eliminates ~99% of tokens
            // from the subsequent sort, turning O(V log V) into O(V) (single pass to
            // build the candidate set) + O(K log K) (sort the much smaller candidate set,
            // typically a few hundred entries).
            //
            // The filter can empty the candidate set outright when topP < 1/vocabSize;
            // that degenerate case is handled explicitly after the loop.
            float cutoff = vocabSize > 1 ? (1.0f - topP) / (vocabSize - 1) : 0.0f;

            int candidateCount = 0;
            for (int i = 0; i < vocabSize; i++)
            {
                if (rentedProbs[i] >= cutoff)
                {
                    rentedProbs[candidateCount] = rentedProbs[i];
                    rentedIndices[candidateCount] = i;
                    candidateCount++;
                }
            }

            if (candidateCount == 0)
            {
                // Every probability fell below the cutoff. Reachable only when
                // topP < 1/vocabSize: summing `p_i < (1 - topP) / (n - 1)` over all n
                // tokens gives 1 < n(1 - topP)/(n - 1), i.e. topP < 1/n. In that regime
                // the un-filtered algorithm keeps exactly one token, because
                // max(p) >= 1/n > topP terminates the cumulative walk on its first step.
                // Seed the candidate set with the argmax so the outcome matches instead
                // of masking the entire vocabulary. (Hit by e.g. topP = 0, or a 2-token
                // vocab with topP = 0.1 where cutoff = 0.9 exceeds both probabilities.)
                //
                // NOTE: no compaction has happened yet, so `rentedProbs` still holds the
                // full softmax and `probs` is a valid view over it.
                int argmax = TensorPrimitives.IndexOfMax(probs);
                float argmaxProb = probs[argmax];
                rentedProbs[0] = argmaxProb;
                rentedIndices[0] = argmax;
                candidateCount = 1;
            }

            // Sort the filtered candidates ascending by probability (IntroSort — O(K log K)).
            Array.Sort(rentedProbs, rentedIndices, 0, candidateCount);

            // Walk backwards (descending probability), accumulate until we exceed topP.
            float cumulative = 0f;
            int cutoffCount = candidateCount;
            for (int i = candidateCount - 1; i >= 0; i--)
            {
                cumulative += rentedProbs[i];
                if (cumulative >= topP)
                {
                    cutoffCount = candidateCount - i; // keep this many from the top
                    break;
                }
            }

            // Build kept-indices set
            var keep = rentedKeep.AsSpan(0, vocabSize);
            keep.Clear();

            int keepStart = candidateCount - cutoffCount;
            for (int i = keepStart; i < candidateCount; i++)
                keep[rentedIndices[i]] = true;

            for (int i = 0; i < vocabSize; i++)
            {
                if (!keep[i])
                    logits[i] = float.NegativeInfinity;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rentedProbs);
            ArrayPool<int>.Shared.Return(rentedIndices);
            ArrayPool<bool>.Shared.Return(rentedKeep);
        }
    }
}
