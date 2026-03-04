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
        try
        {
            var probs = rentedProbs.AsSpan(0, vocabSize);
            var indices = rentedIndices.AsSpan(0, vocabSize);

            // Softmax to get probabilities
            TensorPrimitives.SoftMax(logits, probs);

            // Initialize indices
            for (int i = 0; i < vocabSize; i++)
                indices[i] = i;

            // Sort indices by probability descending
            SortDescendingByProbability(indices, probs);

            // Walk sorted order, accumulate probability, find cutoff
            float cumulative = 0f;
            int cutoffCount = vocabSize;
            for (int i = 0; i < vocabSize; i++)
            {
                cumulative += probs[indices[i]];
                if (cumulative >= topP)
                {
                    cutoffCount = i + 1; // keep this many
                    break;
                }
            }

            // Mask everything beyond the cutoff
            // Build a set of kept indices
            Span<bool> keep = vocabSize <= 4096
                ? stackalloc bool[vocabSize]
                : new bool[vocabSize];
            keep.Clear();

            for (int i = 0; i < cutoffCount; i++)
                keep[indices[i]] = true;

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
        }
    }

    private static void SortDescendingByProbability(Span<int> indices, ReadOnlySpan<float> probs)
    {
        // Simple insertion sort is fine for the indices — probabilities are the key.
        // For production, could use Array.Sort with custom comparer, but this avoids allocation.
        for (int i = 1; i < indices.Length; i++)
        {
            int key = indices[i];
            float keyProb = probs[key];
            int j = i - 1;
            while (j >= 0 && probs[indices[j]] < keyProb)
            {
                indices[j + 1] = indices[j];
                j--;
            }
            indices[j + 1] = key;
        }
    }
}
