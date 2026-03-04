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
            var indices = rentedIndices.AsSpan(0, vocabSize);

            // Softmax to get probabilities
            TensorPrimitives.SoftMax(logits, probs);

            // Initialize indices
            for (int i = 0; i < vocabSize; i++)
                indices[i] = i;

            // Sort ascending by probability using Array.Sort (IntroSort — O(V log V))
            Array.Sort(rentedProbs, rentedIndices, 0, vocabSize);

            // Walk backwards (descending probability), accumulate until we exceed topP
            float cumulative = 0f;
            int cutoffCount = vocabSize;
            for (int i = vocabSize - 1; i >= 0; i--)
            {
                cumulative += rentedProbs[i];
                if (cumulative >= topP)
                {
                    cutoffCount = vocabSize - i; // keep this many from the top
                    break;
                }
            }

            // Build kept-indices set
            var keep = rentedKeep.AsSpan(0, vocabSize);
            keep.Clear();

            int keepStart = vocabSize - cutoffCount;
            for (int i = keepStart; i < vocabSize; i++)
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
