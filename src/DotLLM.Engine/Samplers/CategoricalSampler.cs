using System.Buffers;
using System.Numerics.Tensors;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Samples a token ID from a logit distribution using categorical (multinomial) sampling.
/// Converts logits to probabilities via softmax, then samples from the cumulative distribution.
/// </summary>
public static class CategoricalSampler
{
    /// <summary>
    /// Samples a single token index from the given logits using the provided RNG.
    /// </summary>
    /// <param name="logits">Logit values (will be converted to probabilities internally).</param>
    /// <param name="rng">Random number generator for sampling.</param>
    /// <returns>The sampled token index.</returns>
    public static int Sample(ReadOnlySpan<float> logits, Random rng)
    {
        int vocabSize = logits.Length;
        float[]? rented = null;
        Span<float> probs = vocabSize <= 4096
            ? stackalloc float[vocabSize]
            : (rented = ArrayPool<float>.Shared.Rent(vocabSize)).AsSpan(0, vocabSize);

        try
        {
            TensorPrimitives.SoftMax(logits, probs);

            double r = rng.NextDouble();
            double cumulative = 0.0;

            for (int i = 0; i < vocabSize; i++)
            {
                cumulative += probs[i];
                if (r < cumulative)
                    return i;
            }

            // Floating-point edge case: return highest-probability token
            return TensorPrimitives.IndexOfMax(probs);
        }
        finally
        {
            if (rented is not null)
                ArrayPool<float>.Shared.Return(rented);
        }
    }
}
