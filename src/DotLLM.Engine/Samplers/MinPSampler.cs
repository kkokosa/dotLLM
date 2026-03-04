using System.Buffers;
using System.Numerics.Tensors;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Min-P sampling: masks tokens whose probability is less than minP × maxProbability.
/// This adapts to the confidence of the distribution — more tokens survive when
/// the model is uncertain, fewer when it's confident.
/// </summary>
public sealed class MinPSampler : ISamplerStep
{
    private readonly float? _minP;

    /// <summary>Creates a min-P step that reads from <see cref="SamplerContext"/>.</summary>
    public MinPSampler() { }

    /// <summary>Creates a self-configured min-P step.</summary>
    /// <param name="minP">Minimum probability relative to the max (ignores context).</param>
    public MinPSampler(float minP) => _minP = minP;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, SamplerContext context)
    {
        float minP = _minP ?? context.MinP;
        if (minP <= 0f)
            return;

        int vocabSize = logits.Length;
        float[] rentedProbs = ArrayPool<float>.Shared.Rent(vocabSize);
        try
        {
            var probs = rentedProbs.AsSpan(0, vocabSize);
            TensorPrimitives.SoftMax(logits, probs);

            float maxProb = TensorPrimitives.Max(probs);
            float threshold = minP * maxProb;

            for (int i = 0; i < vocabSize; i++)
            {
                if (probs[i] < threshold)
                    logits[i] = float.NegativeInfinity;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rentedProbs);
        }
    }
}
