using System.Numerics.Tensors;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Min-P sampling: masks tokens whose probability is less than minP × maxProbability.
/// Operates in logit space: logit(i) &lt; maxLogit + ln(minP) is equivalent to
/// prob(i) &lt; minP × maxProb because softmax is monotonic.
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

        float maxLogit = TensorPrimitives.Max(logits);
        float threshold = maxLogit + MathF.Log(minP);

        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] < threshold)
                logits[i] = float.NegativeInfinity;
        }
    }
}
