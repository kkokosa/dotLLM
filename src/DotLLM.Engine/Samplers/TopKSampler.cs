using System.Buffers;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Top-K sampling: keeps only the K highest-probability logits, setting the rest to -infinity.
/// </summary>
public sealed class TopKSampler : ISamplerStep
{
    private readonly int? _topK;

    /// <summary>Creates a top-K step that reads from <see cref="SamplerContext"/>.</summary>
    public TopKSampler() { }

    /// <summary>Creates a self-configured top-K step.</summary>
    /// <param name="topK">Number of top tokens to keep (ignores context).</param>
    public TopKSampler(int topK) => _topK = topK;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, SamplerContext context)
    {
        int k = _topK ?? context.TopK;
        if (k <= 0 || k >= logits.Length)
            return;

        float[] rented = ArrayPool<float>.Shared.Rent(logits.Length);
        try
        {
            var sorted = rented.AsSpan(0, logits.Length);
            logits.CopyTo(sorted);

            // Partial sort: find the k-th largest value as threshold.
            // Full sort is simpler and sufficient for typical vocab sizes with this approach.
            sorted.Sort();
            // sorted is ascending; k-th largest is at index [length - k]
            float threshold = sorted[logits.Length - k];

            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] < threshold)
                    logits[i] = float.NegativeInfinity;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }
}
