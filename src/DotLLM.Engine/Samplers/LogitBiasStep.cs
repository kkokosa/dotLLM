using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Applies per-token additive logit bias from an OpenAI-compatible token-bias map.
/// </summary>
public sealed class LogitBiasStep : ISamplerStep
{
    private readonly KeyValuePair<int, float>[] _biasEntries;

    /// <summary>
    /// Creates a logit-bias step from a token-bias map.
    /// </summary>
    public LogitBiasStep(IReadOnlyDictionary<int, float> logitBias)
    {
        _biasEntries = logitBias.Count == 0
            ? []
            : logitBias.ToArray();
    }

    /// <inheritdoc/>
    public void Apply(Span<float> logits, SamplerContext context)
    {
        for (int i = 0; i < _biasEntries.Length; i++)
        {
            int tokenId = _biasEntries[i].Key;
            if ((uint)tokenId >= (uint)logits.Length)
                continue;

            logits[tokenId] += _biasEntries[i].Value;
        }
    }
}
