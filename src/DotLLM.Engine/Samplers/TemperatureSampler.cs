using System.Numerics.Tensors;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Scales logits by dividing by temperature. Higher temperature = more random,
/// lower temperature = more deterministic. Temperature of 1.0 is a no-op.
/// </summary>
public sealed class TemperatureSampler : ISamplerStep
{
    private readonly float? _temperature;

    /// <summary>Creates a temperature step that reads from <see cref="SamplerContext"/>.</summary>
    public TemperatureSampler() { }

    /// <summary>Creates a self-configured temperature step.</summary>
    /// <param name="temperature">Temperature value to use (ignores context).</param>
    public TemperatureSampler(float temperature) => _temperature = temperature;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, SamplerContext context)
    {
        float temp = _temperature ?? context.Temperature;
        if (temp <= 0f || temp == 1.0f)
            return;

        TensorPrimitives.Multiply(logits, 1f / temp, logits);
    }
}
