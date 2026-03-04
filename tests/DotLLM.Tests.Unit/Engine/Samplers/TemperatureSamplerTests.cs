using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class TemperatureSamplerTests
{
    private readonly TemperatureSampler _sampler = new();

    [Fact]
    public void Apply_DividesLogitsByTemperature()
    {
        float[] logits = [2.0f, 4.0f, 6.0f];
        var context = new SamplerContext(Temperature: 2.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.Equal(1.0f, logits[0], precision: 5);
        Assert.Equal(2.0f, logits[1], precision: 5);
        Assert.Equal(3.0f, logits[2], precision: 5);
    }

    [Fact]
    public void Apply_Temperature1_IsNoOp()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Apply_Temperature0_IsNoOp()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new SamplerContext(Temperature: 0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Apply_NegativeTemperature_IsNoOp()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new SamplerContext(Temperature: -1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Apply_HighTemperature_FlattensDistribution()
    {
        float[] logits = [1.0f, 10.0f];
        var context = new SamplerContext(Temperature: 100.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        // With high temperature, values should be close together
        Assert.True(Math.Abs(logits[1] - logits[0]) < 0.2f);
    }
}
