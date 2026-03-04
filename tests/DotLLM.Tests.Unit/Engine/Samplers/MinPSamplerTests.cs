using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class MinPSamplerTests
{
    private readonly MinPSampler _sampler = new();

    [Fact]
    public void Apply_MasksTokensBelowRelativeThreshold()
    {
        // Create a peaked distribution: one high, rest low
        float[] logits = [10.0f, 0.0f, -5.0f, -10.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0.1f, Seed: null);

        _sampler.Apply(logits, context);

        // The dominant token should always survive
        Assert.False(float.IsNegativeInfinity(logits[0]));
        // Very low probability tokens should be masked
        Assert.True(float.IsNegativeInfinity(logits[3]));
    }

    [Fact]
    public void Apply_MinP0_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Apply_HighMinP_KeepsOnlyDominantTokens()
    {
        // Very peaked distribution
        float[] logits = [10.0f, 0.0f, 0.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0.5f, Seed: null);

        _sampler.Apply(logits, context);

        // Only the dominant token should survive
        Assert.False(float.IsNegativeInfinity(logits[0]));
        Assert.True(float.IsNegativeInfinity(logits[1]));
        Assert.True(float.IsNegativeInfinity(logits[2]));
    }

    [Fact]
    public void Apply_UniformDistribution_KeepsAll()
    {
        float[] logits = [1.0f, 1.0f, 1.0f, 1.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0.5f, Seed: null);

        _sampler.Apply(logits, context);

        // All should survive since all probs are equal
        for (int i = 0; i < logits.Length; i++)
            Assert.False(float.IsNegativeInfinity(logits[i]));
    }
}
