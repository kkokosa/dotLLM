using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class TopPSamplerTests
{
    private readonly TopPSampler _sampler = new();

    [Fact]
    public void Apply_CumulativeProbabilityThreshold()
    {
        // Logits that produce a peaked distribution
        float[] logits = [10.0f, 1.0f, 0.0f, -1.0f, -10.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 0.95f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        // The highest logit should always survive
        Assert.False(float.IsNegativeInfinity(logits[0]));
        // The very low logit should be masked
        Assert.True(float.IsNegativeInfinity(logits[4]));
    }

    [Fact]
    public void Apply_P1_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Apply_VeryLowP_KeepsOnlyTopToken()
    {
        // Very peaked: only one dominant token
        float[] logits = [10.0f, 0.0f, 0.0f, 0.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 0.01f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        // The top token (index 0) should survive, rest masked
        Assert.False(float.IsNegativeInfinity(logits[0]));
        Assert.True(float.IsNegativeInfinity(logits[1]));
        Assert.True(float.IsNegativeInfinity(logits[2]));
        Assert.True(float.IsNegativeInfinity(logits[3]));
    }
}
