using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class TopKSamplerTests
{
    private readonly TopKSampler _sampler = new();

    [Fact]
    public void Apply_KeepsTopK_MasksRest()
    {
        float[] logits = [1.0f, 5.0f, 3.0f, 2.0f, 4.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 2, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        // Top 2 are indices 1 (5.0) and 4 (4.0)
        Assert.True(float.IsNegativeInfinity(logits[0])); // 1.0 masked
        Assert.Equal(5.0f, logits[1]);                     // kept
        Assert.True(float.IsNegativeInfinity(logits[2])); // 3.0 masked
        Assert.True(float.IsNegativeInfinity(logits[3])); // 2.0 masked
        Assert.Equal(4.0f, logits[4]);                     // kept
    }

    [Fact]
    public void Apply_K0_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 0, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Apply_KGreaterThanVocab_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 10, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Apply_K1_KeepsOnlyMax()
    {
        float[] logits = [1.0f, 5.0f, 3.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 1, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.True(float.IsNegativeInfinity(logits[0]));
        Assert.Equal(5.0f, logits[1]);
        Assert.True(float.IsNegativeInfinity(logits[2]));
    }

    [Fact]
    public void Apply_WithTies_KeepsExactlyK()
    {
        // [1, 2, 2, 2, 3] with K=2: should keep exactly 2 tokens (3 and one of the 2s)
        float[] logits = [1.0f, 2.0f, 2.0f, 2.0f, 3.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 2, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        int keptCount = logits.Count(v => !float.IsNegativeInfinity(v));
        Assert.Equal(2, keptCount);
        // The max value (3.0) must always be kept
        Assert.Equal(3.0f, logits[4]);
    }

    [Fact]
    public void Apply_AllSameValue_KeepsExactlyK()
    {
        float[] logits = [5.0f, 5.0f, 5.0f, 5.0f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 2, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        int keptCount = logits.Count(v => !float.IsNegativeInfinity(v));
        Assert.Equal(2, keptCount);
    }
}
