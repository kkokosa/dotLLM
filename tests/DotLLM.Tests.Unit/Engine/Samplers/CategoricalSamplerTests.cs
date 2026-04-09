using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class CategoricalSamplerTests
{
    [Fact]
    public void Sample_ReturnsValidIndex()
    {
        float[] logits = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];
        var rng = new Random(42);

        int result = CategoricalSampler.Sample(logits, rng);

        Assert.InRange(result, 0, logits.Length - 1);
    }

    [Fact]
    public void Sample_SeededDeterminism()
    {
        float[] logits = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f];

        int result1 = CategoricalSampler.Sample(logits, new Random(42));
        int result2 = CategoricalSampler.Sample(logits, new Random(42));

        Assert.Equal(result1, result2);
    }

    [Fact]
    public void Sample_PeakedDistribution_FavorsMax()
    {
        // One very high logit, rest are -inf → softmax gives prob 1.0 at index 0
        float[] logits = [10.0f, float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity];

        // Should always pick index 0
        for (int i = 0; i < 10; i++)
        {
            int result = CategoricalSampler.Sample(logits, new Random(i));
            Assert.Equal(0, result);
        }
    }

    [Fact]
    public void Sample_NegativeInfinityNeverSelected()
    {
        // Only index 2 has a valid logit
        float[] logits = [float.NegativeInfinity, float.NegativeInfinity, 1.0f, float.NegativeInfinity];
        var rng = new Random(42);

        for (int i = 0; i < 10; i++)
        {
            int result = CategoricalSampler.Sample(logits, new Random(i));
            Assert.Equal(2, result);
        }
    }

    [Fact]
    public void Sample_UniformDistribution_ProducesVariety()
    {
        float[] logits = [0f, 0f, 0f, 0f];
        var seen = new HashSet<int>();

        // With enough samples, should see multiple indices
        for (int i = 0; i < 100; i++)
        {
            int result = CategoricalSampler.Sample(logits, new Random(i));
            seen.Add(result);
        }

        Assert.True(seen.Count > 1, "Uniform distribution should produce variety over 100 samples.");
    }

    [Fact]
    public void Sample_FallbackReturnsHighestProbToken()
    {
        // Construct logits where index 2 has the highest logit.
        // The fallback should return index 2, not vocabSize-1 (index 4).
        float[] logits = [-10f, -10f, 10f, -10f, -10f];

        // With such peaked logits, softmax gives ~1.0 to index 2.
        // All seeds should pick index 2 (either through normal sampling or fallback).
        for (int i = 0; i < 20; i++)
        {
            int result = CategoricalSampler.Sample(logits, new Random(i));
            Assert.Equal(2, result);
        }
    }
}
