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

    [Theory]
    [InlineData(128_000, 40)]     // typical Llama vocab, typical K
    [InlineData(128_000, 100)]    // larger K
    [InlineData(128_000, 1)]      // greedy degenerate
    [InlineData(128_000, 511)]    // just under stack-heap threshold
    [InlineData(128_000, 513)]    // just over stack-heap threshold (ArrayPool path)
    [InlineData(32_000, 40)]      // SmolLM-sized vocab
    public void Apply_LargeVocab_MatchesFullSortReference(int vocabSize, int k)
    {
        // Deterministic pseudo-random logits so the test is reproducible
        var rng = new Random(42);
        float[] logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
            logits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);

        // Reference: full sort to find the K-th largest
        float[] sortedRef = (float[])logits.Clone();
        Array.Sort(sortedRef);
        float expectedThreshold = sortedRef[vocabSize - k];

        float[] original = (float[])logits.Clone();
        var context = new SamplerContext(Temperature: 1.0f, TopK: k, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        // Exactly K tokens survive
        int keptCount = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            if (!float.IsNegativeInfinity(logits[i]))
            {
                keptCount++;
                // Every surviving value must be ≥ threshold, and must equal the original (not mutated)
                Assert.True(logits[i] >= expectedThreshold);
                Assert.Equal(original[i], logits[i]);
            }
        }
        Assert.Equal(k, keptCount);

        // Every strictly-above-threshold token in the original must have survived
        for (int i = 0; i < vocabSize; i++)
        {
            if (original[i] > expectedThreshold)
                Assert.False(float.IsNegativeInfinity(logits[i]));
        }
    }

    [Fact]
    public void Apply_LogitsUnchangedWhenKeptExactly()
    {
        // Regression: the old implementation rented a scratch buffer and sorted it;
        // the new min-heap path must not mutate the original logits array at all
        // for values above the threshold.
        float[] logits = [0.1f, 9.9f, 0.5f, 8.8f, 0.3f];
        float[] expectedKept = [9.9f, 8.8f];
        var context = new SamplerContext(Temperature: 1.0f, TopK: 2, TopP: 1.0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        Assert.Equal(9.9f, logits[1]);
        Assert.Equal(8.8f, logits[3]);
        Assert.True(float.IsNegativeInfinity(logits[0]));
        Assert.True(float.IsNegativeInfinity(logits[2]));
        Assert.True(float.IsNegativeInfinity(logits[4]));
    }
}
