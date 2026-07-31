using System.Numerics.Tensors;
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

    /// <summary>
    /// Reference implementation that mirrors the pre-cutoff algorithm: full softmax,
    /// full O(V log V) sort, descending cumulative-mass walk. Used to verify that the
    /// production sampler (which prunes via Karpathy's `(1 - topP) / (n - 1)` cutoff
    /// before sorting) produces bit-identical masked logits for non-tie distributions.
    /// </summary>
    private static void ApplyReference(Span<float> logits, float topP)
    {
        if (topP >= 1.0f) return;

        int vocab = logits.Length;
        var probs = new float[vocab];
        var indices = new int[vocab];
        TensorPrimitives.SoftMax(logits, probs);
        for (int i = 0; i < vocab; i++) indices[i] = i;
        Array.Sort(probs, indices, 0, vocab);

        float cumulative = 0f;
        int cutoffCount = vocab;
        for (int i = vocab - 1; i >= 0; i--)
        {
            cumulative += probs[i];
            if (cumulative >= topP)
            {
                cutoffCount = vocab - i;
                break;
            }
        }

        var keep = new bool[vocab];
        int keepStart = vocab - cutoffCount;
        for (int i = keepStart; i < vocab; i++)
            keep[indices[i]] = true;

        for (int i = 0; i < vocab; i++)
            if (!keep[i])
                logits[i] = float.NegativeInfinity;
    }

    /// <summary>
    /// Bit-exact parity test: the pre-cutoff optimization must produce identical masked
    /// logits to the original full-sort algorithm for a realistically large vocab with
    /// random (non-tied) probabilities. This is the key correctness guarantee.
    /// <para>
    /// Ties are deliberately out of scope: neither algorithm defines which of several
    /// equal-probability tokens wins, because <c>Array.Sort</c> is an unstable IntroSort.
    /// This matters more than it looks at large vocab — with a near-uniform distribution
    /// over 32K+ tokens, distinct logits can round to the <i>same</i> float probability,
    /// so a test that pins a specific surviving index would be asserting sort internals.
    /// The spread used here (~[-10, 10)) keeps probabilities well separated.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData(32_000, 0.9f, 1)]
    [InlineData(32_000, 0.95f, 2)]
    [InlineData(128_000, 0.9f, 3)]
    [InlineData(128_000, 0.5f, 4)]
    [InlineData(128_000, 0.99f, 5)]
    public void Apply_BitExactParityWithFullSort(int vocabSize, float topP, int seed)
    {
        var rng = new Random(seed);
        var refLogits = new float[vocabSize];
        var optLogits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            // Random in roughly [-10, 10) — produces no engineered ties.
            float v = (float)(rng.NextDouble() * 20.0 - 10.0);
            refLogits[i] = v;
            optLogits[i] = v;
        }

        var context = new SamplerContext(
            Temperature: 1.0f, TopK: 0, TopP: topP, MinP: 0f, Seed: null);

        ApplyReference(refLogits, topP);
        _sampler.Apply(optLogits, context);

        for (int i = 0; i < vocabSize; i++)
        {
            // Either both masked or both surviving with the exact same logit value.
            bool refMasked = float.IsNegativeInfinity(refLogits[i]);
            bool optMasked = float.IsNegativeInfinity(optLogits[i]);
            Assert.Equal(refMasked, optMasked);
            if (!refMasked)
                Assert.Equal(refLogits[i], optLogits[i]);
        }
    }

    /// <summary>
    /// Degenerate regime the pre-filter alone cannot handle: when <c>topP &lt; 1/vocabSize</c>
    /// the cutoff <c>(1 - topP) / (n - 1)</c> exceeds every probability, so the filter empties
    /// the candidate set. Top-p must still keep at least one token (the argmax) rather than
    /// masking the whole vocabulary. Cases below are all in that regime — e.g. vocab=2 with
    /// topP=0.1 gives cutoff=0.9 against two probabilities that cannot both reach it.
    /// </summary>
    [Theory]
    [InlineData(2, 0.1f)]
    [InlineData(2, 0.4f)]
    [InlineData(3, 0.2f)]
    [InlineData(4, 0.1f)]
    [InlineData(4, 0f)]
    public void Apply_CutoffFiltersEverything_KeepsArgmax(int vocabSize, float topP)
    {
        var rng = new Random(vocabSize * 31 + (int)(topP * 1000));
        var refLogits = new float[vocabSize];
        var optLogits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            // Near-uniform: small spread keeps every probability under the cutoff while
            // still giving a unique argmax (no ties to disambiguate).
            float v = (float)(rng.NextDouble() * 0.01);
            refLogits[i] = v;
            optLogits[i] = v;
        }

        int expectedArgmax = TensorPrimitives.IndexOfMax(optLogits);
        float expectedValue = optLogits[expectedArgmax];

        var context = new SamplerContext(
            Temperature: 1.0f, TopK: 0, TopP: topP, MinP: 0f, Seed: null);

        ApplyReference(refLogits, topP);
        _sampler.Apply(optLogits, context);

        // Exactly one survivor, and it is the argmax.
        Assert.Equal(expectedValue, optLogits[expectedArgmax]);
        for (int i = 0; i < vocabSize; i++)
        {
            if (i != expectedArgmax)
                Assert.True(float.IsNegativeInfinity(optLogits[i]),
                    $"token {i} should be masked (vocab={vocabSize}, topP={topP})");
        }

        // ...and that matches what the un-filtered full-sort algorithm does.
        for (int i = 0; i < vocabSize; i++)
        {
            Assert.Equal(float.IsNegativeInfinity(refLogits[i]), float.IsNegativeInfinity(optLogits[i]));
            if (!float.IsNegativeInfinity(refLogits[i]))
                Assert.Equal(refLogits[i], optLogits[i]);
        }
    }

    /// <summary>
    /// The same "filter empties the candidate set" regime, but at production vocab size:
    /// a perfectly uniform distribution puts every probability at <c>1/n</c>, which is
    /// strictly below the cutoff <c>1/(n - 1)</c> when <c>topP = 0</c>. Exactly one token
    /// must survive. Which one is unspecified here — every probability is tied, so the
    /// un-filtered reference's choice depends on an unstable sort — hence this asserts the
    /// survivor count rather than a specific index.
    /// </summary>
    [Fact]
    public void Apply_UniformLargeVocabWithZeroTopP_KeepsExactlyOneToken()
    {
        const int vocabSize = 32_000;
        var logits = new float[vocabSize]; // all zero → perfectly uniform softmax

        var context = new SamplerContext(
            Temperature: 1.0f, TopK: 0, TopP: 0f, MinP: 0f, Seed: null);

        _sampler.Apply(logits, context);

        int survivors = 0;
        for (int i = 0; i < vocabSize; i++)
            if (!float.IsNegativeInfinity(logits[i]))
                survivors++;

        Assert.Equal(1, survivors);
    }

    /// <summary>
    /// Differential fuzz: sweeps small vocab sizes (where the cutoff is coarse and the
    /// boundary cases cluster) against a wide range of topP values, asserting bit-exact
    /// parity with the un-filtered reference on every draw. Small vocabs make this cheap
    /// while covering far more of the input space than a handful of large fixed cases.
    /// </summary>
    [Fact]
    public void Apply_SmallVocabFuzz_BitExactParityWithFullSort()
    {
        float[] topPs = [0f, 0.01f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.99f];
        var rng = new Random(20250730);

        for (int vocabSize = 2; vocabSize <= 64; vocabSize++)
        {
            foreach (float topP in topPs)
            {
                for (int trial = 0; trial < 8; trial++)
                {
                    var refLogits = new float[vocabSize];
                    var optLogits = new float[vocabSize];
                    // Spread varies per trial: tight spreads give near-uniform
                    // distributions (cutoff-empties-everything regime), wide spreads
                    // give peaked ones (normal regime).
                    double spread = trial % 2 == 0 ? 0.05 : 12.0;
                    for (int i = 0; i < vocabSize; i++)
                    {
                        float v = (float)((rng.NextDouble() - 0.5) * spread);
                        refLogits[i] = v;
                        optLogits[i] = v;
                    }

                    var context = new SamplerContext(
                        Temperature: 1.0f, TopK: 0, TopP: topP, MinP: 0f, Seed: null);

                    ApplyReference(refLogits, topP);
                    _sampler.Apply(optLogits, context);

                    for (int i = 0; i < vocabSize; i++)
                    {
                        bool refMasked = float.IsNegativeInfinity(refLogits[i]);
                        bool optMasked = float.IsNegativeInfinity(optLogits[i]);
                        if (refMasked != optMasked || (!refMasked && refLogits[i] != optLogits[i]))
                        {
                            Assert.Fail(
                                $"divergence at token {i} (vocab={vocabSize}, topP={topP}, " +
                                $"trial={trial}): reference={refLogits[i]}, optimized={optLogits[i]}");
                        }
                    }
                }
            }
        }
    }
}
