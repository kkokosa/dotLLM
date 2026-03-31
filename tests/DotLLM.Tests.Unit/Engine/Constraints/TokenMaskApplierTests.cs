using DotLLM.Core.Constraints;
using DotLLM.Engine.Constraints;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints;

public class TokenMaskApplierTests
{
    [Fact]
    public void Apply_DisallowedTokens_SetToNegativeInfinity()
    {
        int vocabSize = 10;
        var mask = new TokenMask(vocabSize);
        mask.Allow(0);
        mask.Allow(3);
        mask.Allow(7);

        float[] logits = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f];

        TokenMaskApplier.Apply(logits, mask);

        Assert.Equal(1f, logits[0]);   // allowed
        Assert.Equal(float.NegativeInfinity, logits[1]); // disallowed
        Assert.Equal(float.NegativeInfinity, logits[2]); // disallowed
        Assert.Equal(4f, logits[3]);   // allowed
        Assert.Equal(float.NegativeInfinity, logits[4]); // disallowed
        Assert.Equal(float.NegativeInfinity, logits[5]); // disallowed
        Assert.Equal(float.NegativeInfinity, logits[6]); // disallowed
        Assert.Equal(8f, logits[7]);   // allowed
        Assert.Equal(float.NegativeInfinity, logits[8]); // disallowed
        Assert.Equal(float.NegativeInfinity, logits[9]); // disallowed
    }

    [Fact]
    public void Apply_AllAllowed_LogitsUnchanged()
    {
        int vocabSize = 16;
        var mask = new TokenMask(vocabSize);
        mask.AllowAll();

        float[] logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
            logits[i] = i + 1.0f;

        float[] expected = (float[])logits.Clone();

        TokenMaskApplier.Apply(logits, mask);

        for (int i = 0; i < vocabSize; i++)
            Assert.Equal(expected[i], logits[i]);
    }

    [Fact]
    public void Apply_AllDisallowed_AllNegativeInfinity()
    {
        int vocabSize = 16;
        var mask = new TokenMask(vocabSize);
        // Default is all disallowed

        float[] logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
            logits[i] = i + 1.0f;

        TokenMaskApplier.Apply(logits, mask);

        for (int i = 0; i < vocabSize; i++)
            Assert.Equal(float.NegativeInfinity, logits[i]);
    }

    [Fact]
    public void Apply_LargeVocab_CorrectResults()
    {
        // Test with a vocab size that spans multiple long blocks and exercises SIMD
        int vocabSize = 256;
        var mask = new TokenMask(vocabSize);

        // Allow every 3rd token
        for (int i = 0; i < vocabSize; i += 3)
            mask.Allow(i);

        float[] logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
            logits[i] = i * 0.1f;

        TokenMaskApplier.Apply(logits, mask);

        for (int i = 0; i < vocabSize; i++)
        {
            if (i % 3 == 0)
                Assert.Equal(i * 0.1f, logits[i], 5);
            else
                Assert.Equal(float.NegativeInfinity, logits[i]);
        }
    }

    [Fact]
    public void Apply_NonAligned_VocabSize_Correct()
    {
        // Vocab size not a multiple of 64 — exercises tail handling
        int vocabSize = 100;
        var mask = new TokenMask(vocabSize);
        mask.Allow(0);
        mask.Allow(99);

        float[] logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++)
            logits[i] = 1.0f;

        TokenMaskApplier.Apply(logits, mask);

        Assert.Equal(1.0f, logits[0]);
        Assert.Equal(1.0f, logits[99]);
        for (int i = 1; i < 99; i++)
            Assert.Equal(float.NegativeInfinity, logits[i]);
    }

    [Fact]
    public void Apply_EmptyLogits_NoOp()
    {
        var mask = new TokenMask(0);
        float[] logits = [];
        TokenMaskApplier.Apply(logits, mask); // Should not throw
    }
}
