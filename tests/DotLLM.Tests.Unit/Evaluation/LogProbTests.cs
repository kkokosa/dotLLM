using DotLLM.Engine.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

public sealed class LogProbTests
{
    [Fact]
    public void OfTarget_UniformLogits_IsNegativeLogVocab()
    {
        // Four equal logits => p = 1/4 for each => log p = -log 4.
        var logits = new float[] { 2.5f, 2.5f, 2.5f, 2.5f };
        double actual = LogProb.OfTarget(logits, target: 2);
        Assert.Equal(-Math.Log(4.0), actual, 12);
    }

    [Fact]
    public void OfTarget_IsShiftInvariant()
    {
        var a = new float[] { 1f, 2f, 3f };
        var b = new float[] { 1001f, 1002f, 1003f };
        Assert.Equal(LogProb.OfTarget(a, 1), LogProb.OfTarget(b, 1), 12);
    }

    [Fact]
    public void OfTarget_LargeLogits_DoesNotOverflow()
    {
        // Naive exp() would overflow to infinity here; the max-shift must prevent it.
        var logits = new float[] { 800f, 900f, 1000f };
        double actual = LogProb.OfTarget(logits, target: 2);
        Assert.True(double.IsFinite(actual));
        Assert.Equal(0.0, actual, 6);   // target dominates => p ~ 1 => log p ~ 0
    }

    [Fact]
    public void OfTarget_SumOfProbabilitiesIsOne()
    {
        var logits = new float[] { -1.5f, 0.25f, 3f, 0.5f };
        double sum = 0;
        for (int i = 0; i < logits.Length; i++) sum += Math.Exp(LogProb.OfTarget(logits, i));
        Assert.Equal(1.0, sum, 10);
    }
}
