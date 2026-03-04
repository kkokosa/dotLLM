using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class RepetitionPenaltyProcessorTests
{
    private readonly RepetitionPenaltyProcessor _processor = new();

    [Fact]
    public void Process_PositiveLogits_DividedByPenalty()
    {
        float[] logits = [2.0f, 4.0f, 6.0f, 8.0f];
        var previousTokens = new List<int> { 1, 3 }; // penalize indices 1 and 3
        var context = new ProcessorContext(RepetitionPenalty: 2.0f, RepetitionPenaltyWindow: 0, SequenceId: 0);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(2.0f, logits[0]); // not penalized
        Assert.Equal(2.0f, logits[1]); // 4.0 / 2.0
        Assert.Equal(6.0f, logits[2]); // not penalized
        Assert.Equal(4.0f, logits[3]); // 8.0 / 2.0
    }

    [Fact]
    public void Process_NegativeLogits_MultipliedByPenalty()
    {
        float[] logits = [1.0f, -2.0f, 3.0f, -4.0f];
        var previousTokens = new List<int> { 1, 3 };
        var context = new ProcessorContext(RepetitionPenalty: 2.0f, RepetitionPenaltyWindow: 0, SequenceId: 0);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(1.0f, logits[0]);  // not penalized
        Assert.Equal(-4.0f, logits[1]); // -2.0 * 2.0
        Assert.Equal(3.0f, logits[2]);  // not penalized
        Assert.Equal(-8.0f, logits[3]); // -4.0 * 2.0
    }

    [Fact]
    public void Process_Penalty1_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var previousTokens = new List<int> { 0, 1, 2 };
        var context = new ProcessorContext(RepetitionPenalty: 1.0f, RepetitionPenaltyWindow: 0, SequenceId: 0);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Process_EmptyHistory_Skips()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var previousTokens = new List<int>();
        var context = new ProcessorContext(RepetitionPenalty: 2.0f, RepetitionPenaltyWindow: 0, SequenceId: 0);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(original, logits);
    }

    [Fact]
    public void Process_WindowRespectsLookback()
    {
        float[] logits = [4.0f, 4.0f, 4.0f, 4.0f];
        // Tokens: 0, 1, 2, 3 in order; window=2 → only look at last 2 (tokens 2, 3)
        var previousTokens = new List<int> { 0, 1, 2, 3 };
        var context = new ProcessorContext(RepetitionPenalty: 2.0f, RepetitionPenaltyWindow: 2, SequenceId: 0);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(4.0f, logits[0]); // not in window
        Assert.Equal(4.0f, logits[1]); // not in window
        Assert.Equal(2.0f, logits[2]); // in window, penalized
        Assert.Equal(2.0f, logits[3]); // in window, penalized
    }

    [Fact]
    public void Process_DuplicateTokensInHistory_HandledCorrectly()
    {
        float[] logits = [4.0f, 4.0f, 4.0f];
        var previousTokens = new List<int> { 1, 1, 1 }; // duplicate token 1
        var context = new ProcessorContext(RepetitionPenalty: 2.0f, RepetitionPenaltyWindow: 0, SequenceId: 0);

        _processor.Process(logits, previousTokens, context);

        Assert.Equal(4.0f, logits[0]); // not penalized
        Assert.Equal(2.0f, logits[1]); // penalized once (not triple)
        Assert.Equal(4.0f, logits[2]); // not penalized
    }
}
