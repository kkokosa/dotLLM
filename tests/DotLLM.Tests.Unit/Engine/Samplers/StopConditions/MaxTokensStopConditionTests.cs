using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers.StopConditions;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers.StopConditions;

public class MaxTokensStopConditionTests
{
    [Fact]
    public void ShouldStop_BelowMax_ReturnsContinue()
    {
        var condition = new MaxTokensStopCondition(maxTokens: 5);

        var result = condition.ShouldStop(tokenId: 1, generatedTokens: [1, 2, 3], decodedText: "abc");

        Assert.Equal(StopResult.Continue, result);
    }

    [Fact]
    public void ShouldStop_AtMax_ReturnsStopInclude()
    {
        var condition = new MaxTokensStopCondition(maxTokens: 3);

        var result = condition.ShouldStop(tokenId: 3, generatedTokens: [1, 2, 3], decodedText: "abc");

        Assert.Equal(StopResult.StopInclude, result);
    }

    [Fact]
    public void ShouldStop_AboveMax_ReturnsStopInclude()
    {
        var condition = new MaxTokensStopCondition(maxTokens: 2);

        var result = condition.ShouldStop(tokenId: 3, generatedTokens: [1, 2, 3], decodedText: "abc");

        Assert.Equal(StopResult.StopInclude, result);
    }
}
