using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers.StopConditions;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers.StopConditions;

public class EosStopConditionTests
{
    [Fact]
    public void ShouldStop_EosToken_ReturnsStop()
    {
        var condition = new EosStopCondition(eosTokenId: 2);

        var result = condition.ShouldStop(tokenId: 2, generatedTokens: [2], decodedText: "");

        Assert.Equal(StopResult.Stop, result);
    }

    [Fact]
    public void ShouldStop_NonEosToken_ReturnsContinue()
    {
        var condition = new EosStopCondition(eosTokenId: 2);

        var result = condition.ShouldStop(tokenId: 5, generatedTokens: [5], decodedText: "hello");

        Assert.Equal(StopResult.Continue, result);
    }
}
