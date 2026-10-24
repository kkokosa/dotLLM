using System.Text.Json;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

public sealed class ChoiceCountValidationTests
{
    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(10)]
    public void ExplicitUnsupportedChoiceCount_ReturnsError(int choices)
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new() { Role = "user", Content = "hello" }],
            N = choices,
        };
        Assert.Equal("n must be 1; multiple completion choices are not supported",
            RequestValidator.ValidateChatRequest(request));
    }

    [Theory]
    [InlineData("{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}")]
    [InlineData("{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"n\":1}")]
    public void DefaultOrSingleChoice_IsAccepted(string json)
    {
        var request = JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest)!;
        Assert.Equal(1, request.N);
        Assert.Null(RequestValidator.ValidateChatRequest(request));
    }
}
