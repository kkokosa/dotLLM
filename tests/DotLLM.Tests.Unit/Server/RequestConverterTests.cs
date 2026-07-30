using DotLLM.Core.Configuration;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

public class RequestConverterTests
{
    [Fact]
    public void ToInferenceOptions_ChatRequest_MapsLogitBias()
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = "hi" }],
            LogitBias = new Dictionary<int, float>
            {
                [1] = 0.5f,
                [2] = -1.0f,
            },
        };

        var options = RequestConverter.ToInferenceOptions(
            request,
            stopSequences: [],
            defaults: new SamplingDefaults(),
            threading: ThreadingConfig.Auto);

        Assert.NotNull(options.LogitBias);
        Assert.Equal(0.5f, options.LogitBias![1], precision: 5);
        Assert.Equal(-1.0f, options.LogitBias[2], precision: 5);
    }

    [Fact]
    public void ToInferenceOptions_CompletionRequest_MapsLogitBias()
    {
        var request = new CompletionRequest
        {
            Prompt = "hi",
            LogitBias = new Dictionary<int, float>
            {
                [3] = 2.5f,
            },
        };

        var options = RequestConverter.ToInferenceOptions(
            request,
            defaults: new SamplingDefaults(),
            threading: ThreadingConfig.Auto);

        Assert.NotNull(options.LogitBias);
        Assert.Equal(2.5f, options.LogitBias![3], precision: 5);
    }
}
