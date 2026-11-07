using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

public sealed class StopSequenceConversionTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("null")]
    [InlineData("\"\"")]
    [InlineData("[]")]
    [InlineData("[\"\",null]")]
    public void CompletionWithoutEffectiveStops_HasEmptyStopList(string? json)
    {
        var request = new CompletionRequest { Prompt = "hello", Stop = Parse(json) };
        var options = RequestConverter.ToInferenceOptions(request, new SamplingDefaults(), ThreadingConfig.SingleThreaded);
        Assert.Empty(options.StopSequences);
    }

    [Theory]
    [InlineData("\"end\"")]
    [InlineData("[\"end\",\"\"]")]
    public void ChatStops_PreserveDefaultsAndRequestOrder(string json)
    {
        string[] defaults = ["eos", "chat-end"];
        var request = new ChatCompletionRequest { Messages = [], Stop = Parse(json) };
        var options = RequestConverter.ToInferenceOptions(request, defaults, new SamplingDefaults(), ThreadingConfig.SingleThreaded);
        Assert.Equal(new[] { "eos", "chat-end", "end" }, options.StopSequences);
        defaults[0] = "changed";
        Assert.Equal("eos", options.StopSequences[0]);
    }

    [Fact]
    public void ArrayStops_PreserveDuplicatesAndSkipNullOrEmptyStrings()
    {
        var request = new CompletionRequest
        {
            Prompt = "hello",
            Stop = Parse("[\"end\",null,\"\",\"end\",\"finish\"]"),
        };
        var options = RequestConverter.ToInferenceOptions(request, new SamplingDefaults(), ThreadingConfig.SingleThreaded);
        Assert.Equal(new[] { "end", "end", "finish" }, options.StopSequences);
    }

    [Fact]
    public void ChatWithoutRequestStops_CopiesDefaults()
    {
        string[] defaults = ["eos", "chat-end"];
        var request = new ChatCompletionRequest { Messages = [] };
        var options = RequestConverter.ToInferenceOptions(request, defaults, new SamplingDefaults(), ThreadingConfig.SingleThreaded);
        defaults[0] = "changed";
        Assert.Equal(new[] { "eos", "chat-end" }, options.StopSequences);
    }

    [Fact]
    public void InvalidArrayElement_RetainsConversionError()
    {
        var request = new CompletionRequest { Prompt = "hello", Stop = Parse("[123]") };
        Assert.Throws<InvalidOperationException>(() => RequestConverter.ToInferenceOptions(
            request, new SamplingDefaults(), ThreadingConfig.SingleThreaded));
    }

    private static JsonElement? Parse(string? json)
    {
        if (json is null) return null;
        using var document = JsonDocument.Parse(json);
        return document.RootElement.Clone();
    }
}
