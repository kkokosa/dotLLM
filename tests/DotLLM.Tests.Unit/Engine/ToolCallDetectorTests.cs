using DotLLM.Engine;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

public class ToolCallDetectorTests
{
    [Fact]
    public void DetectToolCalls_WithToolCall_EnrichesResponse()
    {
        var response = new InferenceResponse
        {
            GeneratedTokenIds = [],
            Text = """{"name": "get_weather", "arguments": {"location": "Paris"}}""",
            FinishReason = FinishReason.Stop,
            PromptTokenCount = 10,
            GeneratedTokenCount = 5
        };

        var enriched = ToolCallDetector.DetectToolCalls(response, new GenericToolCallParser());

        Assert.Equal(FinishReason.ToolCalls, enriched.FinishReason);
        Assert.NotNull(enriched.ToolCalls);
        Assert.Single(enriched.ToolCalls);
        Assert.Equal("get_weather", enriched.ToolCalls[0].FunctionName);
    }

    [Fact]
    public void DetectToolCalls_WithoutToolCall_ReturnsOriginal()
    {
        var response = new InferenceResponse
        {
            GeneratedTokenIds = [],
            Text = "The weather is sunny!",
            FinishReason = FinishReason.Stop,
            PromptTokenCount = 10,
            GeneratedTokenCount = 5
        };

        var result = ToolCallDetector.DetectToolCalls(response, new GenericToolCallParser());

        Assert.Same(response, result);
        Assert.Null(result.ToolCalls);
        Assert.Equal(FinishReason.Stop, result.FinishReason);
    }

    [Fact]
    public void DetectToolCalls_EmptyText_ReturnsOriginal()
    {
        var response = new InferenceResponse
        {
            GeneratedTokenIds = [],
            Text = "",
            FinishReason = FinishReason.Length,
            PromptTokenCount = 10,
            GeneratedTokenCount = 0
        };

        var result = ToolCallDetector.DetectToolCalls(response, new GenericToolCallParser());

        Assert.Same(response, result);
    }

    [Fact]
    public void DetectToolCalls_PreservesOtherFields()
    {
        var response = new InferenceResponse
        {
            GeneratedTokenIds = [1, 2, 3],
            Text = """<tool_call>{"name": "func", "arguments": {}}</tool_call>""",
            FinishReason = FinishReason.Stop,
            PromptTokenCount = 42,
            GeneratedTokenCount = 3,
            Timings = new InferenceTimings { PrefillTimeMs = 100 }
        };

        var enriched = ToolCallDetector.DetectToolCalls(response, new HermesToolCallParser());

        Assert.Equal(new[] { 1, 2, 3 }, enriched.GeneratedTokenIds);
        Assert.Equal(42, enriched.PromptTokenCount);
        Assert.Equal(3, enriched.GeneratedTokenCount);
        Assert.Equal(100, enriched.Timings.PrefillTimeMs);
    }

    [Fact]
    public void DetectToolCalls_ParallelCalls_AllDetected()
    {
        var response = new InferenceResponse
        {
            GeneratedTokenIds = [],
            Text = """[TOOL_CALLS][{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]""",
            FinishReason = FinishReason.Stop,
            PromptTokenCount = 10,
            GeneratedTokenCount = 5
        };

        var enriched = ToolCallDetector.DetectToolCalls(response, new MistralToolCallParser());

        Assert.Equal(FinishReason.ToolCalls, enriched.FinishReason);
        Assert.NotNull(enriched.ToolCalls);
        Assert.Equal(2, enriched.ToolCalls.Length);
    }
}
