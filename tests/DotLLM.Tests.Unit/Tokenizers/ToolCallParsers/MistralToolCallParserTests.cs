using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

public class MistralToolCallParserTests
{
    private readonly MistralToolCallParser _parser = new();

    [Fact]
    public void TryParse_SingleElementArray()
    {
        const string text = """[TOOL_CALLS][{"name": "get_weather", "arguments": {"location": "Paris"}}]""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);
        Assert.Contains("Paris", calls[0].Arguments);
    }

    [Fact]
    public void TryParse_MultipleElements_ParallelCalls()
    {
        const string text = """
            [TOOL_CALLS][
              {"name": "get_weather", "arguments": {"location": "Paris"}},
              {"name": "get_time", "arguments": {"timezone": "CET"}}
            ]
            """;

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Equal(2, calls!.Length);
        Assert.Equal("get_weather", calls[0].FunctionName);
        Assert.Equal("get_time", calls[1].FunctionName);
    }

    [Fact]
    public void TryParse_WithLeadingText()
    {
        const string text = """Sure, I'll check that. [TOOL_CALLS][{"name": "search", "arguments": {"q": "dotnet"}}]""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("search", calls![0].FunctionName);
    }

    [Fact]
    public void TryParse_NoMarker_ReturnsNull()
    {
        const string text = """[{"name": "get_weather", "arguments": {"location": "Paris"}}]""";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void TryParse_NoToolCall_ReturnsNull()
    {
        const string text = "The weather is sunny today.";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void TryParse_MalformedJson_ReturnsNull()
    {
        const string text = """[TOOL_CALLS]not json""";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void IsToolCallStart_WithMarker_ReturnsTrue()
    {
        Assert.True(_parser.IsToolCallStart("Some text [TOOL_CALLS]"));
    }

    [Fact]
    public void IsToolCallStart_WithoutMarker_ReturnsFalse()
    {
        Assert.False(_parser.IsToolCallStart("Regular text"));
    }
}
