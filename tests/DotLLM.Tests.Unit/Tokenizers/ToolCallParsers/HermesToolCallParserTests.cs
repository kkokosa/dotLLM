using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

public class HermesToolCallParserTests
{
    private readonly HermesToolCallParser _parser = new();

    [Fact]
    public void TryParse_SingleToolCall()
    {
        const string text = """<tool_call>{"name": "get_weather", "arguments": {"location": "Berlin"}}</tool_call>""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);
        Assert.Contains("Berlin", calls[0].Arguments);
    }

    [Fact]
    public void TryParse_MultipleToolCalls_ParallelBlocks()
    {
        const string text = """
            <tool_call>{"name": "get_weather", "arguments": {"location": "Berlin"}}</tool_call>
            <tool_call>{"name": "get_time", "arguments": {"timezone": "CET"}}</tool_call>
            """;

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Equal(2, calls!.Length);
        Assert.Equal("get_weather", calls[0].FunctionName);
        Assert.Equal("get_time", calls[1].FunctionName);
    }

    [Fact]
    public void TryParse_WithSurroundingText()
    {
        const string text = """I'll look that up for you. <tool_call>{"name": "search", "arguments": {"query": "dotnet"}}</tool_call> Done.""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("search", calls![0].FunctionName);
    }

    [Fact]
    public void TryParse_MissingCloseTag_ParsesPartial()
    {
        const string text = """<tool_call>{"name": "get_weather", "arguments": {"location": "Tokyo"}}""";

        var calls = _parser.TryParse(text);

        // Should still parse since the JSON is complete
        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);
    }

    [Fact]
    public void TryParse_NoToolCall_ReturnsNull()
    {
        const string text = "Just a regular response with no tool calls.";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void TryParse_MalformedJson_ReturnsNull()
    {
        const string text = """<tool_call>not valid json</tool_call>""";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void TryParse_NestedJsonInArguments()
    {
        const string text = """<tool_call>{"name": "create", "arguments": {"config": {"nested": true, "items": [1,2,3]}}}</tool_call>""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Contains("nested", calls![0].Arguments);
        Assert.Contains("[1,2,3]", calls[0].Arguments);
    }

    [Fact]
    public void TryParse_SequentialIds()
    {
        const string text = """
            <tool_call>{"name": "a", "arguments": {}}</tool_call>
            <tool_call>{"name": "b", "arguments": {}}</tool_call>
            """;

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Equal("call_0", calls![0].Id);
        Assert.Equal("call_1", calls[1].Id);
    }

    [Fact]
    public void IsToolCallStart_WithTag_ReturnsTrue()
    {
        Assert.True(_parser.IsToolCallStart("Let me check <tool_call>"));
    }

    [Fact]
    public void IsToolCallStart_WithoutTag_ReturnsFalse()
    {
        Assert.False(_parser.IsToolCallStart("Regular text without tags"));
    }

    [Fact]
    public void TryParse_UnquotedKeys_StillParsed()
    {
        // Qwen2.5 outputs JavaScript-style JSON with unquoted keys
        const string text = """<tool_call>{name: "get_weather", arguments: {"location": "Paris"}}</tool_call>""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);
        Assert.Contains("Paris", calls[0].Arguments);
    }

    [Fact]
    public void TryParse_DoubleBraces_StillParsed()
    {
        // Some models emit doubled braces (Jinja2 escaping artifact)
        const string text = """<tool_call>{{name: "search", arguments: {"q": "test"}}}</tool_call>""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("search", calls![0].FunctionName);
    }
}
