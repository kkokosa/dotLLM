using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

public class LlamaToolCallParserTests
{
    private readonly LlamaToolCallParser _parser = new();

    [Fact]
    public void TryParse_SingleCall_WithMarker()
    {
        const string text = """<|python_tag|>{"name": "get_weather", "parameters": {"location": "Paris"}}""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls[0].FunctionName);
        Assert.Contains("Paris", calls[0].Arguments);
    }

    [Fact]
    public void TryParse_SingleCall_NormalizesParametersToArguments()
    {
        const string text = """<|python_tag|>{"name": "search", "parameters": {"query": "dotnet"}}""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        // Arguments field should contain the parameters JSON
        Assert.Contains("dotnet", calls![0].Arguments);
    }

    [Fact]
    public void TryParse_ParallelCalls_JsonArray()
    {
        const string text = """
            <|python_tag|>[
              {"name": "get_weather", "parameters": {"location": "Paris"}},
              {"name": "get_time", "parameters": {"timezone": "CET"}}
            ]
            """;

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Equal(2, calls!.Length);
        Assert.Equal("get_weather", calls[0].FunctionName);
        Assert.Equal("get_time", calls[1].FunctionName);
    }

    [Fact]
    public void TryParse_WithLeadingText_BeforeMarker()
    {
        const string text = """I'll check the weather for you. <|python_tag|>{"name": "get_weather", "parameters": {"location": "London"}}""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);
    }

    [Fact]
    public void TryParse_WithoutMarker_FallsBackToJsonDetection()
    {
        const string text = """{"name": "get_weather", "parameters": {"location": "Tokyo"}}""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);
    }

    [Fact]
    public void TryParse_NoToolCall_ReturnsNull()
    {
        const string text = "The weather in Paris is sunny and warm today!";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void TryParse_MalformedJson_ReturnsNull()
    {
        const string text = """<|python_tag|>{"name": "get_weather", "parameters": {invalid""";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void TryParse_GeneratesCallIds()
    {
        const string text = """<|python_tag|>{"name": "func", "parameters": {}}""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Equal("call_0", calls![0].Id);
    }

    [Fact]
    public void TryParse_ParallelCalls_SequentialIds()
    {
        const string text = """<|python_tag|>[{"name": "a", "parameters": {}}, {"name": "b", "parameters": {}}]""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Equal("call_0", calls![0].Id);
        Assert.Equal("call_1", calls[1].Id);
    }

    [Fact]
    public void IsToolCallStart_WithMarker_ReturnsTrue()
    {
        Assert.True(_parser.IsToolCallStart("Some text <|python_tag|>"));
    }

    [Fact]
    public void IsToolCallStart_WithoutMarker_ReturnsFalse()
    {
        Assert.False(_parser.IsToolCallStart("Just regular text"));
    }

    [Fact]
    public void TryParse_ArgumentsKey_AlsoWorks()
    {
        const string text = """<|python_tag|>{"name": "search", "arguments": {"query": "test"}}""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Contains("test", calls![0].Arguments);
    }
}
