using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

public class GenericToolCallParserTests
{
    private readonly GenericToolCallParser _parser = new();

    [Fact]
    public void TryParse_JsonWithArguments()
    {
        const string text = """{"name": "get_weather", "arguments": {"location": "London"}}""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);
        Assert.Contains("London", calls[0].Arguments);
    }

    [Fact]
    public void TryParse_JsonWithParameters()
    {
        const string text = """{"name": "get_weather", "parameters": {"location": "Tokyo"}}""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);
        Assert.Contains("Tokyo", calls[0].Arguments);
    }

    [Fact]
    public void TryParse_JsonEmbeddedInText()
    {
        const string text = """Sure! Let me call the function: {"name": "search", "arguments": {"query": "dotnet"}} and get results.""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("search", calls![0].FunctionName);
    }

    [Fact]
    public void TryParse_JsonArray()
    {
        const string text = """[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]""";

        var calls = _parser.TryParse(text);

        Assert.NotNull(calls);
        Assert.Equal(2, calls!.Length);
    }

    [Fact]
    public void TryParse_PlainText_ReturnsNull()
    {
        const string text = "This is just regular text with no JSON.";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void TryParse_JsonWithoutNameKey_ReturnsNull()
    {
        const string text = """{"type": "function", "data": "test"}""";

        var calls = _parser.TryParse(text);

        Assert.Null(calls);
    }

    [Fact]
    public void TryParse_EmptyString_ReturnsNull()
    {
        Assert.Null(_parser.TryParse(""));
        Assert.Null(_parser.TryParse("   "));
    }

    [Fact]
    public void IsToolCallStart_WithJsonAndNameKey_ReturnsTrue()
    {
        Assert.True(_parser.IsToolCallStart("""{"name": "func"""));
    }

    [Fact]
    public void IsToolCallStart_NoBrace_ReturnsFalse()
    {
        Assert.False(_parser.IsToolCallStart("no json here"));
    }

    [Fact]
    public void IsToolCallStart_BraceButNoName_ReturnsFalse()
    {
        Assert.False(_parser.IsToolCallStart("""{"type": "test"}"""));
    }
}
