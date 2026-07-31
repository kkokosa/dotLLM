using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Coverage for <see cref="RequestConverter.IsToolChoiceSupported"/> — the 400 gate that prevents
/// clients from receiving silent <c>auto</c> behaviour when they ask for <c>required</c>, a
/// specific function, or any value other than <c>auto</c>/unset (issue #121, item 3).
/// </summary>
public sealed class RequestConverterToolChoiceTests
{
    [Fact]
    public void Unset_IsSupported()
    {
        Assert.True(RequestConverter.IsToolChoiceSupported(null, out var v));
        Assert.Equal(string.Empty, v);
    }

    [Fact]
    public void Auto_IsSupported()
    {
        using var doc = JsonDocument.Parse("\"auto\"");
        var el = doc.RootElement;
        Assert.True(RequestConverter.IsToolChoiceSupported(el, out var v));
        Assert.Equal(string.Empty, v);
    }

    [Theory]
    [InlineData("\"none\"", "none")]
    [InlineData("\"required\"", "required")]
    [InlineData("\"bogus\"", "bogus")]
    public void StringNonAuto_IsRejected(string json, string expectedRejected)
    {
        using var doc = JsonDocument.Parse(json);
        var el = doc.RootElement;
        Assert.False(RequestConverter.IsToolChoiceSupported(el, out var v));
        Assert.Equal(expectedRejected, v);
    }

    [Fact]
    public void SpecificFunction_IsRejected()
    {
        using var doc = JsonDocument.Parse(
            """{"type":"function","function":{"name":"get_weather"}}""");
        var el = doc.RootElement;
        Assert.False(RequestConverter.IsToolChoiceSupported(el, out var v));
        Assert.Equal("function:get_weather", v);
    }

    /// <summary>
    /// The gate runs before any other validation, so it must answer "unsupported" for malformed
    /// shapes rather than throwing — <c>TryGetProperty</c> raises <see cref="InvalidOperationException"/>
    /// on a non-object and <c>GetString()</c> on a non-string. A throw here would surface as a 500
    /// instead of the OpenAI-shaped 400.
    /// </summary>
    [Theory]
    [InlineData("""{"type":"function","function":5}""")]
    [InlineData("""{"type":"function","function":"get_weather"}""")]
    [InlineData("""{"type":"function","function":null}""")]
    [InlineData("""{"type":"function","function":[]}""")]
    [InlineData("""{"type":"function","function":{"name":5}}""")]
    [InlineData("""{"type":"function","function":{"name":null}}""")]
    [InlineData("""{"type":"function","function":{}}""")]
    [InlineData("{}")]
    [InlineData("[]")]
    [InlineData("5")]
    [InlineData("true")]
    public void MalformedShape_IsRejectedWithoutThrowing(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var el = doc.RootElement;
        Assert.False(RequestConverter.IsToolChoiceSupported(el, out var v));
        Assert.NotEqual(string.Empty, v);
    }

    /// <summary>Companion to the gate: the converter must degrade to Auto, not throw.</summary>
    [Theory]
    [InlineData("""{"type":"function","function":5}""")]
    [InlineData("""{"type":"function","function":{"name":5}}""")]
    [InlineData("""{"type":"function","function":{}}""")]
    public void ParseToolChoice_MalformedShape_FallsBackToAuto(string json)
    {
        using var doc = JsonDocument.Parse(json);
        Assert.IsType<ToolChoice.Auto>(RequestConverter.ParseToolChoice(doc.RootElement));
    }
}
