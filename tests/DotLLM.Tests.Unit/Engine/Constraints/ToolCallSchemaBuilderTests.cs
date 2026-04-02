using System.Text.Json;
using DotLLM.Engine.Constraints;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints;

public class ToolCallSchemaBuilderTests
{
    private static readonly ToolDefinition WeatherTool = new(
        "get_weather",
        "Get current weather",
        """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""");

    private static readonly ToolDefinition TimeTool = new(
        "get_time",
        "Get current time",
        """{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}""");

    [Fact]
    public void BuildForFunction_ProducesValidJson()
    {
        string schema = ToolCallSchemaBuilder.BuildForFunction(WeatherTool);

        using var doc = JsonDocument.Parse(schema);
        var root = doc.RootElement;

        Assert.Equal("object", root.GetProperty("type").GetString());
        Assert.True(root.GetProperty("properties").TryGetProperty("name", out _));
        Assert.True(root.GetProperty("properties").TryGetProperty("arguments", out _));
    }

    [Fact]
    public void BuildForFunction_NameConst_MatchesToolName()
    {
        string schema = ToolCallSchemaBuilder.BuildForFunction(WeatherTool);

        using var doc = JsonDocument.Parse(schema);
        var nameConst = doc.RootElement.GetProperty("properties").GetProperty("name").GetProperty("const").GetString();

        Assert.Equal("get_weather", nameConst);
    }

    [Fact]
    public void BuildForFunction_CustomArgumentsKey()
    {
        string schema = ToolCallSchemaBuilder.BuildForFunction(WeatherTool, argumentsKey: "parameters");

        using var doc = JsonDocument.Parse(schema);
        var props = doc.RootElement.GetProperty("properties");

        Assert.True(props.TryGetProperty("parameters", out _));
        Assert.False(props.TryGetProperty("arguments", out _));
    }

    [Fact]
    public void BuildForFunction_IncludesParametersSchema()
    {
        string schema = ToolCallSchemaBuilder.BuildForFunction(WeatherTool);

        using var doc = JsonDocument.Parse(schema);
        var argsSchema = doc.RootElement.GetProperty("properties").GetProperty("arguments");

        Assert.Equal("object", argsSchema.GetProperty("type").GetString());
        Assert.True(argsSchema.GetProperty("properties").TryGetProperty("location", out _));
    }

    [Fact]
    public void BuildForRequired_SingleTool_SameAsBuildForFunction()
    {
        string single = ToolCallSchemaBuilder.BuildForFunction(WeatherTool);
        string required = ToolCallSchemaBuilder.BuildForRequired([WeatherTool]);

        Assert.Equal(single, required);
    }

    [Fact]
    public void BuildForRequired_MultipleTools_ProducesAnyOf()
    {
        string schema = ToolCallSchemaBuilder.BuildForRequired([WeatherTool, TimeTool]);

        using var doc = JsonDocument.Parse(schema);
        var root = doc.RootElement;

        Assert.True(root.TryGetProperty("anyOf", out var anyOf));
        Assert.Equal(2, anyOf.GetArrayLength());
    }

    [Fact]
    public void BuildForRequired_AnyOf_ContainsBothTools()
    {
        string schema = ToolCallSchemaBuilder.BuildForRequired([WeatherTool, TimeTool]);

        using var doc = JsonDocument.Parse(schema);
        var anyOf = doc.RootElement.GetProperty("anyOf");

        var names = new List<string>();
        foreach (var alt in anyOf.EnumerateArray())
        {
            var name = alt.GetProperty("properties").GetProperty("name").GetProperty("const").GetString();
            names.Add(name!);
        }

        Assert.Contains("get_weather", names);
        Assert.Contains("get_time", names);
    }

    [Fact]
    public void BuildForParallelCalls_ProducesArraySchema()
    {
        string schema = ToolCallSchemaBuilder.BuildForParallelCalls([WeatherTool, TimeTool]);

        using var doc = JsonDocument.Parse(schema);
        var root = doc.RootElement;

        Assert.Equal("array", root.GetProperty("type").GetString());
        Assert.True(root.TryGetProperty("items", out _));
    }

    [Fact]
    public void BuildForParallelCalls_ItemsHaveAnyOf()
    {
        string schema = ToolCallSchemaBuilder.BuildForParallelCalls([WeatherTool, TimeTool]);

        using var doc = JsonDocument.Parse(schema);
        var items = doc.RootElement.GetProperty("items");

        Assert.True(items.TryGetProperty("anyOf", out var anyOf));
        Assert.Equal(2, anyOf.GetArrayLength());
    }

    [Fact]
    public void BuildForFunction_EmptyParametersSchema_FallsBackToObject()
    {
        var tool = new ToolDefinition("noop", "No-op", "");
        string schema = ToolCallSchemaBuilder.BuildForFunction(tool);

        using var doc = JsonDocument.Parse(schema);
        var argsSchema = doc.RootElement.GetProperty("properties").GetProperty("arguments");

        Assert.Equal("object", argsSchema.GetProperty("type").GetString());
    }

    [Fact]
    public void BuildForFunction_InvalidParametersSchema_FallsBackToObject()
    {
        var tool = new ToolDefinition("noop", "No-op", "not valid json");
        string schema = ToolCallSchemaBuilder.BuildForFunction(tool);

        using var doc = JsonDocument.Parse(schema);
        var argsSchema = doc.RootElement.GetProperty("properties").GetProperty("arguments");

        Assert.Equal("object", argsSchema.GetProperty("type").GetString());
    }

    [Fact]
    public void BuildForFunction_SchemaIsCompilableBySchemaCompiler()
    {
        string schema = ToolCallSchemaBuilder.BuildForFunction(WeatherTool);

        // This should not throw — validates the schema is accepted by our compiler
        var compiled = DotLLM.Engine.Constraints.Schema.SchemaCompiler.Compile(schema);
        Assert.NotNull(compiled);
    }

    [Fact]
    public void BuildForRequired_MultiTool_SchemaIsCompilableBySchemaCompiler()
    {
        string schema = ToolCallSchemaBuilder.BuildForRequired([WeatherTool, TimeTool]);

        var compiled = DotLLM.Engine.Constraints.Schema.SchemaCompiler.Compile(schema);
        Assert.NotNull(compiled);
    }
}
