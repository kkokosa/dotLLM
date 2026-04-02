using System.Text;
using System.Text.Json;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Constraints;

/// <summary>
/// Builds JSON Schema strings from <see cref="ToolDefinition"/> arrays for constrained decoding.
/// The generated schemas are consumed by <see cref="JsonSchemaConstraint"/> to guarantee
/// model output matches the tool call format.
/// </summary>
public static class ToolCallSchemaBuilder
{
    /// <summary>
    /// Builds a JSON Schema for a single required function call.
    /// Output must be: <c>{"name": "func_name", "arguments_key": {matches_param_schema}}</c>.
    /// </summary>
    /// <param name="tool">The tool definition.</param>
    /// <param name="argumentsKey">Key name for arguments ("arguments" or "parameters").</param>
    /// <returns>JSON Schema as a string.</returns>
    public static string BuildForFunction(ToolDefinition tool, string argumentsKey = "arguments")
    {
        var sb = new StringBuilder(512);
        sb.Append('{');
        sb.Append("\"type\":\"object\",");
        sb.Append("\"properties\":{");
        sb.Append("\"name\":{\"const\":");
        sb.Append(JsonSerializer.Serialize(tool.Name));
        sb.Append("},");
        sb.Append('"');
        sb.Append(argumentsKey);
        sb.Append("\":");
        sb.Append(NormalizeParametersSchema(tool.ParametersSchema));
        sb.Append("},");
        sb.Append("\"required\":[\"name\",");
        sb.Append(JsonSerializer.Serialize(argumentsKey));
        sb.Append("],");
        sb.Append("\"additionalProperties\":false");
        sb.Append('}');
        return sb.ToString();
    }

    /// <summary>
    /// Builds a JSON Schema for required tool calling with any of the provided tools.
    /// Uses <c>anyOf</c> for multiple tools.
    /// </summary>
    /// <param name="tools">Available tool definitions.</param>
    /// <param name="argumentsKey">Key name for arguments ("arguments" or "parameters").</param>
    /// <returns>JSON Schema as a string.</returns>
    public static string BuildForRequired(ToolDefinition[] tools, string argumentsKey = "arguments")
    {
        if (tools.Length == 1)
            return BuildForFunction(tools[0], argumentsKey);

        var sb = new StringBuilder(1024);
        sb.Append("{\"anyOf\":[");
        for (int i = 0; i < tools.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(BuildForFunction(tools[i], argumentsKey));
        }
        sb.Append("]}");
        return sb.ToString();
    }

    /// <summary>
    /// Builds a JSON Schema for parallel tool calls (array of tool call objects).
    /// </summary>
    /// <param name="tools">Available tool definitions.</param>
    /// <param name="argumentsKey">Key name for arguments ("arguments" or "parameters").</param>
    /// <returns>JSON Schema as a string.</returns>
    public static string BuildForParallelCalls(ToolDefinition[] tools, string argumentsKey = "arguments")
    {
        var itemSchema = tools.Length == 1
            ? BuildForFunction(tools[0], argumentsKey)
            : BuildForRequired(tools, argumentsKey);

        return $"{{\"type\":\"array\",\"items\":{itemSchema}}}";
    }

    /// <summary>
    /// Ensures the parameters schema is valid JSON.
    /// Falls back to permissive object schema if empty or invalid.
    /// </summary>
    private static string NormalizeParametersSchema(string? schema)
    {
        if (string.IsNullOrWhiteSpace(schema))
            return "{\"type\":\"object\"}";

        // Validate it's parseable JSON
        try
        {
            using var doc = JsonDocument.Parse(schema);
            return schema;
        }
        catch (JsonException)
        {
            return "{\"type\":\"object\"}";
        }
    }
}
