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
        sb.Append(JsonQuote(tool.Name));
        sb.Append("},");
        sb.Append('"');
        sb.Append(argumentsKey);
        sb.Append("\":");
        sb.Append(NormalizeParametersSchema(tool.ParametersSchema));
        sb.Append("},");
        sb.Append("\"required\":[\"name\",");
        sb.Append(JsonQuote(argumentsKey));
        sb.Append("],");
        sb.Append("\"additionalProperties\":false");
        sb.Append('}');
        return sb.ToString();
    }

    /// <summary>
    /// Builds a JSON Schema for required tool calling with any of the provided tools.
    /// Uses <c>enum</c> for the name field to constrain to valid tool names.
    /// </summary>
    /// <remarks>
    /// For a single tool, delegates to <see cref="BuildForFunction"/> which uses <c>const</c>
    /// and the tool's full parameter schema. For multiple tools, uses a flat object schema
    /// with <c>enum</c> for names and a permissive object type for arguments — this avoids
    /// <c>anyOf</c> whose nested constraints are not enforced by the current SchemaTracker.
    /// </remarks>
    /// <param name="tools">Available tool definitions.</param>
    /// <param name="argumentsKey">Key name for arguments ("arguments" or "parameters").</param>
    /// <returns>JSON Schema as a string.</returns>
    public static string BuildForRequired(ToolDefinition[] tools, string argumentsKey = "arguments")
    {
        if (tools.Length == 1)
            return BuildForFunction(tools[0], argumentsKey);

        // Multi-tool: use enum for name instead of anyOf with per-tool const.
        // SchemaTracker's anyOf is an overapproximation that doesn't enforce
        // nested property constraints (const, required, additionalProperties).
        var sb = new StringBuilder(1024);
        sb.Append("{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\",\"enum\":[");
        for (int i = 0; i < tools.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(JsonQuote(tools[i].Name));
        }
        sb.Append("]},\"");
        sb.Append(argumentsKey);
        sb.Append("\":{\"type\":\"object\"}},\"required\":[\"name\",");
        sb.Append(JsonQuote(argumentsKey));
        sb.Append("],\"additionalProperties\":false}");
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
        var itemSchema = BuildForRequired(tools, argumentsKey);
        return $"{{\"type\":\"array\",\"items\":{itemSchema}}}";
    }

    /// <summary>
    /// JSON-escapes a string and wraps it in double quotes. Trim/AOT-safe replacement
    /// for <c>JsonSerializer.Serialize(string)</c>.
    /// </summary>
    private static string JsonQuote(string s)
    {
        // Tool names and argument keys are typically ASCII identifiers with no special
        // characters, but we handle the full JSON string grammar for correctness.
        var needsEscape = false;
        foreach (var c in s)
        {
            if (c is '"' or '\\' || c < ' ') { needsEscape = true; break; }
        }

        if (!needsEscape)
            return string.Concat("\"", s, "\"");

        var sb = new StringBuilder(s.Length + 2);
        sb.Append('"');
        foreach (var c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < ' ')
                    {
                        // Zero-alloc hex escape — control chars are always 00xx
                        sb.Append("\\u00");
                        sb.Append("0123456789ABCDEF"[(c >> 4) & 0xF]);
                        sb.Append("0123456789ABCDEF"[c & 0xF]);
                    }
                    else sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
        return sb.ToString();
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
