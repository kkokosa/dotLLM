namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses tool calls from Mistral Instruct format.
/// Supports two variants:
/// <list type="bullet">
/// <item>v3: <c>[TOOL_CALLS]funcname[ARGS]{json}</c></item>
/// <item>Legacy: <c>[TOOL_CALLS][{json array}]</c></item>
/// </list>
/// </summary>
public sealed class MistralToolCallParser : IToolCallParser
{
    private const string Marker = "[TOOL_CALLS]";
    private const string ArgsMarker = "[ARGS]";

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
    {
        int markerIndex = generatedText.IndexOf(Marker, StringComparison.Ordinal);
        if (markerIndex < 0)
            return null;

        string afterMarker = generatedText[(markerIndex + Marker.Length)..].Trim();

        // v3 format: [TOOL_CALLS]funcname[ARGS]{json}
        int argsIndex = afterMarker.IndexOf(ArgsMarker, StringComparison.Ordinal);
        if (argsIndex >= 0)
        {
            string funcName = afterMarker[..argsIndex].Trim();
            string argsJson = afterMarker[(argsIndex + ArgsMarker.Length)..].Trim();

            // Extract balanced JSON for arguments
            string balanced = argsJson.Length > 0 && argsJson[0] == '{'
                ? ExtractFirstBalancedBraces(argsJson) ?? argsJson
                : argsJson;

            return [new ToolCall("call_0", funcName, balanced)];
        }

        // Legacy format: [TOOL_CALLS][{json array}] or {json object}
        return ToolCallJsonHelper.ExtractAndParse(afterMarker, "call");
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(Marker, StringComparison.Ordinal);

    private static string? ExtractFirstBalancedBraces(string text)
    {
        int depth = 0;
        bool inString = false;
        bool escaped = false;

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (escaped) { escaped = false; continue; }
            if (c == '\\' && inString) { escaped = true; continue; }
            if (c == '"') { inString = !inString; continue; }
            if (inString) continue;
            if (c == '{') depth++;
            else if (c == '}' && --depth == 0)
                return text[..(i + 1)];
        }
        return null;
    }
}
