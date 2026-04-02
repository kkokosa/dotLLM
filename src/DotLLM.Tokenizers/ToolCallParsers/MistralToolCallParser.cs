namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses tool calls from Mistral Instruct format.
/// Uses <c>[TOOL_CALLS]</c> marker followed by a JSON array of tool call objects.
/// </summary>
public sealed class MistralToolCallParser : IToolCallParser
{
    private const string Marker = "[TOOL_CALLS]";

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
    {
        int markerIndex = generatedText.IndexOf(Marker, StringComparison.Ordinal);
        if (markerIndex < 0)
            return null;

        // Extract balanced JSON after the marker — ignore trailing tokens
        string afterMarker = generatedText[(markerIndex + Marker.Length)..];
        return ToolCallJsonHelper.ExtractAndParse(afterMarker, "call");
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(Marker, StringComparison.Ordinal);
}
