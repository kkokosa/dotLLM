namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Fallback tool call parser that detects bare JSON with <c>name</c> and
/// <c>arguments</c>/<c>parameters</c> fields. No special markers required.
/// </summary>
public sealed class GenericToolCallParser : IToolCallParser
{
    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
        => ToolCallJsonHelper.ExtractAndParse(generatedText, "call");

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
    {
        // Heuristic: text contains a JSON-like pattern with "name" key
        int braceIndex = text.IndexOf('{');
        if (braceIndex < 0)
            return false;

        // Check if there's a "name" key after the brace
        int nameIndex = text.IndexOf("\"name\"", braceIndex, StringComparison.Ordinal);
        return nameIndex > braceIndex;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The generic parser has no fixed sentinels, so it can't reliably hold
    /// back partial markup from <c>delta.content</c> mid-stream. We use the
    /// buffer-at-flush fallback — equivalent to the pre-#121 streaming behaviour
    /// but routing detected calls to <c>delta.tool_calls</c> rather than
    /// leaking them as content.
    /// </remarks>
    public IIncrementalToolCallParser CreateIncremental()
        => new FallbackIncrementalToolCallParser(this);
}
