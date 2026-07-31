namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses tool calls from Llama 3.1+ output format.
/// Llama uses <c>&lt;|python_tag|&gt;</c> followed by JSON with <c>name</c> and <c>parameters</c> keys.
/// Supports both single calls and parallel calls (JSON array).
/// </summary>
public sealed class LlamaToolCallParser : IToolCallParser
{
    private const string Marker = "<|python_tag|>";

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
    {
        // 1. Explicit <|python_tag|> marker — definitive tool call signal
        int markerIndex = generatedText.IndexOf(Marker, StringComparison.Ordinal);
        if (markerIndex >= 0)
        {
            string afterMarker = generatedText[(markerIndex + Marker.Length)..];
            return ToolCallJsonHelper.ExtractAndParse(afterMarker, "call");
        }

        // 2. No marker — Llama 3.2 lightweight models may omit <|python_tag|>.
        //    If the ENTIRE response is a JSON tool call (starts with { or [),
        //    treat it as a tool call. If there's prose before the JSON,
        //    the model is quoting a schema in its text response — not a tool call.
        string trimmed = generatedText.Trim();
        if (trimmed.Length > 0 && trimmed[0] is '{' or '[')
            return ToolCallJsonHelper.ParseToolCallJson(trimmed, "call");

        return null;
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(Marker, StringComparison.Ordinal);

    /// <inheritdoc/>
    /// <remarks>
    /// Llama 3.1+ tool calls open with <c>&lt;|python_tag|&gt;</c> and have no
    /// dedicated close marker — the call ends at <c>&lt;|eot_id|&gt;</c> or
    /// <c>&lt;|eom_id|&gt;</c>, both already in the server's stop-sequence list.
    /// We therefore use the fallback (buffer-and-parse-at-flush) implementation
    /// for now; this is sufficient to keep raw payloads out of <c>delta.content</c>.
    /// True per-fragment streaming of the Llama JSON arguments is a follow-up.
    /// </remarks>
    public IIncrementalToolCallParser CreateIncremental()
        => new FallbackIncrementalToolCallParser(this);
}
