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
        int markerIndex = generatedText.IndexOf(Marker, StringComparison.Ordinal);
        if (markerIndex < 0)
        {
            // Some Llama models emit tool calls without the special token
            // Try to detect JSON with "name" + "parameters" pattern
            return ToolCallJsonHelper.ExtractAndParse(generatedText, "call");
        }

        string json = generatedText[(markerIndex + Marker.Length)..].Trim();
        return ToolCallJsonHelper.ParseToolCallJson(json, "call");
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(Marker, StringComparison.Ordinal);
}
