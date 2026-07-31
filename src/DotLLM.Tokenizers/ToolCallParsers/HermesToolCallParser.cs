namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses tool calls from Hermes/ChatML tool-calling format.
/// Uses <c>&lt;tool_call&gt;</c> and <c>&lt;/tool_call&gt;</c> tags wrapping JSON.
/// Supports multiple tool call blocks (parallel calls).
/// </summary>
public sealed class HermesToolCallParser : IToolCallParser
{
    private const string OpenTag = "<tool_call>";
    private const string CloseTag = "</tool_call>";

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
    {
        var calls = new List<ToolCall>();
        int searchStart = 0;
        int callIndex = 0;

        while (searchStart < generatedText.Length)
        {
            int openIndex = generatedText.IndexOf(OpenTag, searchStart, StringComparison.Ordinal);
            if (openIndex < 0)
                break;

            int jsonStart = openIndex + OpenTag.Length;
            int closeIndex = generatedText.IndexOf(CloseTag, jsonStart, StringComparison.Ordinal);
            if (closeIndex < 0)
            {
                // No closing tag (consumed as stop sequence, or partial output).
                // Use ExtractAndParse to find balanced JSON despite extra braces.
                string partialText = generatedText[jsonStart..].Trim();
                var partialCalls = ToolCallJsonHelper.ExtractAndParse(partialText, "call");
                if (partialCalls is { Length: > 0 })
                {
                    foreach (var tc in partialCalls)
                        calls.Add(tc with { Id = $"call_{callIndex++}" });
                }
                break;
            }

            string json = generatedText[jsonStart..closeIndex].Trim();
            var parsed = ToolCallJsonHelper.ExtractAndParse(json, "call");
            if (parsed is { Length: > 0 })
            {
                foreach (var tc in parsed)
                    calls.Add(tc with { Id = $"call_{callIndex++}" });
            }

            searchStart = closeIndex + CloseTag.Length;
        }

        return calls.Count > 0 ? calls.ToArray() : null;
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(OpenTag, StringComparison.Ordinal);

    /// <inheritdoc/>
    /// <remarks>
    /// Uses sentinel-based suppression: text outside <c>&lt;tool_call&gt;...&lt;/tool_call&gt;</c>
    /// streams as <c>delta.content</c> (with the open-sentinel prefix held back
    /// across chunk boundaries so partial markup never leaks); text inside is
    /// buffered until close and emitted as a single closing fragment per call,
    /// carrying a parseable <c>function.arguments</c> JSON string. Per-character
    /// argument streaming is intentionally deferred — see the class remarks on
    /// <see cref="SentinelIncrementalToolCallParser"/> for the rationale and the
    /// follow-up plan.
    /// </remarks>
    public IIncrementalToolCallParser CreateIncremental()
        => new SentinelIncrementalToolCallParser(this, OpenTag, CloseTag);
}
