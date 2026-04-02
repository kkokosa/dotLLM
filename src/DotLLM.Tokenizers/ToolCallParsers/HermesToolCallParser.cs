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
                // No closing tag — try to parse what's there (partial output)
                string partialJson = generatedText[jsonStart..].Trim();
                var partialCalls = ToolCallJsonHelper.ParseToolCallJson(partialJson, "call");
                if (partialCalls is { Length: > 0 })
                {
                    foreach (var tc in partialCalls)
                        calls.Add(tc with { Id = $"call_{callIndex++}" });
                }
                break;
            }

            string json = generatedText[jsonStart..closeIndex].Trim();
            var parsed = ToolCallJsonHelper.ParseToolCallJson(json, "call");
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
}
