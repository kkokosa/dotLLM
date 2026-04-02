using DotLLM.Core.Configuration;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Selects the appropriate <see cref="IToolCallParser"/> based on model architecture
/// and chat template content.
/// </summary>
public static class ToolCallParserFactory
{
    /// <summary>
    /// Creates a tool call parser for the given model.
    /// </summary>
    /// <param name="architecture">Model architecture enum.</param>
    /// <param name="chatTemplate">Raw chat template string (for heuristic detection). May be null.</param>
    /// <returns>The best-matching parser for this model.</returns>
    public static IToolCallParser Create(Architecture architecture, string? chatTemplate = null)
    {
        // 1. Template content heuristics (highest priority — template is the source of truth)
        if (!string.IsNullOrEmpty(chatTemplate))
        {
            if (chatTemplate.Contains("<tool_call>", StringComparison.Ordinal))
                return new HermesToolCallParser();

            if (chatTemplate.Contains("python_tag", StringComparison.Ordinal) ||
                chatTemplate.Contains("<|python_tag|>", StringComparison.Ordinal))
                return new LlamaToolCallParser();

            if (chatTemplate.Contains("[TOOL_CALLS]", StringComparison.Ordinal))
                return new MistralToolCallParser();
        }

        // 2. Architecture-based fallback
        return architecture switch
        {
            Architecture.Llama => new LlamaToolCallParser(),
            Architecture.Mistral => new MistralToolCallParser(),
            Architecture.Qwen => new HermesToolCallParser(),
            _ => new GenericToolCallParser()
        };
    }
}
