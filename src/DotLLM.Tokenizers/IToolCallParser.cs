namespace DotLLM.Tokenizers;

/// <summary>
/// Parses tool call invocations from model-generated text.
/// </summary>
public interface IToolCallParser
{
    /// <summary>
    /// Attempts to parse tool calls from the generated text.
    /// </summary>
    /// <param name="generatedText">Model output text.</param>
    /// <returns>Parsed tool calls, or null if no tool calls were found.</returns>
    ToolCall[]? TryParse(string generatedText);

    /// <summary>
    /// Checks whether the text begins with a tool call marker.
    /// Used during streaming to detect partial tool calls early.
    /// </summary>
    /// <param name="text">Text to check.</param>
    /// <returns>True if the text appears to start a tool call.</returns>
    bool IsToolCallStart(string text);
}
