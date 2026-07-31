using DotLLM.Tokenizers.ToolCallParsers;

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

    /// <summary>
    /// Mints a fresh single-use <see cref="IIncrementalToolCallParser"/> for one
    /// streaming inference request. Default implementation returns a generic
    /// fallback that buffers the whole stream and emits a single tool-call
    /// fragment at flush via <see cref="TryParse"/>; format-specific parsers
    /// override to fragment text-as-arriving and route safe prose to
    /// <c>delta.content</c> mid-stream.
    /// </summary>
    /// <returns>A new stateful incremental parser bound to this <see cref="IToolCallParser"/>'s format knowledge.</returns>
    IIncrementalToolCallParser CreateIncremental() => new FallbackIncrementalToolCallParser(this);
}
