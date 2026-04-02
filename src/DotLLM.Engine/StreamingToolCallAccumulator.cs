using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Accumulates streaming token text and detects tool call boundaries.
/// Used by callers of <see cref="TextGenerator.GenerateStreamingTokensAsync"/> to handle
/// tool calls during streaming without buffering all output.
/// </summary>
public sealed class StreamingToolCallAccumulator
{
    private readonly IToolCallParser _parser;
    private readonly StringBuilder _buffer = new();
    private bool _toolCallDetected;

    /// <summary>Whether a tool call start marker has been detected in the accumulated text.</summary>
    public bool IsToolCallInProgress => _toolCallDetected;

    /// <summary>Creates a new accumulator with the given parser.</summary>
    /// <param name="parser">The model-specific tool call parser.</param>
    public StreamingToolCallAccumulator(IToolCallParser parser) => _parser = parser;

    /// <summary>
    /// Appends new text and checks for tool call markers.
    /// </summary>
    /// <param name="text">Incremental text from streaming generation.</param>
    /// <returns>True if this text should be suppressed from user output (it's part of a tool call).</returns>
    public bool Append(string text)
    {
        _buffer.Append(text);
        if (!_toolCallDetected)
            _toolCallDetected = _parser.IsToolCallStart(_buffer.ToString());
        return _toolCallDetected;
    }

    /// <summary>
    /// Attempts to parse completed tool calls from the accumulated buffer.
    /// </summary>
    /// <returns>Parsed tool calls, or null if parsing fails or is incomplete.</returns>
    public ToolCall[]? TryParseCompleted() => _parser.TryParse(_buffer.ToString());

    /// <summary>Gets the full accumulated text.</summary>
    public string GetAccumulatedText() => _buffer.ToString();

    /// <summary>Resets the accumulator for a new generation turn.</summary>
    public void Reset()
    {
        _buffer.Clear();
        _toolCallDetected = false;
    }
}
