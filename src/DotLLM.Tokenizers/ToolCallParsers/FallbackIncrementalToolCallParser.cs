using System.Text;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Default <see cref="IIncrementalToolCallParser"/> for parsers that don't
/// supply a format-aware streaming implementation.
/// </summary>
/// <remarks>
/// <para>
/// Buffers the entire stream and decides on flush whether the accumulated text
/// is a tool call. If it is, the parsed tool calls are emitted as single
/// fragments (one per call, each marked <see cref="ToolCallFragment.IsLast"/>).
/// If it isn't, the buffered text is returned as safe text on flush.
/// </para>
/// <para>
/// This preserves correctness across all existing parsers without breaking the
/// streaming contract — the worst case is that <c>delta.content</c> arrives only
/// at the end of the stream, identical to today's behaviour minus the leaked
/// markup (tool-call payloads stay inside <c>delta.tool_calls</c> when detected).
/// To get true mid-stream fragmenting, supply a format-specific implementation
/// (see <see cref="SentinelIncrementalToolCallParser"/>).
/// </para>
/// </remarks>
internal sealed class FallbackIncrementalToolCallParser : IIncrementalToolCallParser
{
    private readonly IToolCallParser _host;
    private readonly StringBuilder _buffer = new();

    /// <inheritdoc/>
    public bool HasEmittedAnyFragment { get; private set; }

    public FallbackIncrementalToolCallParser(IToolCallParser host)
    {
        ArgumentNullException.ThrowIfNull(host);
        _host = host;
    }

    /// <inheritdoc/>
    public ToolCallParseResult AppendChunk(string chunk)
    {
        if (string.IsNullOrEmpty(chunk))
            return ToolCallParseResult.Empty;
        _buffer.Append(chunk);
        return ToolCallParseResult.Empty;
    }

    /// <inheritdoc/>
    public ToolCallParseResult Flush()
    {
        if (_buffer.Length == 0)
            return ToolCallParseResult.Empty;

        string text = _buffer.ToString();
        _buffer.Clear();

        var calls = _host.TryParse(text);
        if (calls is { Length: > 0 })
        {
            var fragments = new List<ToolCallFragment>(calls.Length);
            for (int i = 0; i < calls.Length; i++)
            {
                fragments.Add(new ToolCallFragment(
                    Index: i,
                    Id: calls[i].Id,
                    Name: calls[i].FunctionName,
                    ArgumentsDelta: calls[i].Arguments,
                    IsLast: true));
            }
            HasEmittedAnyFragment = true;
            return new ToolCallParseResult(string.Empty, fragments);
        }

        return new ToolCallParseResult(text, Array.Empty<ToolCallFragment>());
    }
}
