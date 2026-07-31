namespace DotLLM.Tokenizers;

/// <summary>
/// Stateful, single-use parser that splits a streaming model-output character
/// stream into a regular-text channel and a tool-call channel, emitting structured
/// <see cref="ToolCallFragment"/> values as soon as enough text has arrived to
/// identify a tool-call boundary.
/// </summary>
/// <remarks>
/// <para>
/// One instance per inference request. Implementations are <b>not</b> thread-safe
/// and carry state between <see cref="AppendChunk"/> calls (rolling buffer, current
/// call index, parse state machine).
/// </para>
/// <para>
/// <b>Contract for the server adapter.</b> The server feeds each generated token's
/// text into <see cref="AppendChunk"/>, then:
/// </para>
/// <list type="bullet">
/// <item>
/// Emits everything in <see cref="ToolCallParseResult.SafeText"/> as
/// <c>delta.content</c> (it has been confirmed not to belong to any tool-call region
/// and not to be a partial prefix of a tool-call sentinel).
/// </item>
/// <item>
/// Emits every <see cref="ToolCallParseResult.Fragments"/> entry as one
/// <c>delta.tool_calls[]</c> SSE chunk.
/// </item>
/// </list>
/// <para>
/// On end-of-stream the server calls <see cref="Flush"/> which returns any
/// remaining safe text plus any pending tool-call closure (so an unterminated
/// <c>&lt;tool_call&gt;...</c> block still yields a final fragment with
/// <see cref="ToolCallFragment.IsLast"/> = <c>true</c>).
/// </para>
/// </remarks>
public interface IIncrementalToolCallParser
{
    /// <summary>
    /// Whether any tool-call fragment has been emitted so far. The server uses
    /// this to decide whether to set <c>finish_reason</c> to <c>tool_calls</c>
    /// and to suppress its post-stream full re-parse-and-emit.
    /// </summary>
    bool HasEmittedAnyFragment { get; }

    /// <summary>
    /// Feeds the next chunk of generated text into the parser.
    /// </summary>
    /// <param name="chunk">The raw text of a single generated token (or any contiguous slice).</param>
    /// <returns>
    /// A result whose <see cref="ToolCallParseResult.SafeText"/> can be emitted as
    /// <c>delta.content</c> immediately and whose
    /// <see cref="ToolCallParseResult.Fragments"/> can be emitted as
    /// <c>delta.tool_calls[]</c> chunks immediately.
    /// </returns>
    ToolCallParseResult AppendChunk(string chunk);

    /// <summary>
    /// Drains any internally buffered text and pending tool-call closure at
    /// end-of-stream.
    /// </summary>
    /// <returns>
    /// Final safe text (any sentinel-prefix tail that turned out not to be a tool
    /// call) and a closing fragment for any in-flight tool call.
    /// </returns>
    ToolCallParseResult Flush();
}

/// <summary>
/// Result of a single <see cref="IIncrementalToolCallParser.AppendChunk"/> or
/// <see cref="IIncrementalToolCallParser.Flush"/> call.
/// </summary>
/// <param name="SafeText">Text that may be emitted as <c>delta.content</c> immediately.</param>
/// <param name="Fragments">Tool-call fragments ready for <c>delta.tool_calls[]</c> emission.</param>
public readonly record struct ToolCallParseResult(
    string SafeText,
    IReadOnlyList<ToolCallFragment> Fragments)
{
    /// <summary>An empty result (no safe text, no fragments).</summary>
    public static ToolCallParseResult Empty { get; } = new(string.Empty, Array.Empty<ToolCallFragment>());
}
