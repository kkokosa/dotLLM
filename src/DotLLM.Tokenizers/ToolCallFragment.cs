namespace DotLLM.Tokenizers;

/// <summary>
/// Incremental fragment of a tool call emitted during streaming parse.
/// Mirrors the shape of OpenAI's <c>delta.tool_calls[]</c> SSE element so the server
/// can map fragments 1:1 onto outbound chunks.
/// </summary>
/// <remarks>
/// <para>
/// A tool call is communicated as a sequence of fragments sharing the same
/// <see cref="Index"/>. The first fragment for a call carries <see cref="Id"/>
/// and <see cref="Name"/> (and typically the first slice of arguments). Subsequent
/// fragments carry only <see cref="ArgumentsDelta"/>. The last fragment for the
/// call sets <see cref="IsLast"/> so consumers can know the call is closed without
/// having to peek ahead.
/// </para>
/// <para>
/// All string fields are <c>null</c> when absent so a fragment with only an
/// arguments delta has <see cref="Id"/> = <c>null</c>, <see cref="Name"/> = <c>null</c>,
/// <see cref="ArgumentsDelta"/> = the slice. This matches OpenAI's SSE contract
/// where <c>function.name</c> arrives once and <c>function.arguments</c> accumulates.
/// </para>
/// </remarks>
/// <param name="Index">Zero-based parallel-call index. Increments for each new call within a stream.</param>
/// <param name="Id">Server-generated call id. Present on the first fragment of a call only.</param>
/// <param name="Name">Function name. Present on the first fragment of a call only.</param>
/// <param name="ArgumentsDelta">Incremental slice of the arguments JSON string. May be <c>null</c> on the very first opening fragment if no argument text has been seen yet.</param>
/// <param name="IsLast">True when this fragment closes the call (i.e. the close sentinel was matched, or the buffer flushed at end-of-stream).</param>
public readonly record struct ToolCallFragment(
    int Index,
    string? Id,
    string? Name,
    string? ArgumentsDelta,
    bool IsLast);
