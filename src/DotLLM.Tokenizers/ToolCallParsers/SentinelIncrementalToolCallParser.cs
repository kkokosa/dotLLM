using System.Text;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Streaming character-fed parser that recognises sentinel-wrapped tool-call
/// payloads (e.g. <c>&lt;tool_call&gt;{...}&lt;/tool_call&gt;</c>) and routes the
/// arriving characters into either the safe-text channel
/// (<c>delta.content</c>) or the tool-call channel
/// (<c>delta.tool_calls[].function.arguments</c>).
/// </summary>
/// <remarks>
/// <para>
/// <b>State machine.</b>
/// </para>
/// <list type="bullet">
/// <item><c>OutsideCall</c> — regular text. Any character not part of a possible
/// open-sentinel prefix is emitted as safe text. The trailing characters that
/// could be a prefix of the open sentinel are <i>held back</i> across chunks so a
/// token like <c>&lt;tool_</c> never leaks into <c>delta.content</c>.</item>
/// <item><c>InsideCall</c> — inside a recognised tool-call block. Characters are
/// accumulated into a per-call buffer. <c>delta.content</c> is suppressed for the
/// whole block. On close-sentinel match (or end-of-stream), the buffered payload
/// is parsed via the host <see cref="IToolCallParser.TryParse"/> and emitted as a
/// single closing <see cref="ToolCallFragment"/> per call extracted, carrying
/// id + name + the full arguments JSON.</item>
/// </list>
/// <para>
/// <b>Why suppress and not true fragment streaming.</b> Real OpenAI clients
/// concatenate <c>function.arguments</c> deltas and call <c>json.loads</c> on
/// the result. Streaming a JSON envelope's bytes between the inner arguments
/// object and the envelope close produces an invalid JSON string when
/// concatenated. Buffering the whole payload and parsing once at close keeps the
/// emitted <c>arguments</c> string a valid JSON object — which is the contract
/// item #4 of upstream issue #121 exists to satisfy. True per-character argument
/// streaming requires brace-balanced extraction of just the arguments value;
/// see the doc on <see cref="IToolCallParser.CreateIncremental"/> for the
/// follow-up plan.
/// </para>
/// </remarks>
internal sealed class SentinelIncrementalToolCallParser : IIncrementalToolCallParser
{
    private readonly string _openSentinel;
    private readonly string _closeSentinel;
    private readonly IToolCallParser _host;

    /// <summary>Rolling buffer for the outside-call channel — characters not yet committed as safe text.</summary>
    private readonly StringBuilder _outsideBuffer = new();

    /// <summary>Per-call accumulator. Holds the raw payload between sentinels.</summary>
    private readonly StringBuilder _callBuffer = new();

    private State _state = State.OutsideCall;
    private int _callIndex;

    /// <summary>
    /// Index into <see cref="_callBuffer"/> from which the close-sentinel search resumes.
    /// The outside buffer is self-limiting — <see cref="SafePrefixLength"/> drains everything that
    /// cannot begin a sentinel, so it never exceeds <c>openSentinel.Length - 1</c> and rescanning it
    /// is O(1). The call buffer has no such bound: it accumulates the whole tool-call payload, so
    /// rescanning it from zero on every appended character would be quadratic in payload size. Any
    /// match beginning before this offset would have lain wholly inside an already-scanned region
    /// and been found then, so the search can safely resume here.
    /// </summary>
    private int _callSearchFrom;

    /// <inheritdoc/>
    public bool HasEmittedAnyFragment { get; private set; }

    /// <summary>Creates a new sentinel-based incremental parser.</summary>
    /// <param name="host">The host <see cref="IToolCallParser"/> whose
    /// <see cref="IToolCallParser.TryParse"/> is invoked on close-sentinel match
    /// to extract the structured tool calls from the buffered payload.</param>
    /// <param name="openSentinel">The literal text that marks the start of a tool-call block (e.g. <c>&lt;tool_call&gt;</c>).</param>
    /// <param name="closeSentinel">The literal text that marks the end of a tool-call block (e.g. <c>&lt;/tool_call&gt;</c>).</param>
    public SentinelIncrementalToolCallParser(
        IToolCallParser host,
        string openSentinel,
        string closeSentinel)
    {
        ArgumentNullException.ThrowIfNull(host);
        ArgumentException.ThrowIfNullOrEmpty(openSentinel);
        ArgumentException.ThrowIfNullOrEmpty(closeSentinel);
        _host = host;
        _openSentinel = openSentinel;
        _closeSentinel = closeSentinel;
    }

    /// <inheritdoc/>
    public ToolCallParseResult AppendChunk(string chunk)
    {
        if (string.IsNullOrEmpty(chunk))
            return ToolCallParseResult.Empty;

        var safeText = new StringBuilder();
        List<ToolCallFragment>? fragments = null;

        for (int i = 0; i < chunk.Length; i++)
        {
            char c = chunk[i];

            if (_state == State.OutsideCall)
            {
                _outsideBuffer.Append(c);
                TryTransitionOutside(safeText, ref fragments);
            }
            else // InsideCall
            {
                _callBuffer.Append(c);
                TryTransitionInside(safeText, ref fragments);
            }
        }

        return new ToolCallParseResult(
            safeText.ToString(),
            (IReadOnlyList<ToolCallFragment>?)fragments ?? Array.Empty<ToolCallFragment>());
    }

    /// <inheritdoc/>
    public ToolCallParseResult Flush()
    {
        var safeText = new StringBuilder();
        List<ToolCallFragment>? fragments = null;

        if (_state == State.OutsideCall)
        {
            // Anything still in the outside buffer is not a tool-call prefix — emit it.
            if (_outsideBuffer.Length > 0)
            {
                safeText.Append(_outsideBuffer);
                _outsideBuffer.Clear();
            }
        }
        else
        {
            // Unterminated tool call — close it with whatever is buffered.
            EmitClosingFragments(ref fragments);
            _callBuffer.Clear();
            _callSearchFrom = 0;
            _state = State.OutsideCall;
        }

        return new ToolCallParseResult(
            safeText.ToString(),
            (IReadOnlyList<ToolCallFragment>?)fragments ?? Array.Empty<ToolCallFragment>());
    }

    // ─────────────────────────────────────────────────────────────────────
    // Outside-call transitions: commit safe prefix, hold back possible open-sentinel suffix
    // ─────────────────────────────────────────────────────────────────────

    private void TryTransitionOutside(StringBuilder safeText, ref List<ToolCallFragment>? fragments)
    {
        // If the buffer contains the complete open sentinel, commit text before it,
        // then switch to InsideCall state with any post-sentinel residue already in the call buffer.
        int openIdx = IndexOf(_outsideBuffer, _openSentinel);
        if (openIdx >= 0)
        {
            // Commit everything before the sentinel as safe text.
            if (openIdx > 0)
                safeText.Append(_outsideBuffer, 0, openIdx);

            // Anything after the sentinel goes into the call buffer.
            int residueStart = openIdx + _openSentinel.Length;
            int residueLen = _outsideBuffer.Length - residueStart;
            _callBuffer.Clear();
            _callSearchFrom = 0;
            if (residueLen > 0)
                _callBuffer.Append(_outsideBuffer, residueStart, residueLen);

            _outsideBuffer.Clear();
            _state = State.InsideCall;

            // The residue might already contain the close sentinel; re-run the inside transition.
            TryTransitionInside(safeText, ref fragments);
            return;
        }

        // No full open sentinel yet. Commit the prefix that cannot be the start of one.
        int safePrefixLen = SafePrefixLength(_outsideBuffer, _openSentinel);
        if (safePrefixLen > 0)
        {
            safeText.Append(_outsideBuffer, 0, safePrefixLen);
            _outsideBuffer.Remove(0, safePrefixLen);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Inside-call transitions: detect close, emit fragment, flow back outside
    // ─────────────────────────────────────────────────────────────────────

    private void TryTransitionInside(StringBuilder safeText, ref List<ToolCallFragment>? fragments)
    {
        int closeIdx = IndexOf(_callBuffer, _closeSentinel, _callSearchFrom);
        if (closeIdx < 0)
        {
            // Wait for more bytes; nothing to suppress that isn't already buffered. Only the last
            // (closeSentinel.Length - 1) characters can still take part in a future match, so the
            // next scan resumes there — amortised O(1) per character instead of a full rescan.
            _callSearchFrom = Math.Max(0, _callBuffer.Length - _closeSentinel.Length + 1);
            return;
        }

        // Trim the call buffer at the close sentinel and emit closing fragments.
        int residueStart = closeIdx + _closeSentinel.Length;
        int residueLen = _callBuffer.Length - residueStart;
        string residue = residueLen > 0 ? _callBuffer.ToString(residueStart, residueLen) : string.Empty;
        _callBuffer.Length = closeIdx;

        EmitClosingFragments(ref fragments);
        _callBuffer.Clear();
        _callSearchFrom = 0;
        _state = State.OutsideCall;

        // Push residue back into the outside buffer and re-run the outside
        // transition so a second tool call (or prose) within the same chunk
        // is handled in the same AppendChunk invocation.
        if (residue.Length > 0)
        {
            _outsideBuffer.Append(residue);
            TryTransitionOutside(safeText, ref fragments);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Closing fragments — invoked at close sentinel and at flush
    // ─────────────────────────────────────────────────────────────────────

    private void EmitClosingFragments(ref List<ToolCallFragment>? fragments)
    {
        // Wrap the buffered payload in the host parser's sentinels so its
        // TryParse can apply format-specific extraction (e.g. balanced-brace
        // JSON, name + arguments separation). Multiple calls per block are
        // supported: each parsed ToolCall becomes one fragment with its own Index.
        string wrapped = _openSentinel + _callBuffer.ToString() + _closeSentinel;
        var parsed = _host.TryParse(wrapped);
        if (parsed is { Length: > 0 })
        {
            for (int i = 0; i < parsed.Length; i++)
            {
                var tc = parsed[i];
                fragments ??= new List<ToolCallFragment>();
                fragments.Add(new ToolCallFragment(
                    Index: _callIndex,
                    Id: $"call_{_callIndex}",
                    Name: tc.FunctionName,
                    ArgumentsDelta: tc.Arguments,
                    IsLast: true));
                HasEmittedAnyFragment = true;
                _callIndex++;
            }
        }
        // Nothing parsed: no fragment was emitted, so no index was consumed. Bumping _callIndex
        // here would leave a hole in the tool_calls[] stream — a consumer indexing by `index`
        // would materialise a gap for a call that never existed — and would desync the generated
        // `call_{n}` ids from the fragments actually on the wire. Indices stay contiguous over
        // emitted fragments, which is what the Index contract promises.
    }

    // ─────────────────────────────────────────────────────────────────────
    // Small StringBuilder helpers (no string allocation)
    // ─────────────────────────────────────────────────────────────────────

    private static int IndexOf(StringBuilder sb, string needle, int start = 0)
    {
        if (needle.Length == 0) return start;
        int max = sb.Length - needle.Length;
        for (int i = start; i <= max; i++)
        {
            int j = 0;
            while (j < needle.Length && sb[i + j] == needle[j]) j++;
            if (j == needle.Length) return i;
        }
        return -1;
    }

    /// <summary>
    /// Returns the largest <c>k</c> for which the first <c>k</c> characters of
    /// the buffer cannot be a prefix of <paramref name="sentinel"/>. Equivalently,
    /// the number of characters at the start of the buffer that are safe to
    /// commit because no completion of the buffer can produce a sentinel that
    /// overlaps them.
    /// </summary>
    private static int SafePrefixLength(StringBuilder buffer, string sentinel)
    {
        // The held-back tail is the longest suffix of buffer that is a strict prefix of sentinel.
        int max = Math.Min(sentinel.Length - 1, buffer.Length);
        for (int k = max; k > 0; k--)
        {
            if (BufferEndsWithSentinelPrefix(buffer, sentinel, k))
                return buffer.Length - k;
        }
        return buffer.Length;
    }

    private static bool BufferEndsWithSentinelPrefix(StringBuilder buffer, string sentinel, int k)
    {
        if (k > buffer.Length || k > sentinel.Length) return false;
        int start = buffer.Length - k;
        for (int i = 0; i < k; i++)
        {
            if (buffer[start + i] != sentinel[i]) return false;
        }
        return true;
    }

    private enum State
    {
        OutsideCall,
        InsideCall,
    }
}
