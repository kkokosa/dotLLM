using System.Text;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers.StopConditions;

namespace DotLLM.Engine;

/// <summary>
/// Rolling character-tail buffer for the streaming generation path that prevents
/// stop-string text from leaking through token-boundary-straddling SSE deltas.
/// </summary>
/// <remarks>
/// <para>
/// PR #107 (issue/107-stop-string-suffix) fixed the non-streaming path so a stop-string
/// match is trimmed at the character boundary instead of dropping the whole final token.
/// The streaming path still had two problems:
/// </para>
/// <list type="bullet">
///   <item><description>
///     Each token's decoded delta was emitted to the client as soon as it was generated,
///     so prior tokens that happened to contain a prefix of the eventual stop string were
///     already on the wire before the match could be detected. Example: stop string is
///     <c>"&lt;|im_end|&gt;"</c>; tokens decode as <c>"&lt;|"</c>, <c>"im_"</c>,
///     <c>"end|&gt;"</c>. The leak is <c>"&lt;|"</c> + <c>"im_"</c>.
///   </description></item>
///   <item><description>
///     On a stop-string match the streaming path called <c>generatedIds.RemoveAt(last)</c>
///     before <c>StoreInPrefixCache</c>. The dropped token was already forwarded into the
///     KV cache, so the cached sequence length disagreed with the stored id list.
///   </description></item>
/// </list>
/// <para>
/// This helper holds back the last N characters of the decoded delta stream — where N is
/// the longest registered stop string — until either:
/// </para>
/// <list type="bullet">
///   <item><description>
///     New decoded characters arrive and push older characters out of the holdback window
///     (<see cref="Push"/> releases the now-safe prefix).
///   </description></item>
///   <item><description>
///     Generation ends without a stop-string match — caller invokes <see cref="FlushAll"/>
///     to emit the full remaining tail (the holdback was purely defensive).
///   </description></item>
///   <item><description>
///     A stop-string match is detected — caller invokes <see cref="TrimAndFlush"/> which
///     calls <see cref="StopSuffixTrimmer.MatchedSuffixLength"/> against the buffered tail,
///     drops the matched suffix at the character (UTF-16) boundary, and returns the
///     surviving prefix to emit as one final delta.
///   </description></item>
/// </list>
/// <para>
/// When the registered conditions contain no <see cref="StopStringCondition"/>, holdback
/// is zero and <see cref="Push"/> is a passthrough — preserving today's byte-identical,
/// zero-latency behaviour for the common EOS-only case.
/// </para>
/// </remarks>
internal sealed class StreamingStopBuffer
{
    private readonly int _holdback;
    private readonly StringBuilder _pending = new();

    /// <summary>
    /// Creates a buffer sized for the longest <see cref="StopStringCondition"/> in
    /// <paramref name="conditions"/>. When none is present, holdback is zero and the buffer
    /// behaves as a passthrough.
    /// </summary>
    public StreamingStopBuffer(IReadOnlyList<IStopCondition> conditions)
    {
        int maxStopLen = 0;
        for (int i = 0; i < conditions.Count; i++)
        {
            if (conditions[i] is StopStringCondition ssc && ssc.StopString.Length > maxStopLen)
                maxStopLen = ssc.StopString.Length;
        }
        _holdback = maxStopLen;
    }

    /// <summary>Number of trailing characters held back from emission.</summary>
    public int Holdback => _holdback;

    /// <summary>Number of characters currently buffered (waiting in the holdback window).</summary>
    public int PendingLength => _pending.Length;

    /// <summary>
    /// Appends <paramref name="delta"/> to the pending tail and returns the characters that
    /// are now safe to emit (everything except the last <see cref="Holdback"/> characters).
    /// </summary>
    /// <remarks>
    /// Surrogate-safety: if the emit boundary would split a high+low surrogate pair, the
    /// boundary is moved one character earlier so the emitted string is always valid UTF-16
    /// and the buffer retains the complete surrogate pair for matching.
    /// </remarks>
    public string Push(string delta)
    {
        if (_holdback == 0)
            return delta;
        if (delta.Length == 0)
            return string.Empty;

        _pending.Append(delta);
        if (_pending.Length <= _holdback)
            return string.Empty;

        int safe = _pending.Length - _holdback;
        // Never split a surrogate pair — back off so the trailing high surrogate stays
        // paired with its low surrogate in the pending buffer.
        if (safe > 0 && char.IsHighSurrogate(_pending[safe - 1]))
            safe--;
        if (safe == 0)
            return string.Empty;

        string emit = _pending.ToString(0, safe);
        _pending.Remove(0, safe);
        return emit;
    }

    /// <summary>
    /// Returns the full pending tail and clears the buffer. Use for non-stop-string
    /// terminations (EOS, max-tokens, cache-full, <see cref="StopResult.StopInclude"/>),
    /// where no trimming is appropriate.
    /// </summary>
    public string FlushAll()
    {
        if (_pending.Length == 0)
            return string.Empty;
        string s = _pending.ToString();
        _pending.Clear();
        return s;
    }

    /// <summary>
    /// Trims the matched stop-string suffix from the pending tail at the character boundary
    /// and returns the surviving prefix. Use for <see cref="StopResult.Stop"/> matches that
    /// came from a <see cref="StopStringCondition"/>.
    /// </summary>
    /// <remarks>
    /// The longest registered stop string is at most <see cref="Holdback"/> characters; by
    /// invariant the pending buffer always holds the trailing
    /// <c>min(totalDecodedLength, holdback)</c> characters, so the matched suffix lives
    /// entirely in <see cref="_pending"/> and trimming is local.
    /// </remarks>
    public string TrimAndFlush(IReadOnlyList<IStopCondition> conditions)
    {
        if (_pending.Length == 0)
            return string.Empty;

        // Snapshot the pending span — StringBuilder doesn't expose ReadOnlySpan<char>, but
        // ToString here is bounded by Holdback chars (small constant per request).
        string pending = _pending.ToString();
        int matchedLen = StopSuffixTrimmer.MatchedSuffixLength(pending.AsSpan(), conditions);
        if (matchedLen == 0)
        {
            // Caller expected a stop-string match but none was a suffix of the pending tail.
            // This shouldn't happen given the invariants; fall back to flushing the buffer
            // unchanged rather than silently dropping characters.
            _pending.Clear();
            return pending;
        }

        int end = pending.Length - matchedLen;
        // Surrogate-safety mirrors StopSuffixTrimmer.TrimMatchedSuffix.
        if (end > 0 && char.IsHighSurrogate(pending[end - 1]) && end < pending.Length && char.IsLowSurrogate(pending[end]))
            end--;

        _pending.Clear();
        return end <= 0 ? string.Empty : pending.Substring(0, end);
    }
}
