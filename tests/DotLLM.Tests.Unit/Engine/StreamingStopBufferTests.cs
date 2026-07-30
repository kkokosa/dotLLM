using System.Collections.Generic;
using DotLLM.Core.Sampling;
using DotLLM.Engine;
using DotLLM.Engine.Samplers.StopConditions;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Regression tests for <see cref="StreamingStopBuffer"/>. The bug it replaces:
/// streaming SSE emitted each token's decoded delta the moment it was generated,
/// so prior tokens that contained a prefix of the eventual stop-string were already
/// on the wire by the time the match was detected. Example: stop string is
/// <c>"&lt;|im_end|&gt;"</c> emitted as three tokens <c>"&lt;|"</c>,
/// <c>"im_"</c>, <c>"end|&gt;"</c> — the leak was <c>"&lt;|"</c> + <c>"im_"</c>.
/// See upstream issue #121 item 8.
/// </summary>
public class StreamingStopBufferTests
{
    private static List<IStopCondition> StopStrings(params string[] strings)
    {
        var list = new List<IStopCondition>();
        foreach (var s in strings)
            list.Add(new StopStringCondition(s));
        return list;
    }

    /// <summary>
    /// Discrimination: with a 10-char stop string registered, pushing a 12-char delta must
    /// emit at most 2 chars now (the rest is held back so the trailing stop-string fragment
    /// stays trim-able). Without holdback this test fails — the buffer would echo all 12.
    /// </summary>
    [Fact]
    public void Push_WithStopStrings_HoldsBackLastNChars()
    {
        var buffer = new StreamingStopBuffer(StopStrings("<|im_end|>")); // 10 chars
        Assert.Equal(10, buffer.Holdback);

        string emitted = buffer.Push("Hello, world!"); // 13 chars

        Assert.Equal("Hel", emitted); // 13 - 10 = 3
        Assert.Equal(10, buffer.PendingLength);
    }

    /// <summary>
    /// Discrimination of the headline bug: stop string straddles three pushed deltas
    /// (<c>"&lt;|"</c>, <c>"im_"</c>, <c>"end|&gt;"</c>). Without holdback the
    /// concatenation of <c>Push</c> results would contain <c>"&lt;|im_"</c> — the
    /// fix proves no fragment of the stop string ever leaves.
    /// </summary>
    [Fact]
    public void Push_StopStringStraddlesThreeTokens_NeverLeaksAnyFragment()
    {
        var conditions = StopStrings("<|im_end|>");
        var buffer = new StreamingStopBuffer(conditions);

        // Simulates the per-token deltas the detokenizer would produce.
        string e1 = buffer.Push("Hello, world");
        string e2 = buffer.Push("<|");
        string e3 = buffer.Push("im_");
        string e4 = buffer.Push("end|>");
        string emittedSoFar = e1 + e2 + e3 + e4;

        // After detection, caller trims and flushes.
        string final = buffer.TrimAndFlush(conditions);

        string total = emittedSoFar + final;
        Assert.Equal("Hello, world", total);
        // Critically: at no point did any token's emitted text contain "<|", "im_" etc.
        Assert.DoesNotContain("<|", emittedSoFar);
        Assert.DoesNotContain("im_", emittedSoFar);
        Assert.DoesNotContain("end", emittedSoFar);
    }

    /// <summary>
    /// Discrimination of the "tail-not-lost-on-natural-end" trap (advisor): on a non-stop-string
    /// termination (EOS / max-tokens / cache-full) the holdback must be fully flushed —
    /// otherwise the last <c>Holdback</c> chars of every normal completion are silently dropped.
    /// </summary>
    [Fact]
    public void FlushAll_NonStopTermination_EmitsHeldBackTail()
    {
        var buffer = new StreamingStopBuffer(StopStrings("<|im_end|>"));

        string e1 = buffer.Push("Hello, world!"); // emits "Hel", holds back "lo, world!"
        string e2 = buffer.FlushAll();

        Assert.Equal("Hello, world!", e1 + e2);
    }

    /// <summary>
    /// Trim-and-flush respects the longest matching stop string when multiple are registered —
    /// nested patterns like <c>"&lt;|end|&gt;"</c> vs <c>"&lt;|im_end|&gt;"</c> must trim the
    /// longer one or the shorter pattern's lead chars would remain in the output.
    /// </summary>
    [Fact]
    public void TrimAndFlush_MultipleStopStrings_TrimsLongest()
    {
        var conditions = StopStrings("<|end|>", "<|im_end|>", "end|>");
        var buffer = new StreamingStopBuffer(conditions);

        // Stream the prefix and matched stop string in separate pushes so neither leaks.
        string e1 = buffer.Push("Done");
        string e2 = buffer.Push("<|im_end|>");
        string survived = buffer.TrimAndFlush(conditions);

        // Total emitted across all calls must reconstruct exactly "Done" — and at no
        // point should any fragment of the longest stop string have leaked through.
        string total = e1 + e2 + survived;
        Assert.Equal("Done", total);
        string streamed = e1 + e2;
        Assert.DoesNotContain("<|", streamed);
        Assert.DoesNotContain("end", streamed);
    }

    /// <summary>
    /// Holdback boundary must not split a UTF-16 surrogate pair — otherwise the consumer
    /// receives a dangling high surrogate followed by an orphan low surrogate in the next
    /// chunk. The buffer must back the safe-emit point off one char.
    /// </summary>
    [Fact]
    public void Push_SurrogateAtHoldbackBoundary_NotSplit()
    {
        // Holdback = 5 chars. Push "abc😀ef" (6 chars: a,b,c,HIGH,LOW,e,f → actually 7 chars).
        // We craft input so the boundary would land between the surrogate halves.
        const string emoji = "😀"; // 2 UTF-16 chars (high+low surrogate)
        var buffer = new StreamingStopBuffer(StopStrings("XXXXX")); // 5 char holdback

        // Push "ab" + emoji + "cdef" → 8 chars total → emit first 3 chars normally,
        // but if char[2] is the high surrogate of the emoji we must NOT split → emit 2 chars.
        string emit = buffer.Push("ab" + emoji + "cdef");

        // Verify no high-surrogate trails the emitted string.
        if (emit.Length > 0)
            Assert.False(char.IsHighSurrogate(emit[emit.Length - 1]),
                $"Emit '{emit}' ends with dangling high surrogate.");
        // Round-trip: emit + flush == original
        Assert.Equal("ab" + emoji + "cdef", emit + buffer.FlushAll());
    }

    /// <summary>
    /// Surrogate safety on the trim boundary mirrors <see cref="StopSuffixTrimmer"/>:
    /// if the matched suffix starts with a low surrogate, the high surrogate of the
    /// preceding pair must also be removed. Otherwise the kept tail is malformed UTF-16.
    /// </summary>
    [Fact]
    public void TrimAndFlush_TrimWouldSplitSurrogate_RemovesBothHalves()
    {
        // Same trick as StopSuffixTrimmerTests: stop string starts with a low surrogate.
        var conditions = StopStrings("\uDE00STOP"); // low + STOP, 5 chars
        var buffer = new StreamingStopBuffer(conditions);

        // Stream prefix then the stop region in separate pushes — total reconstruction
        // proves no chars were dropped and the kept tail is valid UTF-16.
        string e1 = buffer.Push("abc");
        string e2 = buffer.Push("😀STOP");
        string survived = buffer.TrimAndFlush(conditions);

        string total = e1 + e2 + survived;
        Assert.Equal("abc", total);
        if (total.Length > 0)
            Assert.False(char.IsHighSurrogate(total[total.Length - 1]),
                "Trimmed output ends with a dangling high surrogate.");
    }

    /// <summary>
    /// No <see cref="StopStringCondition"/> registered → holdback is zero, push is
    /// passthrough. Preserves byte-identical, zero-latency behaviour for the common
    /// EOS-only case (advisor: don't regress this).
    /// </summary>
    [Fact]
    public void NoStopStrings_ZeroHoldback_PushIsPassthrough()
    {
        var conditions = new List<IStopCondition>
        {
            new EosStopCondition(eosTokenId: 2),
            new MaxTokensStopCondition(maxTokens: 100),
        };
        var buffer = new StreamingStopBuffer(conditions);

        Assert.Equal(0, buffer.Holdback);
        Assert.Equal("Hello", buffer.Push("Hello"));
        Assert.Equal(0, buffer.PendingLength);
        Assert.Equal(string.Empty, buffer.FlushAll());
    }

    /// <summary>
    /// Stop-string not at the tail (e.g. it appears mid-stream then text continues
    /// past it) — the buffer keeps draining as new chars arrive. This guards against
    /// false-stop emission and proves the stream-no-truncation property when the
    /// stop never matches.
    /// </summary>
    [Fact]
    public void Push_StopStringNotAtTail_TextContinues_NoTruncation()
    {
        var buffer = new StreamingStopBuffer(StopStrings("END"));

        string e1 = buffer.Push("Begin");
        string e2 = buffer.Push("middle");
        string e3 = buffer.Push("finish");
        string final = buffer.FlushAll();

        Assert.Equal("Beginmiddlefinish", e1 + e2 + e3 + final);
    }

    /// <summary>
    /// Total pending length stays bounded by <see cref="StreamingStopBuffer.Holdback"/>
    /// across many pushes — the buffer drains as new chars arrive, never accumulating
    /// unbounded latency.
    /// </summary>
    [Fact]
    public void Push_ManyChars_PendingStaysBoundedByHoldback()
    {
        var buffer = new StreamingStopBuffer(StopStrings("STOP")); // 4 char holdback
        for (int i = 0; i < 50; i++)
            buffer.Push("xyz");

        Assert.True(buffer.PendingLength <= buffer.Holdback + 1,
            $"Pending {buffer.PendingLength} exceeded holdback {buffer.Holdback}.");
    }
}
