using System.Text;
using System.Text.Json;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ToolCallParsers;

/// <summary>
/// Unit tests for <see cref="IIncrementalToolCallParser"/> implementations.
/// Drives parsers directly with simulated streamed chunks — no live model.
/// Covers the OpenAI SSE contract for <c>delta.tool_calls</c>:
/// id + name + type:"function" on first fragment; arguments-only deltas
/// after; <c>index</c> increments across parallel/sequential calls.
/// </summary>
public class IncrementalToolCallParserTests
{
    /// <summary>Convenience: feed chunks one-by-one and collect everything.</summary>
    private static (string allSafe, List<ToolCallFragment> allFragments) RunStream(
        IIncrementalToolCallParser parser, params string[] chunks)
    {
        var safe = new StringBuilder();
        var fragments = new List<ToolCallFragment>();
        foreach (var chunk in chunks)
        {
            var r = parser.AppendChunk(chunk);
            safe.Append(r.SafeText);
            fragments.AddRange(r.Fragments);
        }
        var flush = parser.Flush();
        safe.Append(flush.SafeText);
        fragments.AddRange(flush.Fragments);
        return (safe.ToString(), fragments);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Hermes / XML — full fragmenting incremental parser
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// A block the host parser cannot parse emits no fragment, so it must not consume an index.
    /// Bumping the counter on parse failure leaves a hole in the tool_calls[] stream — a consumer
    /// indexing by <c>index</c> materialises a slot for a call that never arrived — and desyncs the
    /// generated <c>call_{n}</c> ids from the fragments actually on the wire.
    /// </summary>
    [Fact]
    public void Hermes_UnparseableBlock_DoesNotConsumeAnIndex()
    {
        var parser = new HermesToolCallParser().CreateIncremental();

        var (_, fragments) = RunStream(parser,
            "<tool_call>",
            "not json at all",
            "</tool_call>",
            "<tool_call>",
            """{"name": "lookup", "arguments": {"id": 7}}""",
            "</tool_call>");

        var fragment = Assert.Single(fragments);
        Assert.Equal("lookup", fragment.Name);
        Assert.Equal(0, fragment.Index);
        Assert.Equal("call_0", fragment.Id);
    }

    [Fact]
    public void Hermes_SingleToolCall_NoLeakedContent()
    {
        var parser = new HermesToolCallParser().CreateIncremental();

        var (safe, fragments) = RunStream(parser,
            "<tool_call>",
            """{"name": "get_weather", "arguments": {"location": "Berlin"}}""",
            "</tool_call>");

        Assert.Equal(string.Empty, safe);
        Assert.True(parser.HasEmittedAnyFragment);

        // Sentinel suppress-mode emits exactly one fragment per call — carrying
        // id + name + a parseable arguments JSON string + IsLast=true.
        Assert.Single(fragments);
        Assert.Equal("call_0", fragments[0].Id);
        Assert.Equal("get_weather", fragments[0].Name);
        Assert.Equal(0, fragments[0].Index);
        Assert.True(fragments[0].IsLast);

        // The arguments string MUST be valid JSON — clients call json.loads on it.
        string joinedArgs = string.Concat(
            fragments.Select(f => f.ArgumentsDelta ?? string.Empty));
        using var doc = System.Text.Json.JsonDocument.Parse(joinedArgs);
        Assert.Equal("Berlin", doc.RootElement.GetProperty("location").GetString());
    }

    [Fact]
    public void Hermes_TokenizedSplit_NoMarkupInSafeText()
    {
        // Simulate BPE-style tokenization that splits the sentinel across chunks.
        var parser = new HermesToolCallParser().CreateIncremental();

        var (safe, fragments) = RunStream(parser,
            "<tool",      // partial open sentinel
            "_call>",     // completes open
            """{"name":""",
            """ "search",""",
            """ "arguments": {"q": "abc"}}""",
            "</tool",     // partial close
            "_call>");

        Assert.Equal(string.Empty, safe);
        Assert.True(parser.HasEmittedAnyFragment);

        // The point of the test is that sentinels split across chunk boundaries still yield one
        // complete, well-formed call — so assert the whole fragment, not just its name.
        var fragment = Assert.Single(fragments);
        Assert.Equal("search", fragment.Name);
        Assert.Equal("call_0", fragment.Id);
        Assert.Equal(0, fragment.Index);
        Assert.True(fragment.IsLast);

        // Arguments must survive reassembly as valid JSON — clients json.loads the concatenation.
        Assert.NotNull(fragment.ArgumentsDelta);
        using var args = JsonDocument.Parse(fragment.ArgumentsDelta!);
        Assert.Equal("abc", args.RootElement.GetProperty("q").GetString());
    }

    [Fact]
    public void Hermes_TextBeforeAndAfterCall_SplitCorrectly()
    {
        var parser = new HermesToolCallParser().CreateIncremental();

        var (safe, fragments) = RunStream(parser,
            "Let me check that. ",
            "<tool_call>",
            """{"name": "lookup", "arguments": {"id": 7}}""",
            "</tool_call>",
            " Done.");

        Assert.Equal("Let me check that.  Done.", safe);
        Assert.NotEmpty(fragments);
        Assert.Equal("lookup", fragments[0].Name);
    }

    [Fact]
    public void Hermes_NoToolCall_AllTextIsSafe()
    {
        var parser = new HermesToolCallParser().CreateIncremental();

        var (safe, fragments) = RunStream(parser,
            "Just a normal ",
            "answer with no tools.");

        Assert.Equal("Just a normal answer with no tools.", safe);
        Assert.Empty(fragments);
        Assert.False(parser.HasEmittedAnyFragment);
    }

    [Fact]
    public void Hermes_ParallelCalls_IndexIncrements()
    {
        var parser = new HermesToolCallParser().CreateIncremental();

        var (safe, fragments) = RunStream(parser,
            """<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>""",
            """<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>""");

        Assert.Equal(string.Empty, safe);

        // Suppress-mode: one fragment per call. Index increments across calls
        // (it does not reset).
        Assert.Equal(2, fragments.Count);
        Assert.Equal("a", fragments[0].Name);
        Assert.Equal(0, fragments[0].Index);
        Assert.Equal("call_0", fragments[0].Id);
        Assert.Equal("b", fragments[1].Name);
        Assert.Equal(1, fragments[1].Index);
        Assert.Equal("call_1", fragments[1].Id);
    }

    [Fact]
    public void Hermes_TwoCallsInSameChunk_BothEmittedWithDistinctIndexes()
    {
        var parser = new HermesToolCallParser().CreateIncremental();

        // Both calls arrive in a single chunk — the residue handling after the
        // first close must re-enter the outside transition and find the second
        // open sentinel within the same AppendChunk call.
        var (safe, fragments) = RunStream(parser,
            """<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>""");

        Assert.Equal(string.Empty, safe);
        Assert.Equal(2, fragments.Count);
        Assert.Equal(0, fragments[0].Index);
        Assert.Equal(1, fragments[1].Index);
    }

    [Fact]
    public void Hermes_PartialPrefix_NotEmittedAsSafe()
    {
        var parser = new HermesToolCallParser().CreateIncremental();

        // First chunk ends with what could be the start of <tool_call>;
        // the parser must hold those bytes back rather than leak them as content.
        var r1 = parser.AppendChunk("hello <tool");

        Assert.Equal("hello ", r1.SafeText);
        Assert.Empty(r1.Fragments);

        // Confirm the held-back bytes are still inside the parser by completing
        // the open sentinel and verifying state flips into the call.
        var r2 = parser.AppendChunk("_call>");
        // No safe text should be emitted from this chunk because everything
        // completed the sentinel and the call has yet to produce a fragment.
        Assert.Equal(string.Empty, r2.SafeText);
    }

    [Fact]
    public void Hermes_PartialPrefix_NotAToolCall_FlushedOnEnd()
    {
        var parser = new HermesToolCallParser().CreateIncremental();

        // A tail that looks like a sentinel prefix but isn't followed by one
        // must be released on flush rather than swallowed.
        var (safe, _) = RunStream(parser, "actually <tool");

        Assert.Equal("actually <tool", safe);
    }

    [Fact]
    public void Hermes_UnterminatedCall_FlushClosesIt()
    {
        var parser = new HermesToolCallParser().CreateIncremental();

        // No closing </tool_call> tag — flush must still emit a closing fragment
        // so the consumer doesn't see a dangling call.
        var (safe, fragments) = RunStream(parser,
            """<tool_call>{"name": "x", "arguments": {"k": "v"}}""");

        Assert.Equal(string.Empty, safe);
        Assert.NotEmpty(fragments);
        Assert.Equal("x", fragments[0].Name);
        Assert.True(fragments[^1].IsLast);
    }

    [Fact]
    public void Hermes_ArgumentsDelta_ReconstructsArgumentsJson()
    {
        var parser = new HermesToolCallParser().CreateIncremental();

        var (_, fragments) = RunStream(parser,
            """<tool_call>{"name": "f", "arguments": {"a": 1, "b": [1,2,3], "c": "hello"}}</tool_call>""");

        // Concatenated arguments deltas must form a valid arguments JSON object.
        string joined = string.Concat(fragments.Select(f => f.ArgumentsDelta ?? string.Empty));
        using var doc = System.Text.Json.JsonDocument.Parse(joined);
        Assert.Equal(1, doc.RootElement.GetProperty("a").GetInt32());
        Assert.Equal(3, doc.RootElement.GetProperty("b").GetArrayLength());
        Assert.Equal("hello", doc.RootElement.GetProperty("c").GetString());
    }

    // ─────────────────────────────────────────────────────────────────────
    // Mistral — suppress mode: no leaked content, single closing fragment
    // ─────────────────────────────────────────────────────────────────────

    [Fact]
    public void Mistral_BufferedSuppressMode_NoLeakedContent()
    {
        var parser = new MistralToolCallParser().CreateIncremental();

        var (safe, fragments) = RunStream(parser,
            "[TOOL_CALLS]",
            "get_weather",
            "[ARGS]",
            """{"location": "Berlin"}""");

        Assert.Equal(string.Empty, safe);
        Assert.True(parser.HasEmittedAnyFragment);
        Assert.Single(fragments);
        Assert.Equal("get_weather", fragments[0].Name);
        Assert.Contains("Berlin", fragments[0].ArgumentsDelta);
        Assert.True(fragments[0].IsLast);
    }

    [Fact]
    public void Mistral_NoToolCall_AllTextIsSafe()
    {
        var parser = new MistralToolCallParser().CreateIncremental();

        var (safe, fragments) = RunStream(parser,
            "Just chatting, no tools today.");

        Assert.Equal("Just chatting, no tools today.", safe);
        Assert.Empty(fragments);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Llama — suppress mode via <|python_tag|>
    // ─────────────────────────────────────────────────────────────────────

    [Fact]
    public void Llama_BufferedFallback_NoLeakedMarkupAtFlush()
    {
        // Llama uses the buffer-and-parse-at-flush fallback (see
        // LlamaToolCallParser.CreateIncremental). Each AppendChunk returns
        // Empty; the full classification happens at Flush. The contract this
        // protects: no raw <|python_tag|> markup leaks into delta.content
        // (because nothing leaks until Flush, and at Flush the parser routes
        // a recognised call into delta.tool_calls).
        var parser = new LlamaToolCallParser().CreateIncremental();

        Assert.Equal(ToolCallParseResult.Empty, parser.AppendChunk("<|python_tag|>"));
        Assert.Equal(ToolCallParseResult.Empty,
            parser.AppendChunk("""{"name": "f", "parameters": {"x": 1}}"""));

        var flush = parser.Flush();
        Assert.Equal(string.Empty, flush.SafeText);
        Assert.NotEmpty(flush.Fragments);
        Assert.Equal("f", flush.Fragments[0].Name);
        Assert.True(flush.Fragments[0].IsLast);
        Assert.True(parser.HasEmittedAnyFragment);
    }

    [Fact]
    public void Llama_NoToolCall_AllTextEmittedAtFlush()
    {
        var parser = new LlamaToolCallParser().CreateIncremental();

        // No tool-call markers — buffered text falls through as safe text on flush.
        Assert.Equal(ToolCallParseResult.Empty, parser.AppendChunk("Just a regular reply."));
        var flush = parser.Flush();
        Assert.Equal("Just a regular reply.", flush.SafeText);
        Assert.Empty(flush.Fragments);
        Assert.False(parser.HasEmittedAnyFragment);
    }
}
