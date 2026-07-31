using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server.Endpoints;
using DotLLM.Tokenizers;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Endpoint-level tests for the Anthropic streaming SSE surface: they drive
/// <see cref="MessagesEndpoint.WriteMessageStreamAsync"/> with a scripted token stream
/// and assert the emitted event names, ordering and JSON payload shapes. No model is
/// loaded — the token source and the model gate are injected.
/// </summary>
public sealed class AnthropicStreamingTests
{
    /// <summary>One SSE frame: the <c>event:</c> name and its parsed <c>data:</c> payload.</summary>
    private readonly record struct SseFrame(string Event, JsonElement Data);

    private static async IAsyncEnumerable<GenerationToken> Tokens(
        params (string Text, FinishReason? Finish)[] script)
    {
        foreach (var (text, finish) in script)
        {
            await Task.Yield();
            yield return new GenerationToken(0, text, finish);
        }
    }

    /// <summary>Pass-through gate standing in for <c>ServerState.ExecuteAsync</c>.</summary>
    private static Task NoGate(Func<Task> work, CancellationToken ct) => work();

    private sealed class FixedToolCallParser(ToolCall[]? result) : IToolCallParser
    {
        public ToolCall[]? TryParse(string generatedText) => result;
        public bool IsToolCallStart(string text) => false;
    }

    private static async Task<SseFrame[]> RunAsync(
        IAsyncEnumerable<GenerationToken> tokens,
        IToolCallParser? parser = null,
        string[]? stopSequences = null)
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;

        await MessagesEndpoint.WriteMessageStreamAsync(
            ctx, _ => tokens, NoGate, parser, stopSequences,
            messageId: "msg_test", modelId: "test-model", promptTokenCount: 7,
            CancellationToken.None);

        Assert.Equal("text/event-stream", ctx.Response.ContentType);
        // Connection-specific headers are illegal over HTTP/2 and must not be emitted.
        Assert.False(ctx.Response.Headers.ContainsKey("Connection"));

        return ParseSse(Encoding.UTF8.GetString(body.ToArray()));
    }

    private static SseFrame[] ParseSse(string raw)
    {
        var frames = new List<SseFrame>();
        foreach (var block in raw.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            var lines = block.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            string name = lines[0]["event: ".Length..];
            string data = lines[1]["data: ".Length..];
            frames.Add(new SseFrame(name, JsonDocument.Parse(data).RootElement.Clone()));
        }
        return [.. frames];
    }

    // --- Text-only stream ---------------------------------------------------

    [Fact]
    public async Task Streaming_TextOnly_EmitsExpectedEventSequence()
    {
        var frames = await RunAsync(Tokens(("Hel", null), ("lo", FinishReason.Stop)));

        Assert.Equal(
            ["message_start", "content_block_start", "ping", "content_block_delta",
             "content_block_delta", "content_block_stop", "message_delta", "message_stop"],
            frames.Select(f => f.Event));
    }

    [Fact]
    public async Task Streaming_TextOnly_EmitsExpectedPayloadShapes()
    {
        var frames = await RunAsync(Tokens(("Hel", null), ("lo", FinishReason.Stop)));

        var start = frames[0].Data;
        Assert.Equal("message_start", start.GetProperty("type").GetString());
        var startMsg = start.GetProperty("message");
        Assert.Equal("msg_test", startMsg.GetProperty("id").GetString());
        Assert.Equal("message", startMsg.GetProperty("type").GetString());
        Assert.Equal("assistant", startMsg.GetProperty("role").GetString());
        Assert.Equal("test-model", startMsg.GetProperty("model").GetString());
        Assert.Equal(JsonValueKind.Null, startMsg.GetProperty("stop_reason").ValueKind);
        Assert.Equal(7, startMsg.GetProperty("usage").GetProperty("input_tokens").GetInt32());

        var blockStart = frames[1].Data;
        Assert.Equal(0, blockStart.GetProperty("index").GetInt32());
        Assert.Equal("text", blockStart.GetProperty("content_block").GetProperty("type").GetString());

        var delta = frames[3].Data.GetProperty("delta");
        Assert.Equal("text_delta", delta.GetProperty("type").GetString());
        Assert.Equal("Hel", delta.GetProperty("text").GetString());
        Assert.Equal("lo", frames[4].Data.GetProperty("delta").GetProperty("text").GetString());

        Assert.Equal(0, frames[5].Data.GetProperty("index").GetInt32());

        var messageDelta = frames[6].Data;
        Assert.Equal("end_turn", messageDelta.GetProperty("delta").GetProperty("stop_reason").GetString());
        Assert.Equal(JsonValueKind.Null, messageDelta.GetProperty("delta").GetProperty("stop_sequence").ValueKind);
        Assert.Equal(2, messageDelta.GetProperty("usage").GetProperty("output_tokens").GetInt32());

        Assert.Equal("message_stop", frames[7].Data.GetProperty("type").GetString());
    }

    [Fact]
    public async Task Streaming_MaxTokens_ReportsMaxTokensStopReason()
    {
        var frames = await RunAsync(Tokens(("hi", FinishReason.Length)));

        var messageDelta = frames.Single(f => f.Event == "message_delta").Data;
        Assert.Equal("max_tokens", messageDelta.GetProperty("delta").GetProperty("stop_reason").GetString());
    }

    [Fact]
    public async Task Streaming_MatchingStopSequence_ReportsStopSequence()
    {
        var frames = await RunAsync(
            Tokens(("all done", null), ("END", FinishReason.Stop)),
            stopSequences: ["END"]);

        var delta = frames.Single(f => f.Event == "message_delta").Data.GetProperty("delta");
        Assert.Equal("stop_sequence", delta.GetProperty("stop_reason").GetString());
        Assert.Equal("END", delta.GetProperty("stop_sequence").GetString());
    }

    // --- tool_use stream ----------------------------------------------------

    [Fact]
    public async Task Streaming_ToolUse_EmitsToolBlockAfterTextBlock()
    {
        var parser = new FixedToolCallParser(
            [new ToolCall("toolu_1", "get_weather", """{"city":"Paris"}""")]);
        var frames = await RunAsync(Tokens(("calling", FinishReason.Stop)), parser);

        Assert.Equal(
            ["message_start", "content_block_start", "ping", "content_block_delta",
             "content_block_stop", "content_block_start", "content_block_delta",
             "content_block_stop", "message_delta", "message_stop"],
            frames.Select(f => f.Event));

        // The tool_use block opens at index 1, after the text block at index 0.
        var toolStart = frames[5].Data;
        Assert.Equal(1, toolStart.GetProperty("index").GetInt32());
        var block = toolStart.GetProperty("content_block");
        Assert.Equal("tool_use", block.GetProperty("type").GetString());
        Assert.Equal("toolu_1", block.GetProperty("id").GetString());
        Assert.Equal("get_weather", block.GetProperty("name").GetString());
        // Anthropic opens a tool_use block with an empty input; arguments arrive as deltas.
        Assert.Equal(JsonValueKind.Object, block.GetProperty("input").ValueKind);
        Assert.Empty(block.GetProperty("input").EnumerateObject());

        var toolDelta = frames[6].Data.GetProperty("delta");
        Assert.Equal("input_json_delta", toolDelta.GetProperty("type").GetString());
        Assert.Equal("""{"city":"Paris"}""", toolDelta.GetProperty("partial_json").GetString());

        Assert.Equal(1, frames[7].Data.GetProperty("index").GetInt32());

        var messageDelta = frames[8].Data;
        Assert.Equal("tool_use", messageDelta.GetProperty("delta").GetProperty("stop_reason").GetString());
    }

    [Fact]
    public async Task Streaming_ToolParserFindsNothing_EmitsTextOnlySequence()
    {
        var frames = await RunAsync(Tokens(("plain", FinishReason.Stop)), new FixedToolCallParser(null));

        Assert.DoesNotContain(frames, f =>
            f.Event == "content_block_start" &&
            f.Data.GetProperty("content_block").GetProperty("type").GetString() == "tool_use");
        Assert.Equal("end_turn",
            frames.Single(f => f.Event == "message_delta").Data.GetProperty("delta")
                  .GetProperty("stop_reason").GetString());
    }
}
