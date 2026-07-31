using System.Text.Json;
using DotLLM.Server;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for the server-side mapping of <see cref="ToolCallFragment"/> values
/// onto the OpenAI streaming <c>delta.tool_calls[]</c> wire shape, and for the
/// end-to-end SSE-flow simulation (drive an incremental parser from a fake
/// generated-token stream and assert the chunks the server would emit).
/// </summary>
public class ChatCompletionStreamingToolCallTests
{
    // ─────────────────────────────────────────────────────────────────────
    // RequestConverter.ToToolCallDeltaDto — opening vs argument-only mapping
    // ─────────────────────────────────────────────────────────────────────

    [Fact]
    public void ToToolCallDeltaDto_OpeningFragment_IncludesIdAndNameAndType()
    {
        var fragment = new ToolCallFragment(
            Index: 0,
            Id: "call_0",
            Name: "get_weather",
            ArgumentsDelta: """{"location": """,
            IsLast: false);

        var dto = RequestConverter.ToToolCallDeltaDto(fragment);

        Assert.Equal(0, dto.Index);
        Assert.Equal("call_0", dto.Id);
        Assert.Equal("function", dto.Type);
        Assert.NotNull(dto.Function);
        Assert.Equal("get_weather", dto.Function!.Name);
        Assert.Equal("""{"location": """, dto.Function.Arguments);
    }

    [Fact]
    public void ToToolCallDeltaDto_ArgumentOnlyFragment_OmitsIdAndTypeAndName()
    {
        var fragment = new ToolCallFragment(
            Index: 0,
            Id: null,
            Name: null,
            ArgumentsDelta: "\"Berlin\"}",
            IsLast: false);

        var dto = RequestConverter.ToToolCallDeltaDto(fragment);

        Assert.Equal(0, dto.Index);
        Assert.Null(dto.Id);
        Assert.Null(dto.Type);
        Assert.NotNull(dto.Function);
        Assert.Null(dto.Function!.Name);
        Assert.Equal("\"Berlin\"}", dto.Function.Arguments);
    }

    [Fact]
    public void ToToolCallDeltaDto_OpeningFragment_SerializesToOpenAIShape()
    {
        var fragment = new ToolCallFragment(
            Index: 0,
            Id: "call_0",
            Name: "get_weather",
            ArgumentsDelta: """{""",
            IsLast: false);

        var dto = RequestConverter.ToToolCallDeltaDto(fragment);

        // Round-trip through the source-generated serializer context to confirm
        // snake_case keys + nulls dropped — the on-wire shape OpenAI clients expect.
        var chunk = new ChatCompletionChunk
        {
            Id = "chatcmpl-x",
            Model = "test",
            Choices = [new ChatChunkChoiceDto
            {
                Delta = new ChatDeltaDto { ToolCalls = [dto] },
            }],
        };

        string json = JsonSerializer.Serialize(chunk, ServerJsonContext.Default.ChatCompletionChunk);

        // Required: index, id, type:"function", function.name, function.arguments.
        Assert.Contains("\"index\":0", json);
        Assert.Contains("\"id\":\"call_0\"", json);
        Assert.Contains("\"type\":\"function\"", json);
        Assert.Contains("\"function\":{", json);
        Assert.Contains("\"name\":\"get_weather\"", json);

        // Nullable fields outside the tool-call delta must not appear.
        Assert.DoesNotContain("\"content\":", json);
        Assert.DoesNotContain("\"role\":", json);
    }

    [Fact]
    public void ToToolCallDeltaDto_ArgumentsDelta_SerializesWithoutIdOrType()
    {
        var fragment = new ToolCallFragment(
            Index: 0,
            Id: null,
            Name: null,
            ArgumentsDelta: "\"Berlin\"}",
            IsLast: true);

        var dto = RequestConverter.ToToolCallDeltaDto(fragment);
        var chunk = new ChatCompletionChunk
        {
            Id = "chatcmpl-x",
            Model = "test",
            Choices = [new ChatChunkChoiceDto
            {
                Delta = new ChatDeltaDto { ToolCalls = [dto] },
            }],
        };

        string json = JsonSerializer.Serialize(chunk, ServerJsonContext.Default.ChatCompletionChunk);

        // Navigate the DOM rather than slicing the raw text: the previous `IndexOf("}]")` bound the
        // segment to the first "}]" after tool_calls, so any nested object appearing before the end
        // of the array — an arguments payload ending in `}]`, or a second parallel call — would
        // truncate the segment and silently weaken every DoesNotContain below it.
        using var doc = JsonDocument.Parse(json);
        var toolCalls = doc.RootElement
            .GetProperty("choices")[0]
            .GetProperty("delta")
            .GetProperty("tool_calls");

        var tc = Assert.Single(toolCalls.EnumerateArray().ToList());
        Assert.Equal(0, tc.GetProperty("index").GetInt32());

        // An arguments-only fragment carries neither the call id nor the opening `type`.
        Assert.False(tc.TryGetProperty("id", out _));
        Assert.False(tc.TryGetProperty("type", out _));

        var function = tc.GetProperty("function");
        Assert.False(function.TryGetProperty("name", out _));
        Assert.Equal("\"Berlin\"}", function.GetProperty("arguments").GetString());
    }

    /// <summary>
    /// A fragment carrying only an id must not emit <c>"function":{}</c>. Building the function DTO
    /// for any "opening" fragment produces an all-null object which WhenWritingNull collapses to an
    /// empty object — a shape the DTO contract documents as null and no OpenAI client expects.
    /// </summary>
    [Fact]
    public void ToToolCallDeltaDto_IdOnly_OmitsFunctionEntirely()
    {
        var dto = RequestConverter.ToToolCallDeltaDto(new ToolCallFragment(
            Index: 0, Id: "call_0", Name: null, ArgumentsDelta: null, IsLast: false));

        Assert.Null(dto.Function);
        Assert.Equal("function", dto.Type);

        var chunk = new ChatCompletionChunk
        {
            Id = "chatcmpl-x",
            Model = "test",
            Choices = [new ChatChunkChoiceDto { Delta = new ChatDeltaDto { ToolCalls = [dto] } }],
        };
        string json = JsonSerializer.Serialize(chunk, ServerJsonContext.Default.ChatCompletionChunk);

        // Check the key via the DOM: a substring test for "function" also matches the *value* of
        // "type", which is legitimately present on this fragment.
        using var doc = JsonDocument.Parse(json);
        var tc = doc.RootElement.GetProperty("choices")[0]
            .GetProperty("delta").GetProperty("tool_calls")[0];
        Assert.False(tc.TryGetProperty("function", out _));
        Assert.Equal("function", tc.GetProperty("type").GetString());
    }

    /// <summary>
    /// <c>type:"function"</c> marks the chunk that opens a call, and on the wire that is the chunk
    /// carrying <c>id</c>. A name-only fragment must not announce a call the client cannot address.
    /// </summary>
    [Fact]
    public void ToToolCallDeltaDto_NameWithoutId_OmitsType()
    {
        var dto = RequestConverter.ToToolCallDeltaDto(new ToolCallFragment(
            Index: 0, Id: null, Name: "get_weather", ArgumentsDelta: null, IsLast: false));

        Assert.Null(dto.Type);
        Assert.Null(dto.Id);
        Assert.Equal("get_weather", dto.Function?.Name);
    }

    // ─────────────────────────────────────────────────────────────────────
    // End-to-end SSE-flow simulation (parser + DTO mapping, no HTTP)
    // ─────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Reproduces what the server's HandleStreamingAsync loop produces: feed
    /// generated-token text chunks through the incremental parser and collect
    /// the sequence of (content?, tool_calls?) deltas that would land on the wire.
    /// </summary>
    private static List<ChatDeltaDto> SimulateSse(
        IIncrementalToolCallParser parser, params string[] tokenTexts)
    {
        var deltas = new List<ChatDeltaDto>();
        foreach (var text in tokenTexts)
        {
            var r = parser.AppendChunk(text);
            CollectFromResult(r, deltas);
        }
        var flush = parser.Flush();
        CollectFromResult(flush, deltas);
        return deltas;
    }

    private static void CollectFromResult(ToolCallParseResult r, List<ChatDeltaDto> deltas)
    {
        if (r.SafeText.Length > 0)
            deltas.Add(new ChatDeltaDto { Content = r.SafeText });
        for (int i = 0; i < r.Fragments.Count; i++)
        {
            var dto = RequestConverter.ToToolCallDeltaDto(r.Fragments[i]);
            deltas.Add(new ChatDeltaDto { ToolCalls = [dto] });
        }
    }

    [Fact]
    public void HermesStream_ToolCallText_DoesNotAppearInContent()
    {
        var parser = new HermesToolCallParser().CreateIncremental();
        var deltas = SimulateSse(parser,
            "<tool_call>",
            """{"name": "get_weather", "arguments": {"location": "Berlin"}}""",
            "</tool_call>");

        // No content delta should appear — every character belongs to the call.
        foreach (var d in deltas)
        {
            Assert.True(string.IsNullOrEmpty(d.Content),
                $"Expected no delta.content, but saw: {d.Content}");
        }

        // First tool-call delta carries name; later deltas are argument-only.
        var toolDeltas = deltas.Where(d => d.ToolCalls is { Length: > 0 }).ToList();
        Assert.NotEmpty(toolDeltas);
        var firstTcDto = toolDeltas[0].ToolCalls![0];
        Assert.Equal("get_weather", firstTcDto.Function?.Name);
        Assert.Equal("function", firstTcDto.Type);
        Assert.Equal("call_0", firstTcDto.Id);
    }

    [Fact]
    public void HermesStream_MixedTextAndCall_RoutedToDifferentChannels()
    {
        var parser = new HermesToolCallParser().CreateIncremental();
        var deltas = SimulateSse(parser,
            "I'll check that. ",
            "<tool_call>",
            """{"name": "lookup", "arguments": {"id": 42}}""",
            "</tool_call>",
            " Got it.");

        // Plain text chunks must show up only as delta.content.
        string concatenatedContent = string.Concat(deltas.Select(d => d.Content ?? string.Empty));
        Assert.Contains("I'll check that.", concatenatedContent);
        Assert.Contains("Got it.", concatenatedContent);

        // Tool-call markup must NOT appear in delta.content.
        Assert.DoesNotContain("<tool_call>", concatenatedContent);
        Assert.DoesNotContain("</tool_call>", concatenatedContent);
        Assert.DoesNotContain("lookup", concatenatedContent);

        // A tool-call delta with the expected name must exist.
        Assert.Contains(deltas,
            d => d.ToolCalls is { Length: > 0 } && d.ToolCalls[0].Function?.Name == "lookup");
    }

    [Fact]
    public void HermesStream_ParallelCalls_IndexIncrementsAcrossDeltas()
    {
        var parser = new HermesToolCallParser().CreateIncremental();
        var deltas = SimulateSse(parser,
            """<tool_call>{"name": "a", "arguments": {}}</tool_call>""",
            """<tool_call>{"name": "b", "arguments": {}}</tool_call>""");

        // Collect the opening (name-carrying) tool-call deltas in order.
        var openingDtos = deltas
            .Where(d => d.ToolCalls is { Length: > 0 } && d.ToolCalls[0].Function?.Name is not null)
            .Select(d => d.ToolCalls![0])
            .ToList();

        Assert.Equal(2, openingDtos.Count);
        Assert.Equal(0, openingDtos[0].Index);
        Assert.Equal("a", openingDtos[0].Function!.Name);
        Assert.Equal(1, openingDtos[1].Index);
        Assert.Equal("b", openingDtos[1].Function!.Name);
    }

    [Fact]
    public void HermesStream_ArgumentsFragmentsConcatenateToValidJson()
    {
        var parser = new HermesToolCallParser().CreateIncremental();
        var deltas = SimulateSse(parser,
            "<tool_call>",
            "{\"name\": \"search\", ",
            "\"arguments\": {\"q\": \"dotnet\", \"limit\": 5}}",
            "</tool_call>");

        // Concatenate every argument slice from every tool-call delta for this call.
        string joinedArgs = string.Concat(
            deltas
                .Where(d => d.ToolCalls is { Length: > 0 })
                .SelectMany(d => d.ToolCalls!)
                .Where(t => t.Index == 0)
                .Select(t => t.Function?.Arguments ?? string.Empty));

        // The OpenAI contract: clients accumulate arguments slices and parse the
        // result. The concatenation must be valid JSON.
        using var doc = JsonDocument.Parse(joinedArgs);
        Assert.Equal("dotnet", doc.RootElement.GetProperty("q").GetString());
        Assert.Equal(5, doc.RootElement.GetProperty("limit").GetInt32());
    }

    [Fact]
    public void MistralStream_BufferedSuppressMode_NoLeakedMarkup()
    {
        var parser = new MistralToolCallParser().CreateIncremental();
        var deltas = SimulateSse(parser,
            "[TOOL_CALLS]",
            "get_weather",
            "[ARGS]",
            """{"location": "Berlin"}""");

        // Suppress-mode must keep all of the markup out of delta.content.
        string concatenatedContent = string.Concat(deltas.Select(d => d.Content ?? string.Empty));
        Assert.DoesNotContain("[TOOL_CALLS]", concatenatedContent);
        Assert.DoesNotContain("[ARGS]", concatenatedContent);
        Assert.DoesNotContain("get_weather", concatenatedContent);

        // Exactly one tool-call delta should arrive at flush with the parsed call.
        var toolDeltas = deltas.Where(d => d.ToolCalls is { Length: > 0 }).ToList();
        Assert.Single(toolDeltas);
        Assert.Equal("get_weather", toolDeltas[0].ToolCalls![0].Function?.Name);
    }
}
