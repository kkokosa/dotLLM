using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Extensions.AI;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Microsoft.Extensions.AI;
using Xunit;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using EngineChatMessage = DotLLM.Tokenizers.ChatMessage;

namespace DotLLM.Tests.Unit.Extensions;

/// <summary>
/// Behavioural tests for <see cref="DotLLMChatClient"/> driven by a scripted model:
/// the "model" emits a predetermined token sequence, so response mapping, the streaming
/// update sequence, and tool-call detection are all deterministic without a real model.
/// </summary>
public sealed class DotLLMChatClientTests
{
    // --- Non-streaming ------------------------------------------------------

    [Fact]
    public async Task GetResponseAsync_MapsTextFinishReasonUsageAndModelId()
    {
        using var client = CreateClient(["Hello", " world", Eos]);

        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hi")]);

        Assert.Equal("Hello world", response.Text);
        Assert.Equal(ChatRole.Assistant, response.Messages[0].Role);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.Equal("scripted", response.ModelId);
        Assert.NotNull(response.Usage);
        // The stub tokenizer encodes any prompt to a single token.
        Assert.Equal(1, response.Usage!.InputTokenCount);
        Assert.Equal(2, response.Usage.OutputTokenCount);
        Assert.Equal(3, response.Usage.TotalTokenCount);
    }

    [Fact]
    public async Task GetResponseAsync_HittingMaxTokens_ReportsLengthFinishReason()
    {
        using var client = CreateClient(["a", "b", "c", "d"]);

        var response = await client.GetResponseAsync(
            [new ChatMessage(ChatRole.User, "Hi")],
            new ChatOptions { MaxOutputTokens = 2 });

        Assert.Equal("ab", response.Text);
        Assert.Equal(ChatFinishReason.Length, response.FinishReason);
    }

    [Fact]
    public async Task GetResponseAsync_PassesRenderedTemplatePromptWithTools()
    {
        var template = new RecordingChatTemplate();
        using var client = CreateClient(["ok", Eos], template: template);

        await client.GetResponseAsync(
            [new ChatMessage(ChatRole.User, "Hi")],
            new ChatOptions { Tools = [WeatherFunction] });

        Assert.NotNull(template.LastOptions);
        Assert.True(template.LastOptions!.AddGenerationPrompt);
        Assert.Equal("get_weather", Assert.Single(template.LastOptions.Tools!).Name);
        Assert.Equal("user", Assert.Single(template.LastMessages!).Role);
    }

    [Fact]
    public async Task GetResponseAsync_ToolCallDetected_EmitsFunctionCallAndDropsText()
    {
        using var client = CreateClient(ToolCallScript, parser: new HermesToolCallParser());

        var response = await client.GetResponseAsync(
            [new ChatMessage(ChatRole.User, "Weather?")],
            new ChatOptions { Tools = [WeatherFunction] });

        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
        var call = Assert.IsType<FunctionCallContent>(Assert.Single(response.Messages[0].Contents));
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments!["city"]!.ToString());
    }

    [Fact]
    public async Task GetResponseAsync_ToolCallTextButNoToolsRequested_StaysText()
    {
        // No tools on the request → detection is off, so the raw text passes through.
        using var client = CreateClient(ToolCallScript, parser: new HermesToolCallParser());

        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Weather?")]);

        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.IsType<TextContent>(Assert.Single(response.Messages[0].Contents));
    }

    // --- Streaming ----------------------------------------------------------

    [Fact]
    public async Task GetStreamingResponseAsync_YieldsTextDeltasThenFinalFinishReason()
    {
        using var client = CreateClient(["Hel", "lo", Eos]);

        var updates = await CollectAsync(
            client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "Hi")]));

        Assert.Equal(3, updates.Count);
        Assert.Equal("Hel", updates[0].Text);
        Assert.Equal("lo", updates[1].Text);
        Assert.Null(updates[0].FinishReason);
        // Final update carries only the finish reason; the text was already delivered.
        Assert.Empty(updates[2].Contents);
        Assert.Equal(ChatFinishReason.Stop, updates[2].FinishReason);

        // All updates share one response/message id so they coalesce into a single message.
        Assert.Single(updates.Select(u => u.ResponseId).Distinct());
        Assert.Equal("Hello", updates.ToChatResponse().Text);
    }

    [Fact]
    public async Task GetStreamingResponseAsync_WithTools_BuffersTextAndEmitsToolCallsAtEnd()
    {
        using var client = CreateClient(ToolCallScript, parser: new HermesToolCallParser());

        var updates = await CollectAsync(client.GetStreamingResponseAsync(
            [new ChatMessage(ChatRole.User, "Weather?")],
            new ChatOptions { Tools = [WeatherFunction] }));

        // The raw tool-call syntax must never reach the caller as streamed text.
        var single = Assert.Single(updates);
        var call = Assert.IsType<FunctionCallContent>(Assert.Single(single.Contents));
        Assert.Equal("get_weather", call.Name);
        Assert.Equal(ChatFinishReason.ToolCalls, single.FinishReason);
    }

    [Fact]
    public async Task GetStreamingResponseAsync_WithTools_PlainText_IsDeliveredInFinalUpdate()
    {
        using var client = CreateClient(["No ", "tool ", "needed", Eos],
            parser: new HermesToolCallParser());

        var updates = await CollectAsync(client.GetStreamingResponseAsync(
            [new ChatMessage(ChatRole.User, "Hi")],
            new ChatOptions { Tools = [WeatherFunction] }));

        // Buffered because detection is active, but no text is lost.
        var single = Assert.Single(updates);
        Assert.Equal("No tool needed", single.Text);
        Assert.Equal(ChatFinishReason.Stop, single.FinishReason);
    }

    [Fact]
    public async Task GetStreamingResponseAsync_StreamingAndNonStreaming_AgreeOnContents()
    {
        using var streaming = CreateClient(ToolCallScript, parser: new HermesToolCallParser());
        using var blocking = CreateClient(ToolCallScript, parser: new HermesToolCallParser());
        var options = new ChatOptions { Tools = [WeatherFunction] };
        ChatMessage[] messages = [new ChatMessage(ChatRole.User, "Weather?")];

        var streamed = (await CollectAsync(streaming.GetStreamingResponseAsync(messages, options)))
            .ToChatResponse();
        var direct = await blocking.GetResponseAsync(messages, options);

        Assert.Equal(direct.FinishReason, streamed.FinishReason);
        Assert.Equal(direct.Text, streamed.Text);
        Assert.Equal(
            direct.Messages[0].Contents.Select(c => c.GetType()),
            streamed.Messages[0].Contents.Select(c => c.GetType()));
    }

    // --- Service resolution -------------------------------------------------

    [Fact]
    public void GetService_ResolvesMetadataAndSelf()
    {
        using var client = CreateClient([Eos]);

        var metadata = Assert.IsType<ChatClientMetadata>(client.GetService(typeof(ChatClientMetadata)));
        Assert.Equal("dotLLM", metadata.ProviderName);
        Assert.Equal("scripted", metadata.DefaultModelId);
        Assert.Same(client, client.GetService(typeof(IChatClient)));
        Assert.Null(client.GetService(typeof(ChatClientMetadata), serviceKey: "keyed"));
        Assert.Null(client.GetService(typeof(string)));
    }

    // --- Fixtures -----------------------------------------------------------

    /// <summary>Sentinel for the EOS token in a script — decodes to empty text.</summary>
    private const string Eos = " eos";

    private static readonly string[] ToolCallScript =
        ["<tool_call>", "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}", "</tool_call>", Eos];

    private static AIFunction WeatherFunction { get; } =
        AIFunctionFactory.Create((string city) => $"weather in {city}", "get_weather", "Get the weather");

    private static DotLLMChatClient CreateClient(
        string[] script, IToolCallParser? parser = null, IChatTemplate? template = null)
    {
        var tokenizer = new ScriptedTokenizer(script);
        var model = new ScriptedModel(tokenizer.ScriptedIds, tokenizer.VocabSize);
        var generator = new TextGenerator(model, tokenizer);
        return new DotLLMChatClient(
            generator, template ?? new RecordingChatTemplate(), modelId: "scripted", toolCallParser: parser);
    }

    private static async Task<List<ChatResponseUpdate>> CollectAsync(
        IAsyncEnumerable<ChatResponseUpdate> updates)
    {
        var list = new List<ChatResponseUpdate>();
        await foreach (var update in updates)
            list.Add(update);
        return list;
    }

    /// <summary>Chat template that records its inputs and renders a trivial prompt.</summary>
    private sealed class RecordingChatTemplate : IChatTemplate
    {
        public IReadOnlyList<EngineChatMessage>? LastMessages { get; private set; }
        public ChatTemplateOptions? LastOptions { get; private set; }

        public string Apply(IReadOnlyList<EngineChatMessage> messages, ChatTemplateOptions options)
        {
            LastMessages = messages;
            LastOptions = options;
            return string.Join('\n', messages.Select(m => $"{m.Role}: {m.Content}"));
        }
    }

    /// <summary>
    /// Tokenizer over a fixed script: token id <c>i + 1</c> decodes to <c>script[i]</c>.
    /// Id 0 is EOS (empty text); any prompt encodes to the single token id 1.
    /// </summary>
    private sealed class ScriptedTokenizer : ITokenizer
    {
        private readonly string[] _pieces;

        public ScriptedTokenizer(string[] script)
        {
            // Index 0 = EOS, then one entry per scripted piece.
            _pieces = ["", .. script.Select(s => s == Eos ? "" : s)];
            ScriptedIds = [.. Enumerable.Range(0, script.Length).Select(i => script[i] == Eos ? 0 : i + 1)];
        }

        /// <summary>Token ids the scripted model should emit, in order.</summary>
        public int[] ScriptedIds { get; }

        public int VocabSize => _pieces.Length;
        public int BosTokenId => 0;
        public int EosTokenId => 0;
        public int[] Encode(string text) => [1];
        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new System.Text.StringBuilder();
            foreach (int id in tokenIds)
                sb.Append(DecodeToken(id));
            return sb.ToString();
        }

        public string DecodeToken(int tokenId) =>
            (uint)tokenId < (uint)_pieces.Length ? _pieces[tokenId] : "";

        public int CountTokens(string text) => 1;
    }

    /// <summary>
    /// Model that ignores its input and returns one-hot logits selecting the next scripted
    /// token on each forward pass, making greedy generation fully deterministic.
    /// </summary>
    private sealed class ScriptedModel : IModel
    {
        private readonly int[] _scriptedIds;
        private readonly int _vocabSize;
        private int _step;

        public ScriptedModel(int[] scriptedIds, int vocabSize)
        {
            _scriptedIds = scriptedIds;
            _vocabSize = vocabSize;
        }

        public ModelConfig Config => new()
        {
            VocabSize = _vocabSize,
            NumLayers = 1,
            NumAttentionHeads = 1,
            NumKvHeads = 1,
            HiddenSize = 8,
            IntermediateSize = 16,
            HeadDim = 8,
            MaxSequenceLength = 128,
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache)
        {
            int batchSize = tokenIds.Length;
            nuint bytes = (nuint)((long)batchSize * _vocabSize * sizeof(float));
            nint ptr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
            NativeMemory.Clear((void*)ptr, bytes);

            int next = _scriptedIds[Math.Min(_step++, _scriptedIds.Length - 1)];
            float* dst = (float*)ptr;
            for (int b = 0; b < batchSize; b++)
                dst[(long)b * _vocabSize + next] = 100f;

            return new UnmanagedTensor(new TensorShape(batchSize, _vocabSize), DType.Float32, deviceId, ptr);
        }

        public void Dispose() { }
    }
}
