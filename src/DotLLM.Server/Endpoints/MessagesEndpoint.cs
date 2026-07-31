using System.Text;
using System.Text.Json;
using DotLLM.Engine;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// Anthropic-compatible Messages API:
/// <list type="bullet">
/// <item><c>POST /v1/messages</c> — non-streaming (JSON) and streaming (named SSE events).</item>
/// <item><c>POST /v1/messages/count_tokens</c> — prompt token count.</item>
/// </list>
/// Reshapes the Anthropic wire format onto the shared engine pipeline that also
/// backs <see cref="ChatCompletionEndpoint"/>. Reference: https://docs.anthropic.com/en/api/messages
/// </summary>
public static class MessagesEndpoint
{
    private static readonly string[] CommonStopSequences =
        ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "</s>", "</tool_call>"];

    public static void Map(WebApplication app)
    {
        app.MapPost("/v1/messages", HandleAsync);
        app.MapPost("/v1/messages/count_tokens", HandleCountTokensAsync);
    }

    private static async Task HandleAsync(
        AnthropicMessagesRequest request,
        ServerState state,
        HttpContext httpContext)
    {
        if (!state.IsReady || state.Generator is null || state.ChatTemplate is null)
        {
            await WriteErrorAsync(httpContext, 503, "api_error", "No model loaded");
            return;
        }

        var validationError = ValidateRequest(request, requireMaxTokens: true);
        if (validationError is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error", validationError);
            return;
        }

        var ct = httpContext.RequestAborted;
        var messageId = AnthropicConverter.GenerateMessageId();
        var modelId = request.Model ?? state.Options.ModelId;
        var generator = state.Generator;

        var messages = AnthropicConverter.ToMessages(request);
        var tools = AnthropicConverter.ToTools(request.Tools);

        var templateOptions = new ChatTemplateOptions
        {
            AddGenerationPrompt = true,
            Tools = tools,
        };
        string prompt = state.ChatTemplate.Apply(messages, templateOptions);

        int maxTokens = request.MaxTokens ?? state.SamplingDefaults.MaxTokens;
        var promptError = RequestValidator.ValidatePromptLength(
            prompt, state.Tokenizer!, state.Config!.MaxSequenceLength,
            maxTokens, out int effectiveMaxTokens, out int promptTokenCount);
        if (promptError is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error", promptError);
            return;
        }

        var options = AnthropicConverter.ToInferenceOptions(request, CommonStopSequences,
            state.SamplingDefaults,
            new DotLLM.Core.Configuration.ThreadingConfig(state.Options.Threads, state.Options.DecodeThreads));
        options = options with { MaxTokens = effectiveMaxTokens };

        if (request.Stream)
            await HandleStreamingAsync(request, generator, state, httpContext, prompt, options,
                messageId, modelId, tools, promptTokenCount, ct);
        else
            await HandleNonStreamingAsync(request, generator, state, httpContext, prompt, options,
                messageId, modelId, tools, ct);
    }

    private static async Task HandleNonStreamingAsync(
        AnthropicMessagesRequest request,
        TextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        DotLLM.Core.Configuration.InferenceOptions options,
        string messageId, string modelId,
        ToolDefinition[]? tools,
        CancellationToken ct)
    {
        InferenceResponse? result = null;
        await state.ExecuteAsync(async () =>
        {
            result = generator.Generate(prompt, options);
        }, ct);

        string text = result!.Text;
        ToolCall[]? toolCalls = null;
        var finishReason = result.FinishReason;

        if (state.ToolCallParser is not null && tools is { Length: > 0 })
        {
            var enriched = ToolCallDetector.DetectToolCalls(result, state.ToolCallParser);
            text = enriched.Text;
            toolCalls = enriched.ToolCalls;
            finishReason = enriched.FinishReason;
        }

        // Determine whether a caller-supplied stop sequence ended generation, and strip it.
        bool matchedStopSequence = false;
        if (finishReason == FinishReason.Stop)
            text = StripAndDetectStopSequence(text, request.StopSequences, options.StopSequences,
                out matchedStopSequence, out _);

        AnthropicContentBlockDto[] content;
        string stopReason;
        if (toolCalls is { Length: > 0 })
        {
            content = AnthropicConverter.ToToolUseBlocks(toolCalls);
            stopReason = "tool_use";
        }
        else
        {
            content = [new AnthropicContentBlockDto { Type = "text", Text = text }];
            stopReason = AnthropicConverter.ToStopReason(finishReason, matchedStopSequence);
        }

        string? stopSequence = stopReason == "stop_sequence"
            ? MatchStopSequence(result.Text, request.StopSequences)
            : null;

        var response = new AnthropicMessageResponse
        {
            Id = messageId,
            Model = modelId,
            Content = content,
            StopReason = stopReason,
            StopSequence = stopSequence,
            Usage = new AnthropicUsageDto
            {
                InputTokens = result.PromptTokenCount,
                OutputTokens = result.GeneratedTokenCount,
            },
        };

        httpContext.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(httpContext.Response.Body, response,
            ServerJsonContext.Default.AnthropicMessageResponse, ct);
    }

    private static async Task HandleStreamingAsync(
        AnthropicMessagesRequest request,
        TextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        DotLLM.Core.Configuration.InferenceOptions options,
        string messageId, string modelId,
        ToolDefinition[]? tools,
        int promptTokenCount,
        CancellationToken ct)
        => await WriteMessageStreamAsync(
            httpContext,
            innerCt => generator.GenerateStreamingTokensAsync(prompt, options, innerCt),
            state.ExecuteAsync,
            tools is { Length: > 0 } ? state.ToolCallParser : null,
            request.StopSequences,
            messageId, modelId, promptTokenCount, ct);

    /// <summary>
    /// Emits the Anthropic SSE event sequence for one streaming request:
    /// <c>message_start</c>, <c>content_block_start</c> + <c>ping</c>, a
    /// <c>content_block_delta</c> per generated token, <c>content_block_stop</c>,
    /// an optional start/delta/stop trio per detected <c>tool_use</c> block, then
    /// <c>message_delta</c> and <c>message_stop</c>.
    /// </summary>
    /// <remarks>
    /// The token source and the model-serialisation gate are injected rather than read
    /// from <see cref="ServerState"/> so the emitted event sequence can be asserted in
    /// unit tests without loading a model. <paramref name="execute"/> wraps only the
    /// generation loop, so <c>message_start</c> still reaches the client before the
    /// request queues behind the model lock.
    /// </remarks>
    internal static async Task WriteMessageStreamAsync(
        HttpContext httpContext,
        Func<CancellationToken, IAsyncEnumerable<GenerationToken>> tokenSource,
        Func<Func<Task>, CancellationToken, Task> execute,
        IToolCallParser? toolCallParser,
        string[]? requestStopSequences,
        string messageId,
        string modelId,
        int promptTokenCount,
        CancellationToken ct)
    {
        httpContext.Response.ContentType = "text/event-stream";
        // No `Connection: keep-alive` — it is a connection-specific header that is
        // illegal over HTTP/2 and HTTP/3, and content-type + no-cache is all SSE needs.
        httpContext.Response.Headers.CacheControl = "no-cache";

        // message_start — input_tokens known up front from the prompt.
        var startMessage = new AnthropicMessageResponse
        {
            Id = messageId,
            Model = modelId,
            Content = [],
            StopReason = null,
            StopSequence = null,
            Usage = new AnthropicUsageDto { InputTokens = promptTokenCount, OutputTokens = 0 },
        };
        await WriteEventAsync(httpContext, "message_start",
            new AnthropicMessageStartEvent { Message = startMessage },
            ServerJsonContext.Default.AnthropicMessageStartEvent, ct);

        // Text content block opens at index 0.
        await WriteEventAsync(httpContext, "content_block_start",
            new AnthropicContentBlockStartEvent
            {
                Index = 0,
                ContentBlock = new AnthropicContentBlockDto { Type = "text", Text = "" },
            },
            ServerJsonContext.Default.AnthropicContentBlockStartEvent, ct);
        await WriteEventAsync(httpContext, "ping", new AnthropicPingEvent(),
            ServerJsonContext.Default.AnthropicPingEvent, ct);

        var sb = new StringBuilder();
        FinishReason finishReason = FinishReason.Length;
        int completionTokens = 0;

        await execute(async () =>
        {
            await foreach (var token in tokenSource(ct))
            {
                if (token.Text.Length > 0)
                {
                    completionTokens++;
                    sb.Append(token.Text);
                    await WriteEventAsync(httpContext, "content_block_delta",
                        new AnthropicContentBlockDeltaEvent
                        {
                            Index = 0,
                            Delta = new AnthropicStreamDeltaDto { Type = "text_delta", Text = token.Text },
                        },
                        ServerJsonContext.Default.AnthropicContentBlockDeltaEvent, ct);
                }

                if (token.FinishReason.HasValue)
                    finishReason = token.FinishReason.Value;
            }
        }, ct);

        // Close the text block.
        await WriteEventAsync(httpContext, "content_block_stop",
            new AnthropicContentBlockStopEvent { Index = 0 },
            ServerJsonContext.Default.AnthropicContentBlockStopEvent, ct);

        // Post-generation tool-call detection (mirrors the OpenAI streaming endpoint).
        string text = sb.ToString();
        ToolCall[]? toolCalls = null;
        if (toolCallParser is not null)
        {
            toolCalls = toolCallParser.TryParse(text);
            if (toolCalls is { Length: > 0 })
                finishReason = FinishReason.ToolCalls;
        }

        // Emit detected tool calls as tool_use blocks after the text block.
        if (toolCalls is { Length: > 0 })
        {
            var blocks = AnthropicConverter.ToToolUseBlocks(toolCalls);
            for (int i = 0; i < blocks.Length; i++)
            {
                int index = i + 1;
                var block = blocks[i];
                await WriteEventAsync(httpContext, "content_block_start",
                    new AnthropicContentBlockStartEvent
                    {
                        Index = index,
                        ContentBlock = new AnthropicContentBlockDto
                        {
                            Type = "tool_use",
                            Id = block.Id,
                            Name = block.Name,
                            Input = AnthropicConverter.ParseInput("{}"),
                        },
                    },
                    ServerJsonContext.Default.AnthropicContentBlockStartEvent, ct);
                await WriteEventAsync(httpContext, "content_block_delta",
                    new AnthropicContentBlockDeltaEvent
                    {
                        Index = index,
                        Delta = new AnthropicStreamDeltaDto
                        {
                            Type = "input_json_delta",
                            PartialJson = block.Input?.GetRawText() ?? "{}",
                        },
                    },
                    ServerJsonContext.Default.AnthropicContentBlockDeltaEvent, ct);
                await WriteEventAsync(httpContext, "content_block_stop",
                    new AnthropicContentBlockStopEvent { Index = index },
                    ServerJsonContext.Default.AnthropicContentBlockStopEvent, ct);
            }
        }

        bool matchedStopSequence = false;
        if (finishReason == FinishReason.Stop)
            matchedStopSequence = MatchStopSequence(text, requestStopSequences) is not null;

        string stopReason = AnthropicConverter.ToStopReason(finishReason, matchedStopSequence);
        string? stopSequence = stopReason == "stop_sequence"
            ? MatchStopSequence(text, requestStopSequences)
            : null;

        await WriteEventAsync(httpContext, "message_delta",
            new AnthropicMessageDeltaEvent
            {
                Delta = new AnthropicMessageDeltaBody { StopReason = stopReason, StopSequence = stopSequence },
                Usage = new AnthropicUsageDto { InputTokens = promptTokenCount, OutputTokens = completionTokens },
            },
            ServerJsonContext.Default.AnthropicMessageDeltaEvent, ct);

        await WriteEventAsync(httpContext, "message_stop", new AnthropicMessageStopEvent(),
            ServerJsonContext.Default.AnthropicMessageStopEvent, ct);
        await httpContext.Response.Body.FlushAsync(ct);
    }

    private static async Task HandleCountTokensAsync(
        AnthropicMessagesRequest request,
        ServerState state,
        HttpContext httpContext)
    {
        if (!state.IsReady || state.ChatTemplate is null || state.Tokenizer is null)
        {
            await WriteErrorAsync(httpContext, 503, "api_error", "No model loaded");
            return;
        }

        var validationError = ValidateRequest(request, requireMaxTokens: false);
        if (validationError is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error", validationError);
            return;
        }

        var ct = httpContext.RequestAborted;
        var messages = AnthropicConverter.ToMessages(request);
        var tools = AnthropicConverter.ToTools(request.Tools);
        string prompt = state.ChatTemplate.Apply(messages,
            new ChatTemplateOptions { AddGenerationPrompt = true, Tools = tools });

        int count = state.Tokenizer.CountTokens(prompt);

        httpContext.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(httpContext.Response.Body,
            new AnthropicCountTokensResponse { InputTokens = count },
            ServerJsonContext.Default.AnthropicCountTokensResponse, ct);
    }

    /// <summary>Validates the structural invariants of an Anthropic request.</summary>
    internal static string? ValidateRequest(AnthropicMessagesRequest request, bool requireMaxTokens)
    {
        if (request.Messages is null || request.Messages.Length == 0)
            return "messages: at least one message is required";

        if (request.Messages.Length > RequestValidator.MaxMessages)
            return $"messages: array exceeds maximum of {RequestValidator.MaxMessages}";

        if (requireMaxTokens)
        {
            if (!request.MaxTokens.HasValue)
                return "max_tokens: field required";
            if (request.MaxTokens.Value <= 0)
                return "max_tokens: must be a positive integer";
        }

        // Roles and content kinds are checked here rather than left to the converter:
        // ToMessages passes `role` straight through to the chat template, so an unchecked
        // `system` (or arbitrary) role would let a caller inject a system turn mid-
        // conversation, and an unchecked content kind would silently flatten to an empty
        // message instead of surfacing the client's mistake as a 400.
        for (int i = 0; i < request.Messages.Length; i++)
        {
            var msg = request.Messages[i];
            if (msg is null)
                return $"messages[{i}]: must be an object";

            if (msg.Role is not ("user" or "assistant"))
                return $"messages[{i}].role: must be one of \"user\", \"assistant\"";

            if (msg.Content.ValueKind is not (JsonValueKind.String or JsonValueKind.Array))
                return $"messages[{i}].content: must be a string or an array of content blocks";
        }

        return null;
    }

    /// <summary>
    /// Returns the request stop sequence that is a suffix of <paramref name="text"/>,
    /// or null if none matches.
    /// </summary>
    private static string? MatchStopSequence(string text, string[]? stopSequences)
    {
        if (stopSequences is null)
            return null;
        foreach (var seq in stopSequences)
        {
            if (!string.IsNullOrEmpty(seq) && text.EndsWith(seq, StringComparison.Ordinal))
                return seq;
        }
        return null;
    }

    /// <summary>
    /// Strips a trailing stop-sequence suffix from <paramref name="text"/> and reports
    /// whether a caller-supplied stop sequence matched.
    /// </summary>
    private static string StripAndDetectStopSequence(
        string text, string[]? requestStops, IReadOnlyList<string> allStops,
        out bool matchedRequestStop, out string? matched)
    {
        matchedRequestStop = false;
        matched = null;

        // Caller-supplied stop sequences are reported as "stop_sequence".
        string? requestMatch = MatchStopSequence(text, requestStops);
        if (requestMatch is not null)
        {
            matchedRequestStop = true;
            matched = requestMatch;
            return text[..^requestMatch.Length];
        }

        // Built-in/template stop sequences are stripped but reported as "end_turn".
        foreach (var seq in allStops)
        {
            if (text.EndsWith(seq, StringComparison.Ordinal))
            {
                matched = seq;
                return text[..^seq.Length];
            }
        }
        return text;
    }

    private static async Task WriteEventAsync<T>(
        HttpContext ctx, string eventName, T payload,
        System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, CancellationToken ct)
    {
        await ctx.Response.WriteAsync($"event: {eventName}\n", ct);
        await ctx.Response.WriteAsync("data: ", ct);
        await JsonSerializer.SerializeAsync(ctx.Response.Body, payload, typeInfo, ct);
        await ctx.Response.WriteAsync("\n\n", ct);
        await ctx.Response.Body.FlushAsync(ct);
    }

    private static async Task WriteErrorAsync(
        HttpContext ctx, int statusCode, string errorType, string message)
    {
        ctx.Response.StatusCode = statusCode;
        await ctx.Response.WriteAsJsonAsync(
            new AnthropicErrorResponse { Error = new AnthropicErrorBody { Type = errorType, Message = message } },
            ServerJsonContext.Default.AnthropicErrorResponse,
            contentType: null,
            ctx.RequestAborted);
    }
}
