using System.Text.Json;
using DotLLM.Engine;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// POST /v1/chat/completions — OpenAI-compatible chat completion endpoint.
/// Supports both non-streaming (JSON response) and streaming (SSE).
/// </summary>
public static class ChatCompletionEndpoint
{
    private static readonly string[] CommonStopSequences =
        ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "</s>", "</tool_call>"];

    public static void Map(WebApplication app) =>
        app.MapPost("/v1/chat/completions", HandleAsync);

    private static async Task HandleAsync(
        ChatCompletionRequest request,
        ServerState state,
        HttpContext httpContext)
    {
        if (!state.IsReady || state.Generator is null || state.ChatTemplate is null)
        {
            httpContext.Response.StatusCode = 503;
            await httpContext.Response.WriteAsJsonAsync(
                new ErrorResponse { Error = "No model loaded" },
                ServerJsonContext.Default.ErrorResponse,
                contentType: null,
                httpContext.RequestAborted);
            return;
        }

        // Validate request structure
        var validationError = RequestValidator.ValidateChatRequest(request);
        if (validationError is not null)
        {
            httpContext.Response.StatusCode = 400;
            await httpContext.Response.WriteAsJsonAsync(
                new ErrorResponse { Error = validationError },
                ServerJsonContext.Default.ErrorResponse,
                contentType: null,
                httpContext.RequestAborted);
            return;
        }

        var ct = httpContext.RequestAborted;
        var requestId = RequestConverter.GenerateRequestId();
        var modelId = state.Options.ModelId;
        var generator = state.Generator;

        // Convert DTOs to engine types
        var messages = RequestConverter.ToMessages(request.Messages);
        var tools = RequestConverter.ToTools(request.Tools);
        var toolChoice = RequestConverter.ParseToolChoice(request.ToolChoice);

        // Apply chat template
        var templateOptions = new ChatTemplateOptions
        {
            AddGenerationPrompt = true,
            Tools = tools,
        };
        string prompt = state.ChatTemplate.Apply(messages, templateOptions);

        // Validate prompt length against model context
        int maxTokens = request.MaxTokens ?? state.SamplingDefaults.MaxTokens;
        var promptError = RequestValidator.ValidatePromptLength(
            prompt, state.Tokenizer!, state.Config!.MaxSequenceLength,
            maxTokens, out int effectiveMaxTokens, out _);
        if (promptError is not null)
        {
            httpContext.Response.StatusCode = 400;
            await httpContext.Response.WriteAsJsonAsync(
                new ErrorResponse { Error = promptError },
                ServerJsonContext.Default.ErrorResponse,
                contentType: null,
                httpContext.RequestAborted);
            return;
        }

        // Build inference options with clamped max_tokens
        var stopSequences = CommonStopSequences;
        var options = RequestConverter.ToInferenceOptions(request, stopSequences,
            state.SamplingDefaults,
            new DotLLM.Core.Configuration.ThreadingConfig(
                state.Options.Threads, state.Options.DecodeThreads));
        options = options with { MaxTokens = effectiveMaxTokens };

        if (request.Stream)
            await HandleStreamingAsync(request, generator, state, httpContext, prompt, options,
                requestId, modelId, tools, ct);
        else
            await HandleNonStreamingAsync(request, generator, state, httpContext, prompt, options,
                requestId, modelId, tools, ct);
    }

    private static async Task HandleNonStreamingAsync(
        ChatCompletionRequest request,
        TextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        DotLLM.Core.Configuration.InferenceOptions options,
        string requestId, string modelId,
        ToolDefinition[]? tools,
        CancellationToken ct)
    {
        InferenceResponse? result = null;

        await state.ExecuteAsync(async () =>
        {
            result = generator.Generate(prompt, options);
        }, ct);

        // Detect tool calls
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

        // Strip stop sequence suffixes
        foreach (var seq in options.StopSequences)
        {
            if (text.EndsWith(seq, StringComparison.Ordinal))
            {
                text = text[..^seq.Length];
                break;
            }
        }

        var message = new ChatMessageDto
        {
            Role = "assistant",
            Content = toolCalls is { Length: > 0 } ? null : text,
            ToolCalls = toolCalls is { Length: > 0 }
                ? RequestConverter.ToToolCallDtos(toolCalls)
                : null,
        };

        var logprobsDto = result.Logprobs is { Length: > 0 }
            ? RequestConverter.ToLogprobsDto(result.Logprobs)
            : null;

        var response = new ChatCompletionResponse
        {
            Id = requestId,
            Model = modelId,
            Choices = [new ChatChoiceDto
            {
                Index = 0,
                Message = message,
                Logprobs = logprobsDto,
                FinishReason = RequestConverter.ToFinishReasonString(finishReason),
            }],
            Usage = new UsageDto
            {
                PromptTokens = result.PromptTokenCount,
                CompletionTokens = result.GeneratedTokenCount,
                TotalTokens = result.PromptTokenCount + result.GeneratedTokenCount,
            },
        };

        httpContext.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(httpContext.Response.Body, response, ServerJsonContext.Default.ChatCompletionResponse, ct);
    }

    private static async Task HandleStreamingAsync(
        ChatCompletionRequest request,
        TextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        DotLLM.Core.Configuration.InferenceOptions options,
        string requestId, string modelId,
        ToolDefinition[]? tools,
        CancellationToken ct)
    {
        httpContext.Response.ContentType = "text/event-stream";
        httpContext.Response.Headers.CacheControl = "no-cache";
        httpContext.Response.Headers.Connection = "keep-alive";

        // First chunk: role
        var roleChunk = new ChatCompletionChunk
        {
            Id = requestId,
            Model = modelId,
            Choices = [new ChatChunkChoiceDto
            {
                Delta = new ChatDeltaDto { Role = "assistant" },
            }],
        };
        await WriteSseChunk(httpContext, roleChunk, ct);

        FinishReason finishReason = FinishReason.Length;
        InferenceTimings? timings = null;
        int completionTokens = 0;

        // Per-request incremental tool-call parser. When tools are not in scope
        // (no parser registered or no tools attached to the request), this stays
        // null and the streaming path mirrors the legacy behaviour: every token's
        // text flows straight to delta.content. When tools are in scope, each
        // token's text is fed through the parser; the parser routes regular text
        // to delta.content and tool-call fragments to delta.tool_calls — per the
        // OpenAI SSE contract.
        IIncrementalToolCallParser? incrementalParser =
            state.ToolCallParser is not null && tools is { Length: > 0 }
                ? state.ToolCallParser.CreateIncremental()
                : null;

        await state.ExecuteAsync(async () =>
        {
            await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options, ct))
            {
                if (token.Text.Length > 0)
                {
                    completionTokens++;
                    var tokenLogprobs = token.Logprobs.HasValue
                        ? RequestConverter.ToLogprobsDto(token.Logprobs.Value)
                        : null;

                    if (incrementalParser is null)
                    {
                        // Legacy path: no tools attached — token.Text goes straight to delta.content.
                        var contentChunk = new ChatCompletionChunk
                        {
                            Id = requestId,
                            Model = modelId,
                            Choices = [new ChatChunkChoiceDto
                            {
                                Delta = new ChatDeltaDto { Content = token.Text },
                                Logprobs = tokenLogprobs,
                            }],
                        };
                        await WriteSseChunk(httpContext, contentChunk, ct);
                    }
                    else
                    {
                        // Tools attached — split this token into safe text and any tool-call fragments.
                        var parseResult = incrementalParser.AppendChunk(token.Text);

                        // tokenLogprobs describe the whole of token.Text. The parser may split it
                        // (prose + sentinel), hold part of it back as a possible sentinel prefix, or
                        // release text held back from an earlier token — in all of those the content
                        // chunk is no longer this token, and attaching its logprobs would report
                        // per-token probabilities against text they don't cover. Emit them only when
                        // the safe text is exactly the token and nothing was routed to a tool call.
                        bool logprobsCoverEmission =
                            parseResult.Fragments.Count == 0 &&
                            string.Equals(parseResult.SafeText, token.Text, StringComparison.Ordinal);

                        await EmitSplitChunksAsync(
                            httpContext, requestId, modelId, parseResult,
                            logprobsCoverEmission ? tokenLogprobs : null, ct);
                    }
                }

                if (token.FinishReason.HasValue)
                {
                    finishReason = token.FinishReason.Value;
                    timings = token.Timings;
                }
            }
        }, ct);

        // Drain the parser at end-of-stream: surface any held-back safe-text tail
        // and close any in-flight tool call.
        if (incrementalParser is not null)
        {
            var flushResult = incrementalParser.Flush();
            await EmitSplitChunksAsync(
                httpContext, requestId, modelId, flushResult, logprobs: null, ct);

            if (incrementalParser.HasEmittedAnyFragment)
                finishReason = FinishReason.ToolCalls;
        }

        // Final chunk with finish_reason. No re-parsing of accumulated content —
        // the incremental parser already emitted the full tool_calls stream.
        var finalDelta = new ChatDeltaDto();

        int promptTokens = timings?.PrefillTokenCount ?? 0;

        var finalChunk = new ChatCompletionChunk
        {
            Id = requestId,
            Model = modelId,
            Choices = [new ChatChunkChoiceDto
            {
                Delta = finalDelta,
                FinishReason = RequestConverter.ToFinishReasonString(finishReason),
            }],
            Usage = new UsageDto
            {
                PromptTokens = promptTokens,
                CompletionTokens = completionTokens,
                TotalTokens = promptTokens + completionTokens,
            },
            Timings = timings.HasValue ? new TimingsDto
            {
                PrefillTimeMs = timings.Value.PrefillTimeMs,
                DecodeTimeMs = timings.Value.DecodeTimeMs,
                SamplingTimeMs = timings.Value.SamplingTimeMs,
                PrefillTokensPerSec = timings.Value.PrefillTokensPerSec,
                DecodeTokensPerSec = timings.Value.DecodeTokensPerSec,
                PromptTokens = timings.Value.PrefillTokenCount,
                GeneratedTokens = timings.Value.DecodeTokenCount,
                CachedTokens = timings.Value.CachedTokenCount,
                SpeculativeDraftTokens = timings.Value.SpeculativeDraftTokens,
                SpeculativeAcceptedTokens = timings.Value.SpeculativeAcceptedTokens,
                SpeculativeAcceptanceRate = timings.Value.SpeculativeAcceptanceRate,
            } : null,
        };
        await WriteSseChunk(httpContext, finalChunk, ct);

        // [DONE] sentinel
        await httpContext.Response.WriteAsync("data: [DONE]\n\n", ct);
        await httpContext.Response.Body.FlushAsync(ct);
    }

    private static async Task WriteSseChunk(HttpContext ctx, ChatCompletionChunk chunk, CancellationToken ct)
    {
        await ctx.Response.WriteAsync("data: ", ct);
        await JsonSerializer.SerializeAsync(ctx.Response.Body, chunk, ServerJsonContext.Default.ChatCompletionChunk, ct);
        await ctx.Response.WriteAsync("\n\n", ct);
        await ctx.Response.Body.FlushAsync(ct);
    }

    /// <summary>
    /// Emits the safe-text and tool-call fragments produced by an
    /// <see cref="IIncrementalToolCallParser"/> step as separate SSE chunks.
    /// Logprobs (if any) attach to the safe-text chunk only — tool-call
    /// fragments do not carry logprobs.
    /// </summary>
    private static async Task EmitSplitChunksAsync(
        HttpContext httpContext,
        string requestId,
        string modelId,
        ToolCallParseResult result,
        LogprobsDto? logprobs,
        CancellationToken ct)
    {
        if (result.SafeText.Length > 0)
        {
            var contentChunk = new ChatCompletionChunk
            {
                Id = requestId,
                Model = modelId,
                Choices = [new ChatChunkChoiceDto
                {
                    Delta = new ChatDeltaDto { Content = result.SafeText },
                    Logprobs = logprobs,
                }],
            };
            await WriteSseChunk(httpContext, contentChunk, ct);
        }

        if (result.Fragments.Count > 0)
        {
            // Each fragment emits as its own SSE chunk so consumers see
            // delta.tool_calls[] arrive incrementally rather than as a batch.
            for (int i = 0; i < result.Fragments.Count; i++)
            {
                var dto = RequestConverter.ToToolCallDeltaDto(result.Fragments[i]);
                var toolChunk = new ChatCompletionChunk
                {
                    Id = requestId,
                    Model = modelId,
                    Choices = [new ChatChunkChoiceDto
                    {
                        Delta = new ChatDeltaDto { ToolCalls = [dto] },
                    }],
                };
                await WriteSseChunk(httpContext, toolChunk, ct);
            }
        }
    }
}
