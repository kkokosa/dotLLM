using System.Text;
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

        // Build inference options
        var stopSequences = CommonStopSequences;
        var options = RequestConverter.ToInferenceOptions(request, stopSequences,
            new DotLLM.Core.Configuration.ThreadingConfig(
                state.Options.Threads, state.Options.DecodeThreads));

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

        var response = new ChatCompletionResponse
        {
            Id = requestId,
            Model = modelId,
            Choices = [new ChatChoiceDto
            {
                Index = 0,
                Message = message,
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

        var sb = new StringBuilder();
        FinishReason finishReason = FinishReason.Length;
        InferenceTimings? timings = null;
        int completionTokens = 0;

        await state.ExecuteAsync(async () =>
        {
            await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options, ct))
            {
                if (token.Text.Length > 0)
                {
                    completionTokens++;
                    sb.Append(token.Text);
                    var contentChunk = new ChatCompletionChunk
                    {
                        Id = requestId,
                        Model = modelId,
                        Choices = [new ChatChunkChoiceDto
                        {
                            Delta = new ChatDeltaDto { Content = token.Text },
                        }],
                    };
                    await WriteSseChunk(httpContext, contentChunk, ct);
                }

                if (token.FinishReason.HasValue)
                {
                    finishReason = token.FinishReason.Value;
                    timings = token.Timings;
                }
            }
        }, ct);

        // Detect tool calls in accumulated text
        string text = sb.ToString();
        ToolCall[]? toolCalls = null;
        if (state.ToolCallParser is not null && tools is { Length: > 0 })
        {
            toolCalls = state.ToolCallParser.TryParse(text);
            if (toolCalls is { Length: > 0 })
                finishReason = FinishReason.ToolCalls;
        }

        // Final chunk with finish_reason
        var finalDelta = toolCalls is { Length: > 0 }
            ? new ChatDeltaDto { ToolCalls = RequestConverter.ToToolCallDtos(toolCalls) }
            : new ChatDeltaDto();

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
            } : null,
            Prompt = prompt,
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
}
