using System.Text;
using System.Text.Json;
using DotLLM.Engine;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// POST /v1/completions — OpenAI-compatible raw completion endpoint (no chat template).
/// </summary>
public static class CompletionEndpoint
{

    public static void Map(WebApplication app) =>
        app.MapPost("/v1/completions", HandleAsync);

    private static async Task HandleAsync(
        CompletionRequest request,
        ServerState state,
        HttpContext httpContext)
    {
        if (!state.IsReady || state.Generator is null)
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
        var validationError = RequestValidator.ValidateCompletionRequest(request);
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

        // Validate prompt length against model context
        int maxTokens = request.MaxTokens ?? state.SamplingDefaults.MaxTokens;
        var promptError = RequestValidator.ValidatePromptLength(
            request.Prompt, state.Tokenizer!, state.Config!.MaxSequenceLength,
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

        var options = RequestConverter.ToInferenceOptions(request,
            state.SamplingDefaults,
            new DotLLM.Core.Configuration.ThreadingConfig(
                state.Options.Threads, state.Options.DecodeThreads));
        options = options with { MaxTokens = effectiveMaxTokens };

        if (request.Stream)
            await HandleStreamingAsync(generator, state, httpContext, request.Prompt, options,
                requestId, modelId, ct);
        else
            await HandleNonStreamingAsync(generator, state, httpContext, request.Prompt, options,
                requestId, modelId, ct);
    }

    private static async Task HandleNonStreamingAsync(
        TextGenerator generator, ServerState state, HttpContext httpContext,
        string prompt, DotLLM.Core.Configuration.InferenceOptions options,
        string requestId, string modelId, CancellationToken ct)
    {
        InferenceResponse? result = null;
        await state.ExecuteAsync(async () =>
        {
            result = generator.Generate(prompt, options);
        }, ct);

        var logprobsDto = result!.Logprobs is { Length: > 0 }
            ? RequestConverter.ToLogprobsDto(result.Logprobs)
            : null;

        var response = new CompletionResponse
        {
            Id = requestId,
            Model = modelId,
            Choices = [new CompletionChoiceDto
            {
                Index = 0,
                Text = result.Text,
                Logprobs = logprobsDto,
                FinishReason = RequestConverter.ToFinishReasonString(result.FinishReason),
            }],
            Usage = new UsageDto
            {
                PromptTokens = result.PromptTokenCount,
                CompletionTokens = result.GeneratedTokenCount,
                TotalTokens = result.PromptTokenCount + result.GeneratedTokenCount,
            },
        };

        httpContext.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(httpContext.Response.Body, response, ServerJsonContext.Default.CompletionResponse, ct);
    }

    private static async Task HandleStreamingAsync(
        TextGenerator generator, ServerState state, HttpContext httpContext,
        string prompt, DotLLM.Core.Configuration.InferenceOptions options,
        string requestId, string modelId, CancellationToken ct)
    {
        httpContext.Response.ContentType = "text/event-stream";
        httpContext.Response.Headers.CacheControl = "no-cache";
        httpContext.Response.Headers.Connection = "keep-alive";

        await state.ExecuteAsync(async () =>
        {
            await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options, ct))
            {
                var tokenLogprobs = token.Logprobs.HasValue
                    ? RequestConverter.ToLogprobsDto(token.Logprobs.Value)
                    : null;
                var chunk = new CompletionChunk
                {
                    Id = requestId,
                    Model = modelId,
                    Choices = [new CompletionChunkChoiceDto
                    {
                        Text = token.Text,
                        Logprobs = tokenLogprobs,
                        FinishReason = token.FinishReason.HasValue
                            ? RequestConverter.ToFinishReasonString(token.FinishReason.Value)
                            : null,
                    }],
                };
                await httpContext.Response.WriteAsync("data: ", ct);
                await JsonSerializer.SerializeAsync(httpContext.Response.Body, chunk, ServerJsonContext.Default.CompletionChunk, ct);
                await httpContext.Response.WriteAsync("\n\n", ct);
                await httpContext.Response.Body.FlushAsync(ct);
            }
        }, ct);

        await httpContext.Response.WriteAsync("data: [DONE]\n\n", ct);
        await httpContext.Response.Body.FlushAsync(ct);
    }
}
