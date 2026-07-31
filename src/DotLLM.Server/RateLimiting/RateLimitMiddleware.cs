using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using DotLLM.Server.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace DotLLM.Server.RateLimiting;

/// <summary>
/// ASP.NET middleware that admits or rejects requests based on the
/// configured per-API-key limits. Sits before the endpoints, so a
/// rejection short-circuits with HTTP 429 and a <c>Retry-After</c>
/// header — the inference pipeline is never invoked.
/// </summary>
/// <remarks>
/// <para>
/// <b>Scope.</b> The middleware only consults rate limits on requests
/// targeting the inference endpoints (<c>/v1/chat/completions</c>,
/// <c>/v1/completions</c>, <c>/v1/embeddings</c>). Health probes,
/// model-management, and the chat UI are intentionally unconstrained.
/// </para>
/// <para>
/// <b>Token estimation.</b> For accurate per-token accounting the
/// middleware would need to tokenize the request body up front. Today it
/// uses a cheap approximation (~1 token per 4 chars of payload) plus
/// <c>max_tokens</c> from the parsed JSON.
/// On admission the estimate is reserved; the endpoint reports the
/// authoritative count via <see cref="RateLimitLease.ReportActualTokens"/>
/// for true-up.
/// </para>
/// </remarks>
public sealed class RateLimitMiddleware
{
    /// <summary>
    /// <see cref="HttpContext.Items"/> key under which the admitted lease
    /// is stashed. Endpoints retrieve it to report actual completion
    /// tokens before disposal.
    /// </summary>
    public const string LeaseItemKey = "__DotLLM_RateLimitLease";

    private readonly RequestDelegate _next;
    private readonly RateLimitManager _manager;
    private readonly IApiKeyResolver _resolver;

    public RateLimitMiddleware(RequestDelegate next, RateLimitManager manager, IApiKeyResolver resolver)
    {
        _next = next;
        _manager = manager;
        _resolver = resolver;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        // Pass-through when disabled — keep zero overhead on the hot path
        // beyond a single bool check.
        if (!_manager.Config.Enabled)
        {
            await _next(context);
            return;
        }

        // Only meter inference endpoints. Everything else (UI assets,
        // /health, /v1/models, /v1/lora, /v1/cache/clear, ...) is free.
        if (!IsMeteredPath(context.Request.Path))
        {
            await _next(context);
            return;
        }

        var apiKey = _resolver.Resolve(context);
        int estimated = await EstimateTotalTokensAsync(context, _manager.Config).ConfigureAwait(false);

        var result = await _manager.TryAcquireAsync(apiKey, estimated, context.RequestAborted)
            .ConfigureAwait(false);

        if (!result.IsAcquired)
        {
            await WriteRejection(context, result).ConfigureAwait(false);
            return;
        }

        var lease = result.Lease!;
        context.Items[LeaseItemKey] = lease;
        try
        {
            await _next(context);
        }
        finally
        {
            // If the endpoint never reported actuals (e.g. exception path),
            // the lease still releases — the reservation is just consumed.
            lease.Dispose();
            context.Items.Remove(LeaseItemKey);
        }
    }

    /// <summary>
    /// Returns the admitted lease stashed on this request, or <c>null</c>
    /// when rate-limiting is disabled or the path is unmetered. Used by
    /// endpoints to call <see cref="RateLimitLease.ReportActualTokens"/>
    /// after generation completes.
    /// </summary>
    public static RateLimitLease? GetLease(HttpContext context) =>
        context.Items.TryGetValue(LeaseItemKey, out var v) ? v as RateLimitLease : null;

    private static bool IsMeteredPath(PathString path)
    {
        if (!path.HasValue) return false;
        var p = path.Value!;
        return p.StartsWith("/v1/chat/completions", StringComparison.OrdinalIgnoreCase)
            || p.StartsWith("/v1/completions", StringComparison.OrdinalIgnoreCase)
            || p.StartsWith("/v1/embeddings", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Best-effort estimate of total tokens this request will consume.
    /// Reads <c>max_tokens</c> from the parsed body when present, plus a
    /// cheap char-count approximation for the prompt/messages. We buffer
    /// the body so the downstream endpoint can re-read it.
    /// </summary>
    private static async ValueTask<int> EstimateTotalTokensAsync(HttpContext context, RateLimitConfig config)
    {
        var req = context.Request;
        if (!HttpMethods.IsPost(req.Method))
            return config.EstimatedCompletionTokensFallback;

        // Buffer body so model binding still sees it.
        req.EnableBuffering();
        long startPos = req.Body.CanSeek ? req.Body.Position : 0;
        int promptChars = 0;
        int? maxTokens = null;

        try
        {
            using var reader = new StreamReader(req.Body, Encoding.UTF8, detectEncodingFromByteOrderMarks: false, bufferSize: 8192, leaveOpen: true);
            // Limit to a reasonable size — past this we just fall back to the cap.
            const int maxScanChars = 256 * 1024;
            var sb = new StringBuilder();
            char[] buf = new char[4096];
            int read;
            while ((read = await reader.ReadAsync(buf.AsMemory(), context.RequestAborted).ConfigureAwait(false)) > 0)
            {
                sb.Append(buf, 0, read);
                if (sb.Length >= maxScanChars) break;
            }
            string body = sb.ToString();
            promptChars = body.Length;
            maxTokens = TryReadMaxTokens(body);
        }
        catch (Exception) // malformed body — defer to endpoint validator
        {
            // Conservative fallback.
            maxTokens = null;
        }
        finally
        {
            if (req.Body.CanSeek)
                req.Body.Position = startPos;
        }

        // Heuristic: ~4 bytes/token is a common approximation across BPE
        // tokenizers. Cheap and stable; trued up after generation.
        int promptEstimate = Math.Max(1, promptChars / 4);
        int completionEstimate = maxTokens ?? config.EstimatedCompletionTokensFallback;
        return promptEstimate + Math.Max(0, completionEstimate);
    }

    private static int? TryReadMaxTokens(string body)
    {
        try
        {
            using var doc = JsonDocument.Parse(body);
            if (doc.RootElement.ValueKind != JsonValueKind.Object) return null;
            if (doc.RootElement.TryGetProperty("max_tokens", out var mt) &&
                mt.ValueKind == JsonValueKind.Number &&
                mt.TryGetInt32(out var v) && v > 0)
            {
                return v;
            }
        }
        catch (JsonException) { }
        return null;
    }

    // The rejected API key is deliberately absent from the response body and
    // headers — it is a caller credential, not diagnostic output.
    private static async Task WriteRejection(HttpContext context, AcquireResult result)
    {
        context.Response.StatusCode = StatusCodes.Status429TooManyRequests;
        context.Response.Headers["Retry-After"] = result.RetryAfter.ToString(CultureInfo.InvariantCulture);
        context.Response.Headers["X-RateLimit-Limiter"] = result.Rejected.ToString();

        string reason = result.Rejected switch
        {
            LimiterKind.Requests => "requests-per-minute",
            LimiterKind.Tokens => "tokens-per-minute",
            LimiterKind.Concurrency => "max-concurrent",
            _ => "rate-limit",
        };
        var body = new ErrorResponse
        {
            Error = $"Rate limit exceeded ({reason}). Retry in {result.RetryAfter}s.",
        };
        context.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(
            context.Response.Body, body,
            ServerJsonContext.Default.ErrorResponse,
            context.RequestAborted).ConfigureAwait(false);
    }
}

/// <summary>Extension methods to wire the middleware into an ASP.NET pipeline.</summary>
public static class RateLimitMiddlewareExtensions
{
    /// <summary>
    /// Add the dotLLM rate-limit middleware to the request pipeline. Must
    /// be placed before <c>MapDotLLMEndpoints</c>.
    /// </summary>
    /// <remarks>
    /// The middleware is always wired, including when
    /// <see cref="RateLimitConfig.Enabled"/> is <c>false</c>: it short-circuits
    /// internally on a single bool check, so a disabled configuration costs one
    /// branch per request and can be toggled without rebuilding the pipeline.
    /// <paramref name="manager"/> and <paramref name="resolver"/> are required.
    /// </remarks>
    /// <exception cref="ArgumentNullException">
    /// <paramref name="manager"/> or <paramref name="resolver"/> is <c>null</c>.
    /// </exception>
    public static IApplicationBuilder UseDotLLMRateLimiting(this IApplicationBuilder app,
        RateLimitManager manager, IApiKeyResolver resolver)
    {
        ArgumentNullException.ThrowIfNull(manager);
        ArgumentNullException.ThrowIfNull(resolver);
        return app.UseMiddleware<RateLimitMiddleware>(manager, resolver);
    }
}
