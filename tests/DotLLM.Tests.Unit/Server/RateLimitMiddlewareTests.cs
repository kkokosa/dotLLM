using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using DotLLM.Server.RateLimiting;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Integration-style tests for <see cref="RateLimitMiddleware"/>. Each test
/// constructs the middleware with a small in-memory <see cref="RateLimitManager"/>
/// + the default <see cref="HeaderApiKeyResolver"/>, then drives requests
/// through a <see cref="DefaultHttpContext"/>. Asserts cover:
///   <list type="bullet">
///   <item><description>burst → 429 with <c>Retry-After</c> header</description></item>
///   <item><description>unmetered paths (e.g. <c>/health</c>) bypass the limiter</description></item>
///   <item><description>disabled config is pass-through</description></item>
///   <item><description>per-request lease is stashed on <see cref="HttpContext.Items"/></description></item>
///   </list>
/// </summary>
public class RateLimitMiddlewareTests
{
    private static RateLimitMiddleware Build(RateLimitConfig cfg, out RateLimitManager manager)
    {
        manager = new RateLimitManager(cfg);
        var resolver = new HeaderApiKeyResolver();
        return new RateLimitMiddleware(_ => Task.CompletedTask, manager, resolver);
    }

    private static DefaultHttpContext NewRequest(string path, string? apiKey, string? body = null)
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Method = HttpMethods.Post;
        ctx.Request.Path = path;
        if (apiKey is not null)
            ctx.Request.Headers[HeaderApiKeyResolver.ApiKeyHeader] = apiKey;
        if (body is not null)
        {
            var bytes = Encoding.UTF8.GetBytes(body);
            ctx.Request.ContentType = "application/json";
            ctx.Request.ContentLength = bytes.Length;
            ctx.Request.Body = new MemoryStream(bytes);
        }
        ctx.Response.Body = new MemoryStream();
        return ctx;
    }

    [Fact]
    public async Task DisabledConfig_PassesThrough_NoLeaseStashed()
    {
        var mw = Build(new RateLimitConfig { Enabled = false }, out var mgr);
        using (mgr)
        {
            var ctx = NewRequest("/v1/chat/completions", "alice", "{\"messages\":[]}");
            await mw.InvokeAsync(ctx);
            Assert.Equal(200, ctx.Response.StatusCode);
            Assert.False(ctx.Items.ContainsKey(RateLimitMiddleware.LeaseItemKey));
        }
    }

    [Fact]
    public async Task UnmeteredPath_BypassesLimiter()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 1 },
        };
        var mw = Build(cfg, out var mgr);
        using (mgr)
        {
            for (int i = 0; i < 10; i++)
            {
                var ctx = NewRequest("/v1/models", "alice");
                await mw.InvokeAsync(ctx);
                Assert.Equal(200, ctx.Response.StatusCode);
            }
        }
    }

    [Fact]
    public async Task BurstExceedsRequestsPerMinute_ReturnsHttp429WithRetryAfter()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 2 },
        };
        var mw = Build(cfg, out var mgr);
        using (mgr)
        {
            // First two requests succeed.
            for (int i = 0; i < 2; i++)
            {
                var ctx = NewRequest("/v1/chat/completions", "alice", "{}");
                await mw.InvokeAsync(ctx);
                Assert.Equal(200, ctx.Response.StatusCode);
            }

            // Third is rejected.
            var rejected = NewRequest("/v1/chat/completions", "alice", "{}");
            await mw.InvokeAsync(rejected);

            Assert.Equal(StatusCodes.Status429TooManyRequests, rejected.Response.StatusCode);
            Assert.True(rejected.Response.Headers.ContainsKey("Retry-After"),
                "Retry-After header is required on 429 (Step 38 acceptance criteria).");
            Assert.True(int.TryParse(rejected.Response.Headers["Retry-After"]!, out var ra) && ra > 0,
                "Retry-After must be a positive integer seconds value.");
            Assert.Equal("Requests", rejected.Response.Headers["X-RateLimit-Limiter"].ToString());

            // Body is a JSON ErrorResponse with the rejected limiter named.
            rejected.Response.Body.Position = 0;
            using var doc = await JsonDocument.ParseAsync(rejected.Response.Body);
            Assert.True(doc.RootElement.TryGetProperty("error", out var errProp));
            Assert.Contains("requests-per-minute", errProp.GetString()!);
        }
    }

    [Fact]
    public async Task PerKeyIsolation_RejectionDoesNotAffectOtherKey()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 1 },
        };
        var mw = Build(cfg, out var mgr);
        using (mgr)
        {
            var aliceOk = NewRequest("/v1/chat/completions", "alice", "{}");
            await mw.InvokeAsync(aliceOk);
            Assert.Equal(200, aliceOk.Response.StatusCode);

            var aliceBlocked = NewRequest("/v1/chat/completions", "alice", "{}");
            await mw.InvokeAsync(aliceBlocked);
            Assert.Equal(429, aliceBlocked.Response.StatusCode);

            var bobOk = NewRequest("/v1/chat/completions", "bob", "{}");
            await mw.InvokeAsync(bobOk);
            Assert.Equal(200, bobOk.Response.StatusCode);
        }
    }

    [Fact]
    public async Task LeaseStashedOnHttpContext_AndDisposedAfterDownstream()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 60, MaxConcurrent = 2 },
        };
        var mgr = new RateLimitManager(cfg);
        using (mgr)
        {
            RateLimitLease? observed = null;
            var mw = new RateLimitMiddleware(ctx =>
            {
                observed = RateLimitMiddleware.GetLease(ctx);
                observed?.ReportActualTokens(42);
                return Task.CompletedTask;
            }, mgr, new HeaderApiKeyResolver());

            var ctx = NewRequest("/v1/chat/completions", "alice", "{}");
            await mw.InvokeAsync(ctx);

            Assert.NotNull(observed);
            Assert.False(ctx.Items.ContainsKey(RateLimitMiddleware.LeaseItemKey),
                "Lease must be removed from HttpContext.Items after the request completes.");
        }
    }

    [Fact]
    public async Task DownstreamException_ReleasesLease()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { MaxConcurrent = 1, QueueTimeout = TimeSpan.FromMilliseconds(50) },
        };
        var mgr = new RateLimitManager(cfg);
        using (mgr)
        {
            var mw = new RateLimitMiddleware(_ => throw new InvalidOperationException("boom"),
                mgr, new HeaderApiKeyResolver());

            var ctx = NewRequest("/v1/chat/completions", "alice", "{}");
            await Assert.ThrowsAsync<InvalidOperationException>(() => mw.InvokeAsync(ctx));

            // After the exception the concurrency slot must be free again —
            // otherwise the next call would hang for QueueTimeout and return 429.
            var nextMw = new RateLimitMiddleware(_ => Task.CompletedTask, mgr, new HeaderApiKeyResolver());
            var next = NewRequest("/v1/chat/completions", "alice", "{}");
            await nextMw.InvokeAsync(next);
            Assert.Equal(200, next.Response.StatusCode);
        }
    }

    [Fact]
    public async Task PriorityWaiter_JumpsAheadOnConcurrencyContest()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            ApiKeys = new Dictionary<string, RateLimitPolicy>
            {
                ["low"] = new RateLimitPolicy { MaxConcurrent = 1, Priority = RequestPriority.Low, QueueTimeout = TimeSpan.FromSeconds(5) },
                ["high"] = new RateLimitPolicy { MaxConcurrent = 1, Priority = RequestPriority.High, QueueTimeout = TimeSpan.FromSeconds(5) },
                ["holder"] = new RateLimitPolicy { MaxConcurrent = 1, Priority = RequestPriority.Normal, QueueTimeout = TimeSpan.FromSeconds(5) },
            },
        };

        // The middleware buckets per-key, so to observe priority ordering we
        // exercise the gate directly with a shared key. (The middleware only
        // gives a priority advantage *within* the same key bucket because
        // each key has its own gate. Cross-key priority is handled by the
        // scheduler, which is out of scope for Step 38.)
        using var gate = new PriorityConcurrencyGate(maxConcurrent: 1);
        var held = (await gate.AcquireAsync(RequestPriority.Normal, TimeSpan.FromSeconds(1), default))!;

        var lowTask = gate.AcquireAsync(RequestPriority.Low, TimeSpan.FromSeconds(5), default).AsTask();
        await WaitForQueueLength(gate, 1);
        var highTask = gate.AcquireAsync(RequestPriority.High, TimeSpan.FromSeconds(5), default).AsTask();
        await WaitForQueueLength(gate, 2);

        held.Dispose();

        var highLease = await highTask;
        Assert.NotNull(highLease);
        Assert.False(lowTask.IsCompleted);
        highLease!.Dispose();
        (await lowTask)!.Dispose();
    }

    private static async Task WaitForQueueLength(PriorityConcurrencyGate gate, int expected)
    {
        var deadline = DateTime.UtcNow.AddSeconds(2);
        while (gate.QueueLength != expected)
        {
            if (DateTime.UtcNow >= deadline)
                throw new TimeoutException($"Expected QueueLength={expected}, observed {gate.QueueLength}");
            await Task.Yield();
        }
    }
}
