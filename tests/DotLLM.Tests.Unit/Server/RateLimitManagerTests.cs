using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using DotLLM.Server.RateLimiting;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for the cross-limiter <see cref="RateLimitManager"/> behaviour:
/// disabled pass-through, requests/min burst, tokens/min reservation, and
/// the concurrent-cap fast-fail. Long token-bucket refill windows are
/// intentionally avoided — those are integration concerns covered by
/// <see cref="RateLimitMiddlewareTests"/>.
/// </summary>
public class RateLimitManagerTests
{
    [Fact]
    public async Task Disabled_AdmitsEverything()
    {
        var cfg = new RateLimitConfig { Enabled = false };
        using var mgr = new RateLimitManager(cfg);

        for (int i = 0; i < 100; i++)
        {
            var result = await mgr.TryAcquireAsync("anyone", 9_999_999, default);
            Assert.True(result.IsAcquired);
            result.Lease!.Dispose();
        }
    }

    [Fact]
    public async Task UnknownKeyWithoutDefault_AdmitsAsPassThrough()
    {
        var cfg = new RateLimitConfig { Enabled = true };
        using var mgr = new RateLimitManager(cfg);
        var result = await mgr.TryAcquireAsync("ghost", 1000, default);
        Assert.True(result.IsAcquired);
        result.Lease!.Dispose();
    }

    [Fact]
    public async Task RequestsPerMinute_BurstExhaustsBudget_ReturnsRequestsRejection()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 3 },
        };
        using var mgr = new RateLimitManager(cfg);

        var leases = new List<RateLimitLease>();
        for (int i = 0; i < 3; i++)
        {
            var ok = await mgr.TryAcquireAsync("key", 0, default);
            Assert.True(ok.IsAcquired);
            leases.Add(ok.Lease!);
        }

        var rejected = await mgr.TryAcquireAsync("key", 0, default);
        Assert.False(rejected.IsAcquired);
        Assert.Equal(LimiterKind.Requests, rejected.Rejected);
        Assert.True(rejected.RetryAfter >= 1, $"RetryAfter should be >=1, got {rejected.RetryAfter}");

        foreach (var l in leases) l.Dispose();
    }

    [Fact]
    public async Task TokensPerMinute_OversizedRequest_ReturnsTokensRejection()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy
            {
                RequestsPerMinute = 100, // not the bottleneck
                TokensPerMinute = 100,
            },
        };
        using var mgr = new RateLimitManager(cfg);

        var ok = await mgr.TryAcquireAsync("key", 80, default);
        Assert.True(ok.IsAcquired);

        var rejected = await mgr.TryAcquireAsync("key", 80, default);
        Assert.False(rejected.IsAcquired);
        Assert.Equal(LimiterKind.Tokens, rejected.Rejected);
        ok.Lease!.Dispose();
    }

    [Fact]
    public async Task MaxConcurrent_HoldsSecondRequest_UntilFirstReleases()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy
            {
                MaxConcurrent = 1,
                QueueTimeout = System.TimeSpan.FromSeconds(5),
            },
        };
        using var mgr = new RateLimitManager(cfg);

        var first = await mgr.TryAcquireAsync("key", 0, default);
        Assert.True(first.IsAcquired);

        var pending = mgr.TryAcquireAsync("key", 0, default).AsTask();
        await Task.Yield();
        Assert.False(pending.IsCompleted, "Second admission must wait while concurrency==1");

        first.Lease!.Dispose();

        var second = await pending;
        Assert.True(second.IsAcquired);
        second.Lease!.Dispose();
    }

    [Fact]
    public async Task MaxConcurrent_QueueTimeoutExpires_ReturnsConcurrencyRejection()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy
            {
                MaxConcurrent = 1,
                QueueTimeout = System.TimeSpan.FromMilliseconds(80),
            },
        };
        using var mgr = new RateLimitManager(cfg);

        var holder = await mgr.TryAcquireAsync("key", 0, default);
        Assert.True(holder.IsAcquired);

        var rejected = await mgr.TryAcquireAsync("key", 0, default);
        Assert.False(rejected.IsAcquired);
        Assert.Equal(LimiterKind.Concurrency, rejected.Rejected);
        Assert.True(rejected.RetryAfter >= 0);

        holder.Lease!.Dispose();
    }

    [Fact]
    public async Task PerKey_LimitsAreIsolated()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 1 },
        };
        using var mgr = new RateLimitManager(cfg);

        var a1 = await mgr.TryAcquireAsync("alice", 0, default);
        Assert.True(a1.IsAcquired);

        var a2 = await mgr.TryAcquireAsync("alice", 0, default);
        Assert.False(a2.IsAcquired);

        // Bob has his own bucket — Alice's exhaustion must not bleed across.
        var b1 = await mgr.TryAcquireAsync("bob", 0, default);
        Assert.True(b1.IsAcquired);

        a1.Lease!.Dispose();
        b1.Lease!.Dispose();
    }

    [Fact]
    public async Task PerKey_OverridePolicyApplies()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 1 },
            ApiKeys = new Dictionary<string, RateLimitPolicy>
            {
                ["premium"] = new RateLimitPolicy { RequestsPerMinute = 5 },
            },
        };
        using var mgr = new RateLimitManager(cfg);

        // Default bucket exhausts after 1.
        var d1 = await mgr.TryAcquireAsync("free", 0, default);
        Assert.True(d1.IsAcquired);
        var d2 = await mgr.TryAcquireAsync("free", 0, default);
        Assert.False(d2.IsAcquired);

        // Premium bucket gets 5 in a row.
        var leases = new List<RateLimitLease>();
        for (int i = 0; i < 5; i++)
        {
            var ok = await mgr.TryAcquireAsync("premium", 0, default);
            Assert.True(ok.IsAcquired);
            leases.Add(ok.Lease!);
        }
        var blocked = await mgr.TryAcquireAsync("premium", 0, default);
        Assert.False(blocked.IsAcquired);
        Assert.Equal(LimiterKind.Requests, blocked.Rejected);

        d1.Lease!.Dispose();
        foreach (var l in leases) l.Dispose();
    }

    /// <summary>
    /// Regression: the bucket used to replenish <c>max(1, permitsPerMinute/60)</c>
    /// permits every second, so a 2/min policy actually refilled at 60/min. The
    /// replenishment period is now derived from the batch size, giving an exact
    /// per-minute rate — a 2/min key gets one permit back every 30 s, so nothing
    /// is admitted within a second of exhausting the budget.
    /// </summary>
    [Fact]
    public async Task RequestsPerMinute_SmallLimit_DoesNotRefillWithinASecond()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 2 },
        };
        using var mgr = new RateLimitManager(cfg);

        var leases = new List<RateLimitLease>();
        for (int i = 0; i < 2; i++)
        {
            var ok = await mgr.TryAcquireAsync("key", 0, default);
            Assert.True(ok.IsAcquired);
            leases.Add(ok.Lease!);
        }

        Assert.False((await mgr.TryAcquireAsync("key", 0, default)).IsAcquired);

        // Comfortably past the old 1-second replenishment tick.
        await Task.Delay(1200);

        var afterWait = await mgr.TryAcquireAsync("key", 0, default);
        Assert.False(afterWait.IsAcquired);
        Assert.Equal(LimiterKind.Requests, afterWait.Rejected);

        foreach (var l in leases) l.Dispose();
    }

    [Fact]
    public async Task ReportActualTokens_DoesNotThrow_WhenLeaseDisposed()
    {
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { TokensPerMinute = 1000 },
        };
        using var mgr = new RateLimitManager(cfg);
        var result = await mgr.TryAcquireAsync("k", 500, default);
        Assert.True(result.IsAcquired);

        // Generated only 50 of the 500 reserved.
        result.Lease!.ReportActualTokens(50);
        result.Lease!.Dispose();
        // No exception expected on double dispose.
        result.Lease!.Dispose();
    }
}
