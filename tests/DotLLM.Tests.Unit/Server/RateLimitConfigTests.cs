using System.Collections.Generic;
using DotLLM.Server.RateLimiting;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for <see cref="RateLimitConfig"/> defaults, policy lookup, and
/// the disabled/empty pass-through path.
/// </summary>
public class RateLimitConfigTests
{
    [Fact]
    public void Defaults_IsDisabledWithNoPolicies()
    {
        var cfg = new RateLimitConfig();
        Assert.False(cfg.Enabled);
        Assert.Null(cfg.DefaultPolicy);
        Assert.Empty(cfg.ApiKeys);
        Assert.Equal(256, cfg.EstimatedCompletionTokensFallback);
    }

    [Fact]
    public void PolicyFor_KnownKey_ReturnsOverride()
    {
        var premium = new RateLimitPolicy { RequestsPerMinute = 600, Priority = RequestPriority.High };
        var cfg = new RateLimitConfig
        {
            Enabled = true,
            DefaultPolicy = new RateLimitPolicy { RequestsPerMinute = 60 },
            ApiKeys = new Dictionary<string, RateLimitPolicy>
            {
                ["key-premium"] = premium,
            },
        };
        Assert.Same(premium, cfg.PolicyFor("key-premium"));
    }

    [Fact]
    public void PolicyFor_UnknownKey_FallsBackToDefault()
    {
        var def = new RateLimitPolicy { RequestsPerMinute = 60 };
        var cfg = new RateLimitConfig { Enabled = true, DefaultPolicy = def };
        Assert.Same(def, cfg.PolicyFor("does-not-exist"));
    }

    [Fact]
    public void PolicyFor_NoDefault_NoOverride_ReturnsNull()
    {
        var cfg = new RateLimitConfig { Enabled = true };
        Assert.Null(cfg.PolicyFor("anything"));
    }

    [Fact]
    public void ApiKeyLookup_IsCaseSensitive()
    {
        var p = new RateLimitPolicy { RequestsPerMinute = 60 };
        var cfg = new RateLimitConfig
        {
            ApiKeys = new Dictionary<string, RateLimitPolicy>(System.StringComparer.Ordinal)
            {
                ["MixedCaseKey"] = p,
            },
        };
        Assert.Same(p, cfg.PolicyFor("MixedCaseKey"));
        Assert.Null(cfg.PolicyFor("mixedcasekey"));
    }

    [Fact]
    public void Policy_DefaultsToNormalPriorityAndFiveSecondQueueTimeout()
    {
        var p = new RateLimitPolicy();
        Assert.Equal(RequestPriority.Normal, p.Priority);
        Assert.Equal(System.TimeSpan.FromSeconds(5), p.QueueTimeout);
    }
}
