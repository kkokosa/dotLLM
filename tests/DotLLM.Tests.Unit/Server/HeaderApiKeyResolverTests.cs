using DotLLM.Server.RateLimiting;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for the minimal <see cref="HeaderApiKeyResolver"/>. Mirrors the
/// fallback ordering documented in <c>docs/SERVER.md § Rate Limiting</c>.
/// </summary>
public class HeaderApiKeyResolverTests
{
    [Fact]
    public void NoHeader_ReturnsAnonymous()
    {
        var ctx = new DefaultHttpContext();
        var resolver = new HeaderApiKeyResolver();
        Assert.Equal(HeaderApiKeyResolver.AnonymousKey, resolver.Resolve(ctx));
    }

    [Fact]
    public void XApiKeyHeader_TakesPrecedence()
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Headers[HeaderApiKeyResolver.ApiKeyHeader] = "primary-key";
        ctx.Request.Headers["Authorization"] = "Bearer different-key";

        var resolver = new HeaderApiKeyResolver();
        Assert.Equal("primary-key", resolver.Resolve(ctx));
    }

    [Fact]
    public void BearerToken_FallbackWhenNoXApiKey()
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Headers["Authorization"] = "Bearer my-token";

        var resolver = new HeaderApiKeyResolver();
        Assert.Equal("my-token", resolver.Resolve(ctx));
    }

    [Fact]
    public void EmptyXApiKey_FallsThroughToBearer()
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Headers[HeaderApiKeyResolver.ApiKeyHeader] = "  ";
        ctx.Request.Headers["Authorization"] = "Bearer fallback";

        var resolver = new HeaderApiKeyResolver();
        Assert.Equal("fallback", resolver.Resolve(ctx));
    }

    [Fact]
    public void BearerPrefix_IsCaseInsensitive()
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Headers["Authorization"] = "bearer lowercase-token";

        var resolver = new HeaderApiKeyResolver();
        Assert.Equal("lowercase-token", resolver.Resolve(ctx));
    }
}
