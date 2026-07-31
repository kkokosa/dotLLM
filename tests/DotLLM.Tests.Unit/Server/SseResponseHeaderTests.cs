using DotLLM.Server.Endpoints;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Guards the response headers of the SSE streaming endpoints. Both OpenAI-compatible streaming
/// endpoints (<c>/v1/chat/completions</c> and <c>/v1/completions</c>) configure their response
/// through <see cref="SseResponse.ApplyHeaders"/>, so asserting on that one place covers both.
/// </summary>
public sealed class SseResponseHeaderTests
{
    [Fact]
    public void ApplyHeaders_SetsTheHeadersSseActuallyNeeds()
    {
        var ctx = new DefaultHttpContext();

        SseResponse.ApplyHeaders(ctx);

        Assert.Equal("text/event-stream", ctx.Response.ContentType);
        Assert.Equal("no-cache", ctx.Response.Headers.CacheControl.ToString());
    }

    /// <summary>
    /// Regression guard for #422: <c>Connection</c> is a hop-by-hop, connection-specific header
    /// field. RFC 9113 §8.2.2 forbids it over HTTP/2 (and HTTP/3), and it is redundant over
    /// HTTP/1.1 where persistent connections are the default. It must never be emitted.
    /// </summary>
    [Fact]
    public void ApplyHeaders_DoesNotSetConnectionHeader()
    {
        var ctx = new DefaultHttpContext();

        SseResponse.ApplyHeaders(ctx);

        Assert.False(ctx.Response.Headers.ContainsKey("Connection"));
    }
}
