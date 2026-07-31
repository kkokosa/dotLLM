namespace DotLLM.Server.Endpoints;

/// <summary>
/// Shared response setup for the server's Server-Sent Events (SSE) streaming endpoints.
/// </summary>
internal static class SseResponse
{
    /// <summary>
    /// Applies the response headers an SSE stream needs: the <c>text/event-stream</c> content type
    /// and <c>Cache-Control: no-cache</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A <c>Connection: keep-alive</c> header is deliberately <b>not</b> set. <c>Connection</c> is a
    /// hop-by-hop, connection-specific header field: it is prohibited over HTTP/2 and HTTP/3
    /// (RFC 9113 §8.2.2 — an endpoint MUST treat a message containing connection-specific header
    /// fields as malformed), and it is redundant over HTTP/1.1, where persistent connections are
    /// already the default. Kestrel filters it on HTTP/2, but a strict intermediary in the request
    /// path need not be as forgiving. Please do not re-add it.
    /// </para>
    /// </remarks>
    /// <param name="httpContext">The HTTP context whose response is being turned into an SSE stream.</param>
    internal static void ApplyHeaders(HttpContext httpContext)
    {
        httpContext.Response.ContentType = "text/event-stream";
        httpContext.Response.Headers.CacheControl = "no-cache";
    }
}
