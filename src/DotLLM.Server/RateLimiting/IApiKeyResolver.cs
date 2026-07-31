using Microsoft.AspNetCore.Http;

namespace DotLLM.Server.RateLimiting;

/// <summary>
/// Resolves the API key (and associated rate-limit context) for an incoming
/// HTTP request. dotLLM ships a minimal <see cref="HeaderApiKeyResolver"/>
/// that reads <c>X-API-Key</c> / <c>Authorization: Bearer &lt;key&gt;</c>;
/// host applications can swap in a real authentication backend.
/// </summary>
/// <remarks>
/// This is intentionally separate from any authn/authz surface — its sole
/// purpose is to bucket requests for rate-limiting. Returning a stable key
/// for unauthenticated requests (e.g. <c>"anonymous"</c>) is the recommended
/// behaviour so anonymous traffic still shares a fair bucket.
/// </remarks>
public interface IApiKeyResolver
{
    /// <summary>
    /// Returns a stable key identifying the caller for rate-limit bucketing.
    /// Must never return <c>null</c> — return a sentinel (e.g.
    /// <c>"anonymous"</c>) for unauthenticated traffic.
    /// </summary>
    string Resolve(HttpContext context);

    /// <summary>
    /// Resolves the <see cref="RateLimitPolicy"/> for the given key from the
    /// active configuration. Equivalent to <c>config.PolicyFor(key)</c> —
    /// exposed on the resolver so custom implementations can override
    /// (e.g. dynamic per-tenant lookup).
    /// </summary>
    RateLimitPolicy? GetRateLimitContext(string apiKey, RateLimitConfig config) =>
        config.PolicyFor(apiKey);
}
