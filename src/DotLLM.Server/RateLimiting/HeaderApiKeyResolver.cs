using Microsoft.AspNetCore.Http;
using Microsoft.Net.Http.Headers;

namespace DotLLM.Server.RateLimiting;

/// <summary>
/// Default <see cref="IApiKeyResolver"/>. Reads the API key from
/// <c>X-API-Key</c> first, then falls back to the bearer token in the
/// <c>Authorization</c> header. Returns <see cref="AnonymousKey"/> when
/// neither is present.
/// </summary>
/// <remarks>
/// dotLLM's server is a development tool — see <c>docs/SERVER.md § Security</c>.
/// This resolver only exists so rate-limit buckets can be partitioned per
/// caller; it does NOT perform any authentication. Host applications wiring
/// real auth (OAuth, JWT, mTLS) should swap in a custom resolver that
/// returns the authenticated principal's stable ID.
/// </remarks>
public sealed class HeaderApiKeyResolver : IApiKeyResolver
{
    /// <summary>Sentinel returned when no API key is present on the request.</summary>
    public const string AnonymousKey = "anonymous";

    /// <summary>Header name checked first. Mirrors OpenAI's own ergonomics.</summary>
    public const string ApiKeyHeader = "X-API-Key";

    /// <inheritdoc/>
    public string Resolve(HttpContext context)
    {
        // 1. X-API-Key header
        if (context.Request.Headers.TryGetValue(ApiKeyHeader, out var apiKeyValues))
        {
            var apiKey = apiKeyValues.ToString();
            if (!string.IsNullOrWhiteSpace(apiKey))
                return apiKey;
        }

        // 2. Authorization: Bearer <key>
        if (context.Request.Headers.TryGetValue(HeaderNames.Authorization, out var authValues))
        {
            var auth = authValues.ToString();
            const string bearerPrefix = "Bearer ";
            if (auth.StartsWith(bearerPrefix, StringComparison.OrdinalIgnoreCase))
            {
                var token = auth.AsSpan(bearerPrefix.Length).Trim();
                if (!token.IsEmpty)
                    return token.ToString();
            }
        }

        return AnonymousKey;
    }
}
