using System.Text.Json;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET/POST /v1/config — read or update mutable sampling defaults.
/// </summary>
public static class ConfigEndpoint
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/v1/config", (ServerState state) => PropsEndpoint.ToDto(state.SamplingDefaults));

        app.MapPost("/v1/config", async (HttpContext httpContext, ServerState state) =>
        {
            // Partial update: only fields present in the JSON body are applied.
            using var doc = await JsonDocument.ParseAsync(httpContext.Request.Body, cancellationToken: httpContext.RequestAborted);
            var root = doc.RootElement;

            var defaults = state.SamplingDefaults;
            if (root.TryGetProperty("temperature", out var temp))
                defaults.Temperature = temp.GetSingle();
            if (root.TryGetProperty("top_p", out var topP))
                defaults.TopP = topP.GetSingle();
            if (root.TryGetProperty("top_k", out var topK))
                defaults.TopK = topK.GetInt32();
            if (root.TryGetProperty("min_p", out var minP))
                defaults.MinP = minP.GetSingle();
            if (root.TryGetProperty("repetition_penalty", out var rep))
                defaults.RepetitionPenalty = rep.GetSingle();
            if (root.TryGetProperty("max_tokens", out var maxTok))
                defaults.MaxTokens = maxTok.GetInt32();
            if (root.TryGetProperty("seed", out var seed))
                defaults.Seed = seed.ValueKind == JsonValueKind.Null ? null : seed.GetInt32();

            return Results.Ok(PropsEndpoint.ToDto(defaults));
        });
    }
}
