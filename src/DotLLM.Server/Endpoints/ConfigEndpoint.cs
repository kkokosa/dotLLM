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

            var d = state.SamplingDefaults;
            state.SamplingDefaults = d with
            {
                Temperature = root.TryGetProperty("temperature", out var temp) ? temp.GetSingle() : d.Temperature,
                TopP = root.TryGetProperty("top_p", out var topP) ? topP.GetSingle() : d.TopP,
                TopK = root.TryGetProperty("top_k", out var topK) ? topK.GetInt32() : d.TopK,
                MinP = root.TryGetProperty("min_p", out var minP) ? minP.GetSingle() : d.MinP,
                RepetitionPenalty = root.TryGetProperty("repetition_penalty", out var rep) ? rep.GetSingle() : d.RepetitionPenalty,
                MaxTokens = root.TryGetProperty("max_tokens", out var maxTok) ? maxTok.GetInt32() : d.MaxTokens,
                Seed = root.TryGetProperty("seed", out var seed) ? (seed.ValueKind == JsonValueKind.Null ? null : seed.GetInt32()) : d.Seed,
            };

            return Results.Ok(PropsEndpoint.ToDto(state.SamplingDefaults));
        });

        app.MapPost("/v1/cache/clear", (ServerState state) =>
        {
            state.PrefixCache?.Clear();
            return Results.Ok(new StatusResponse { Status = "cleared" });
        });
    }
}
