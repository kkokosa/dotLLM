namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET /health and /ready — health and readiness probes.
/// </summary>
public static class HealthEndpoint
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

        app.MapGet("/ready", (ServerState state) =>
            state.IsReady
                ? Results.Ok(new { status = "ready" })
                : Results.StatusCode(503));
    }
}
