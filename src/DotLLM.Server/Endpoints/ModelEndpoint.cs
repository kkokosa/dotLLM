using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET /v1/models — list loaded models.
/// </summary>
public static class ModelEndpoint
{
    public static void Map(WebApplication app) =>
        app.MapGet("/v1/models", (ServerState state) => new ModelListResponse
        {
            Data =
            [
                new ModelInfoDto
                {
                    Id = state.Options.ModelId,
                    Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                }
            ],
        });
}
