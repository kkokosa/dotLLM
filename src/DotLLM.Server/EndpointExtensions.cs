using DotLLM.Server.Endpoints;

namespace DotLLM.Server;

/// <summary>
/// Extension methods for registering all dotLLM API endpoints.
/// </summary>
public static class EndpointExtensions
{
    /// <summary>
    /// Maps all dotLLM OpenAI-compatible API endpoints.
    /// </summary>
    /// <param name="app">The web application.</param>
    /// <param name="serveUi">When true, also serves the embedded web chat UI at <c>GET /</c>.</param>
    public static WebApplication MapDotLLMEndpoints(this WebApplication app, bool serveUi = false)
    {
        ChatCompletionEndpoint.Map(app);
        CompletionEndpoint.Map(app);
        ModelEndpoint.Map(app);
        TokenizeEndpoint.Map(app);
        HealthEndpoint.Map(app);
        PropsEndpoint.Map(app);
        ConfigEndpoint.Map(app);
        ModelManagementEndpoint.Map(app);

        if (serveUi)
            WebUIEndpoint.Map(app);

        return app;
    }
}
