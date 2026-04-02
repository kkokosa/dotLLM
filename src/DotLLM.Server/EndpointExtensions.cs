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
    public static WebApplication MapDotLLMEndpoints(this WebApplication app)
    {
        ChatCompletionEndpoint.Map(app);
        CompletionEndpoint.Map(app);
        ModelEndpoint.Map(app);
        TokenizeEndpoint.Map(app);
        HealthEndpoint.Map(app);
        return app;
    }
}
