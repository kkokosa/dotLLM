using Microsoft.Extensions.DependencyInjection;

namespace DotLLM.Server;

/// <summary>
/// Dependency injection helpers for hosting dotLLM inside a consumer ASP.NET Core app.
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Registers the services required by <see cref="EndpointExtensions.MapDotLLMEndpoints"/>:
    /// the loaded <see cref="ServerState"/> as a singleton, and the source-generated JSON type
    /// info resolver for the OpenAI-compatible request/response DTOs.
    /// </summary>
    /// <remarks>
    /// Call this on your <see cref="IServiceCollection"/> <em>before</em> <c>builder.Build()</c>
    /// so the endpoints mapped by <see cref="EndpointExtensions.MapDotLLMEndpoints"/> can resolve
    /// their <see cref="ServerState"/> parameter from DI.
    /// </remarks>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="state">A populated server state. Obtain via <see cref="ServerStartup.LoadModel"/>.</param>
    /// <returns>The same service collection for chaining.</returns>
    public static IServiceCollection AddDotLLM(this IServiceCollection services, ServerState state)
    {
        services.AddSingleton(state);
        services.ConfigureHttpJsonOptions(options =>
            options.SerializerOptions.TypeInfoResolverChain.Insert(0, ServerJsonContext.Default));
        return services;
    }
}
