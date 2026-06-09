using DotLLM.Core.Lora;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// LoRA adapter administration endpoints:
/// <list type="bullet">
///   <item><c>GET    /v1/lora</c> — list registered adapter names (always available).</item>
///   <item><c>POST   /v1/lora/load</c> — register a new adapter (gated by <c>Server:AllowLoraAdminApi</c>).</item>
///   <item><c>DELETE /v1/lora/{name}</c> — unload an adapter (gated by <c>Server:AllowLoraAdminApi</c>).</item>
/// </list>
/// The write endpoints are disabled by default — operators must opt-in via
/// <see cref="ServerOptions.AllowLoraAdminApi"/> to expose them. This matches
/// the existing pattern for any state-mutating admin surface.
/// </summary>
public static class LoraEndpoints
{
    public static void Map(WebApplication app)
    {
        // ── GET /v1/lora — read-only list (always available) ──
        app.MapGet("/v1/lora", (ServerState state) =>
        {
            var registry = state.LoraRegistry;
            var names = registry?.List() ?? Array.Empty<string>();
            string[] arr = names is string[] a ? a : names.ToArray();
            return Results.Ok(new LoraListResponse { Adapters = arr });
        });

        // ── POST /v1/lora/load — admin (gated) ──
        app.MapPost("/v1/lora/load", (LoraLoadRequest request, ServerState state) =>
        {
            if (!state.Options.AllowLoraAdminApi)
                return Results.StatusCode(403);

            if (string.IsNullOrWhiteSpace(request.Name))
                return Results.BadRequest(new ErrorResponse { Error = "name is required" });
            if (string.IsNullOrWhiteSpace(request.Path))
                return Results.BadRequest(new ErrorResponse { Error = "path is required" });

            var registry = state.LoraRegistry;
            if (registry is null)
                return Results.StatusCode(503);

            try
            {
                registry.Load(request.Name, request.Path);
                var adapter = registry.Get(request.Name);
                if (adapter is null)
                    return Results.StatusCode(500);

                return Results.Ok(new LoraLoadResponse
                {
                    Status = "loaded",
                    Name = adapter.Name,
                    Rank = adapter.Rank,
                    Alpha = adapter.Alpha,
                    TargetModules = adapter.TargetModules.ToArray(),
                });
            }
            catch (InvalidOperationException ex)
            {
                return Results.BadRequest(new ErrorResponse { Error = ex.Message });
            }
            catch (DirectoryNotFoundException ex)
            {
                return Results.BadRequest(new ErrorResponse { Error = ex.Message });
            }
            catch (FileNotFoundException ex)
            {
                return Results.BadRequest(new ErrorResponse { Error = ex.Message });
            }
            catch (NotSupportedException ex)
            {
                return Results.BadRequest(new ErrorResponse { Error = ex.Message });
            }
            catch (InvalidDataException ex)
            {
                return Results.BadRequest(new ErrorResponse { Error = ex.Message });
            }
        });

        // ── DELETE /v1/lora/{name} — admin (gated) ──
        app.MapDelete("/v1/lora/{name}", (string name, ServerState state) =>
        {
            if (!state.Options.AllowLoraAdminApi)
                return Results.StatusCode(403);

            var registry = state.LoraRegistry;
            if (registry is null)
                return Results.StatusCode(503);

            registry.Unload(name);
            return Results.Ok(new StatusResponse { Status = "unloaded" });
        });
    }

    /// <summary>
    /// Resolves a <c>lora_adapter</c> request field against the server's
    /// registry. Returns <c>null</c> when the field is unset or empty;
    /// throws <see cref="LoraAdapterNotFoundException"/> with the available
    /// names when the requested adapter is unknown.
    /// </summary>
    public static ILoraAdapter? Resolve(string? loraAdapterName, ServerState state)
    {
        if (string.IsNullOrWhiteSpace(loraAdapterName))
            return null;

        var registry = state.LoraRegistry;
        var adapter = registry?.Get(loraAdapterName);
        if (adapter is not null) return adapter;

        var available = registry?.List() ?? Array.Empty<string>();
        string availableStr = available.Count == 0
            ? "none loaded"
            : string.Join(", ", available);
        throw new LoraAdapterNotFoundException(
            $"LoRA adapter '{loraAdapterName}' is not loaded. Available adapters: [{availableStr}]. "
            + "Load via POST /v1/lora/load (requires AllowLoraAdminApi=true).");
    }
}

/// <summary>
/// Thrown when a request references a LoRA adapter that the server
/// has no record of. The message includes the list of currently-loaded
/// adapters to aid debugging.
/// </summary>
public sealed class LoraAdapterNotFoundException : Exception
{
    public LoraAdapterNotFoundException(string message) : base(message) { }
}
