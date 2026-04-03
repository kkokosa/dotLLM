using DotLLM.HuggingFace;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET /v1/models/available — list locally downloaded models.
/// POST /v1/models/load — hot-swap the loaded model.
/// </summary>
public static class ModelManagementEndpoint
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/v1/models/available", () =>
        {
            var models = HuggingFaceDownloader.ListLocalModels();
            return new AvailableModelsResponse
            {
                Models = models.Select(m => new AvailableModelDto
                {
                    RepoId = m.RepoId,
                    Filename = m.Filename,
                    FullPath = m.FullPath,
                    SizeBytes = m.SizeBytes,
                }).ToArray(),
            };
        });

        app.MapPost("/v1/models/load", async (ModelLoadRequest request, ServerState state, CancellationToken ct) =>
        {
            var resolvedPath = ServerStartup.ResolveModelPath(request.Model, request.Quant);
            if (resolvedPath is null)
                return Results.BadRequest(new ErrorResponse { Error = $"Model not found: {request.Model}" });

            try
            {
                await state.SwapModelAsync(async () =>
                {
                    var newOptions = state.Options with
                    {
                        Model = request.Model,
                        Quant = request.Quant,
                        Device = request.Device ?? state.Options.Device,
                        GpuLayers = request.GpuLayers ?? state.Options.GpuLayers,
                        CacheTypeK = request.CacheTypeK ?? state.Options.CacheTypeK,
                        CacheTypeV = request.CacheTypeV ?? state.Options.CacheTypeV,
                        Threads = request.Threads ?? state.Options.Threads,
                        DecodeThreads = request.DecodeThreads ?? state.Options.DecodeThreads,
                        ModelId = Path.GetFileNameWithoutExtension(resolvedPath),
                    };
                    var newState = await Task.Run(() => ServerStartup.LoadModel(resolvedPath, newOptions), ct);

                    // Transfer new state fields into the existing ServerState
                    state.Options = newOptions;
                    state.Config = newState.Config;
                    state.Model = newState.Model;
                    state.Tokenizer = newState.Tokenizer;
                    state.ChatTemplate = newState.ChatTemplate;
                    state.Generator = newState.Generator;
                    state.ToolCallParser = newState.ToolCallParser;
                    state.KvCacheConfig = newState.KvCacheConfig;
                    state.KvCacheFactory = newState.KvCacheFactory;
                    state.PrefixCache = newState.PrefixCache;
                    state.LoadedModelPath = resolvedPath;
                    state.CurrentGguf = newState.CurrentGguf;

                    await Task.CompletedTask;
                }, ct);

                return Results.Ok(new ModelLoadResponse
                {
                    Status = "loaded",
                    Model = request.Model,
                });
            }
            catch
            {
                return Results.StatusCode(500);
            }
        });
    }
}
