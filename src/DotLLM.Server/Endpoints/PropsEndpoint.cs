using DotLLM.Core.Configuration;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET /props — server configuration, model info, and sampling defaults.
/// </summary>
public static class PropsEndpoint
{
    public static void Map(WebApplication app) =>
        app.MapGet("/props", (ServerState state) =>
        {
            var threading = new ThreadingConfig(state.Options.Threads, state.Options.DecodeThreads);
            return new PropsResponse
            {
                ModelId = state.IsReady ? state.Options.ModelId : null,
                ModelPath = state.IsReady ? state.LoadedModelPath : null,
                Architecture = state.Config?.Architecture.ToString(),
                NumLayers = state.Config?.NumLayers ?? 0,
                HiddenSize = state.Config?.HiddenSize ?? 0,
                VocabSize = state.Config?.VocabSize ?? 0,
                MaxSequenceLength = state.Config?.MaxSequenceLength ?? 0,
                Device = state.Options.Device,
                GpuLayers = state.Options.GpuLayers,
                Threads = threading.EffectiveThreadCount,
                SamplingDefaults = ToDto(state.SamplingDefaults),
                DraftModelPath = string.IsNullOrEmpty(state.DraftModelPath) ? null : state.DraftModelPath,
                IsReady = state.IsReady,
            };
        });

    internal static SamplingDefaultsDto ToDto(SamplingDefaults defaults) => new()
    {
        Temperature = defaults.Temperature,
        TopP = defaults.TopP,
        TopK = defaults.TopK,
        MinP = defaults.MinP,
        RepetitionPenalty = defaults.RepetitionPenalty,
        MaxTokens = defaults.MaxTokens,
        Seed = defaults.Seed,
    };
}
