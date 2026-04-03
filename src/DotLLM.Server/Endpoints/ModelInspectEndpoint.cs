using DotLLM.Models.Gguf;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET /v1/models/inspect?path=... — read GGUF metadata without loading the model.
/// Returns layer count, architecture, and file size for UI configuration.
/// </summary>
public static class ModelInspectEndpoint
{
    public static void Map(WebApplication app) =>
        app.MapGet("/v1/models/inspect", (string path) =>
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
                return Results.BadRequest(new ErrorResponse { Error = "File not found" });

            try
            {
                using var gguf = GgufFile.Open(path);
                var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
                var fileSize = new FileInfo(path).Length;

                return Results.Ok(new ModelInspectResponse
                {
                    Architecture = config.Architecture.ToString(),
                    NumLayers = config.NumLayers,
                    HiddenSize = config.HiddenSize,
                    NumKvHeads = config.NumKvHeads,
                    HeadDim = config.HeadDim,
                    VocabSize = config.VocabSize,
                    MaxSequenceLength = config.MaxSequenceLength,
                    FileSizeBytes = fileSize,
                });
            }
            catch
            {
                return Results.BadRequest(new ErrorResponse { Error = "Failed to read GGUF metadata" });
            }
        });
}
