using DotLLM.HuggingFace;
using DotLLM.Models.Gguf;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// GET /v1/models/inspect?path=... — read GGUF metadata without loading the model.
/// Returns layer count, architecture, and file size for UI configuration.
/// Path is restricted to the configured model directory to prevent path traversal.
/// </summary>
public static class ModelInspectEndpoint
{
    public static void Map(WebApplication app) =>
        app.MapGet("/v1/models/inspect", (string path, ServerState state) =>
        {
            if (string.IsNullOrEmpty(path))
                return Results.BadRequest(new ErrorResponse { Error = "Path is required" });

            var fullPath = Path.GetFullPath(path);

            if (!IsAllowedModelPath(fullPath, state))
                return Results.Json(
                    new ErrorResponse { Error = "Path is outside allowed model directories" },
                    statusCode: 403);

            if (!fullPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                return Results.BadRequest(new ErrorResponse { Error = "Only .gguf files are supported" });

            if (!File.Exists(fullPath))
                return Results.BadRequest(new ErrorResponse { Error = "File not found" });

            try
            {
                using var gguf = GgufFile.Open(fullPath);
                var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
                var fileSize = new FileInfo(fullPath).Length;

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

    /// <summary>
    /// Checks whether the given normalized path is within an allowed model directory.
    /// Allowed directories: the default HuggingFace model cache and the directory of the currently loaded model.
    /// </summary>
    internal static bool IsAllowedModelPath(string fullPath, ServerState state)
    {
        var modelsDir = Path.GetFullPath(HuggingFaceDownloader.DefaultModelsDirectory);
        if (!modelsDir.EndsWith(Path.DirectorySeparatorChar))
            modelsDir += Path.DirectorySeparatorChar;

        if (fullPath.StartsWith(modelsDir, StringComparison.OrdinalIgnoreCase))
            return true;

        if (!string.IsNullOrEmpty(state.LoadedModelPath))
        {
            var loadedDir = Path.GetFullPath(Path.GetDirectoryName(state.LoadedModelPath)!);
            if (!loadedDir.EndsWith(Path.DirectorySeparatorChar))
                loadedDir += Path.DirectorySeparatorChar;

            if (fullPath.StartsWith(loadedDir, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }
}
