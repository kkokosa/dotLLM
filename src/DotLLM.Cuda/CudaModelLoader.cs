using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda;

/// <summary>
/// Convenience helper for loading a model onto a GPU from a GGUF file.
/// </summary>
public static class CudaModelLoader
{
    /// <summary>
    /// Loads a transformer model from a GGUF file onto the specified GPU.
    /// </summary>
    /// <param name="path">Path to the GGUF model file.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null for auto-detect.</param>
    /// <returns>The loaded model, GGUF file handle, and model configuration.</returns>
    public static (CudaTransformerModel Model, GgufFile Gguf, ModelConfig Config) LoadFromGguf(
        string path, int deviceId = 0, string? ptxDir = null)
    {
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId, ptxDir);
        return (model, gguf, config);
    }
}
