using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Models;

/// <summary>
/// Convenience helper encapsulating the GGUF-open → config-extract → model-load pattern.
/// Single dispatch point for all architecture creation.
/// </summary>
public static class ModelLoader
{
    /// <summary>
    /// Loads a model from a GGUF file path. Opens the file, extracts config,
    /// and creates the appropriate model instance.
    /// </summary>
    /// <param name="path">Path to the GGUF model file.</param>
    /// <param name="threading">Threading configuration. Null defaults to single-threaded.</param>
    /// <returns>The loaded model, GGUF file handle, and model configuration.</returns>
    public static (IModel Model, GgufFile Gguf, ModelConfig Config) LoadFromGguf(
        string path, ThreadingConfig? threading = null)
    {
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        IModel model = config.Architecture switch
        {
            Architecture.NemotronH => NemotronHTransformerModel.LoadFromGguf(gguf, config),
            _ => TransformerModel.LoadFromGguf(gguf, config, threading ?? ThreadingConfig.SingleThreaded),
        };
        return (model, gguf, config);
    }
}
