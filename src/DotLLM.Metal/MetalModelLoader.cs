using DotLLM.Core.Models;
using DotLLM.Metal.Weights;
using DotLLM.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Metal;

/// <summary>
/// Provides functionality to load transformer models from GGUF files onto a Metal backend (GPU).
/// </summary>
public class MetalModelLoader
{
    /// <summary>
    /// Loads a transformer model from a GGUF file onto the specified GPU.
    /// </summary>
    /// <param name="path">Path to the GGUF model file.</param>
    /// <param name="strategy"></param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null for auto-detect.</param>
    /// <returns>The loaded model, GGUF file handle, and model configuration.</returns>
    public static ModelRunContext LoadFromGguf(
        string path, IWeightLoadStrategy strategy, string? ptxDir = null)
    {
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = MetalTransformerModel.LoadFromGguf(gguf, config, strategy, ptxDir: ptxDir);

        var ctx = model.Context;
        return new ModelRunContext(model, gguf, config,
            KvCacheFactory: (cfg, maxSeqLen) => new MetalKvCache(ctx, cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, maxSeqLen));
    }
}
