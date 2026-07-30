using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models;

/// <summary>
/// Represents the context required to execute a model run. This includes the loaded model,
/// the associated GGUF file containing tensor data, and the configuration details of the model.
/// </summary>
/// <param name="Model">The loaded transformer model.</param>
/// <param name="Gguf">The opened GGUF file handle (must outlive the model).</param>
/// <param name="Config">Model configuration extracted from GGUF metadata.</param>
/// <param name="KvCacheFactory">
/// Optional factory for creating backend-appropriate KV-caches.
/// When null, the text generator falls back to the default CPU cache.
/// </param>
public record ModelRunContext(
    IModel Model,
    GgufFile Gguf,
    ModelConfig Config,
    Func<ModelConfig, int, IKvCache>? KvCacheFactory = null);
