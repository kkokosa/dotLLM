using DotLLM.Core.Backends;
using DotLLM.Core.Configuration;

namespace DotLLM.Core.Models;

/// <summary>
/// Factory for creating model instances from a configuration.
/// Each supported architecture registers an implementation of this interface.
/// </summary>
public interface IModelArchitecture
{
    /// <summary>Architectures this factory can instantiate.</summary>
    IReadOnlyList<Architecture> SupportedArchitectures { get; }

    /// <summary>
    /// Creates a model instance from a configuration and backend.
    /// </summary>
    /// <param name="config">Model configuration (from GGUF metadata or explicit).</param>
    /// <param name="backend">Compute backend to use.</param>
    /// <returns>A loaded model ready for inference.</returns>
    IModel CreateModel(ModelConfig config, IBackend backend);
}
