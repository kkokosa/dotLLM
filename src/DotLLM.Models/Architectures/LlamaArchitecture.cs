using DotLLM.Core.Backends;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Factory for creating Llama-family models from GGUF files.
/// </summary>
public sealed class LlamaArchitecture : IModelArchitecture
{
    private readonly GgufFile _gguf;

    /// <summary>
    /// Creates a new Llama architecture factory bound to the given GGUF file.
    /// The <paramref name="gguf"/> must remain alive for the lifetime of any model created by this factory.
    /// </summary>
    /// <param name="gguf">An opened GGUF file containing Llama-family weights.</param>
    public LlamaArchitecture(GgufFile gguf)
    {
        _gguf = gguf ?? throw new ArgumentNullException(nameof(gguf));
    }

    /// <inheritdoc/>
    public IReadOnlyList<Architecture> SupportedArchitectures { get; } = [Architecture.Llama];

    /// <inheritdoc/>
    public IModel CreateModel(ModelConfig config, IBackend backend)
    {
        if (config.Architecture != Architecture.Llama)
            throw new ArgumentException(
                $"LlamaArchitecture does not support {config.Architecture}.", nameof(config));

        return LlamaModel.LoadFromGguf(_gguf, config);
    }
}
