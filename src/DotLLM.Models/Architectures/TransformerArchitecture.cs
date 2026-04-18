using DotLLM.Core.Backends;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Factory for creating standard transformer architecture models from GGUF files.
/// </summary>
public sealed class TransformerArchitecture : IModelArchitecture
{
    private readonly GgufFile _gguf;

    /// <summary>
    /// Creates a new transformer architecture factory bound to the given GGUF file.
    /// The <paramref name="gguf"/> must remain alive for the lifetime of any model created by this factory.
    /// </summary>
    /// <param name="gguf">An opened GGUF file containing transformer weights.</param>
    public TransformerArchitecture(GgufFile gguf)
    {
        _gguf = gguf ?? throw new ArgumentNullException(nameof(gguf));
    }

    /// <inheritdoc/>
    public IReadOnlyList<Architecture> SupportedArchitectures { get; } =
        [Architecture.Llama, Architecture.Mistral, Architecture.Phi, Architecture.Qwen, Architecture.NemotronH];

    /// <inheritdoc/>
    public IModel CreateModel(ModelConfig config, IBackend backend)
    {
        if (config.Architecture is Architecture.DeepSeek)
            throw new NotSupportedException(
                "DeepSeek models require Multi-Latent Attention (MLA) which is not yet implemented. " +
                "See roadmap Step 48.");

        if (config.Architecture is Architecture.NemotronH)
            return NemotronHTransformerModel.LoadFromGguf(_gguf, config);

        if (config.Architecture is not (Architecture.Llama or Architecture.Mistral
                                    or Architecture.Phi or Architecture.Qwen))
            throw new ArgumentException(
                $"TransformerArchitecture does not support {config.Architecture}.", nameof(config));

        return TransformerModel.LoadFromGguf(_gguf, config);
    }
}
