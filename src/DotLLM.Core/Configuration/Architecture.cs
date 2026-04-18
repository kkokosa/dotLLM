namespace DotLLM.Core.Configuration;

/// <summary>
/// Supported model architectures.
/// </summary>
public enum Architecture
{
    /// <summary>Meta Llama family.</summary>
    Llama,

    /// <summary>Mistral AI family.</summary>
    Mistral,

    /// <summary>Microsoft Phi family.</summary>
    Phi,

    /// <summary>Alibaba Qwen family.</summary>
    Qwen,

    /// <summary>DeepSeek family.</summary>
    DeepSeek,

    /// <summary>
    /// NVIDIA Nemotron-H — hybrid Mamba2 SSM + Transformer attention/MLP per layer.
    /// Used by the Nemotron-3 family. GGUF architecture string: <c>nemotron_h</c>.
    /// </summary>
    NemotronH
}
