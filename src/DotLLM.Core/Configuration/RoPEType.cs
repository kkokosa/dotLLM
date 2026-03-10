namespace DotLLM.Core.Configuration;

/// <summary>
/// Determines the element-pairing convention used when applying Rotary Position Embeddings.
/// This must match the GGUF Q/K weight layout: models whose converter permutes Q/K weights
/// (Llama, Mistral) use <see cref="Norm"/>, while models that keep HuggingFace weight order
/// (Qwen, Phi) use <see cref="NeoX"/>.
/// </summary>
public enum RoPEType
{
    /// <summary>
    /// Interleaved pairing (GPT-J / llama.cpp "NORM"): rotates pairs <c>(2i, 2i+1)</c>.
    /// Used when the GGUF converter permuted Q/K weights (Llama, Mistral).
    /// </summary>
    Norm = 0,

    /// <summary>
    /// Non-interleaved pairing (GPT-NeoX / HuggingFace "rotate_half"):
    /// rotates pairs <c>(i, i + headDim/2)</c>.
    /// Used when Q/K weights are in original HuggingFace order (Qwen, Phi).
    /// </summary>
    NeoX = 2,
}
