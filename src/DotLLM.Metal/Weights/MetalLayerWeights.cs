namespace DotLLM.Metal.Weights;

/// <summary>
/// Per-layer GPU weight pointers. Linear projections are bundled in
/// <see cref="LoadedWeight"/> (pointer + format + dimensions) — the strategy
/// chosen at load time decides which of the two pointers (quantized / FP16)
/// is populated.
/// </summary>
public readonly struct MetalLayerWeights
{
    /// <summary>Attention query projection (Q).</summary>
    public LoadedWeight Q { get; init; }

    /// <summary>Attention key projection (K).</summary>
    public LoadedWeight K { get; init; }

    /// <summary>Attention value projection (V).</summary>
    public LoadedWeight V { get; init; }

    /// <summary>Attention output projection (O).</summary>
    public LoadedWeight O { get; init; }

    /// <summary>FFN gate projection (SwiGLU).</summary>
    public LoadedWeight Gate { get; init; }

    /// <summary>FFN up projection (SwiGLU).</summary>
    public LoadedWeight Up { get; init; }

    /// <summary>FFN down projection (SwiGLU).</summary>
    public LoadedWeight Down { get; init; }

    /// <summary>Pre-attention RMSNorm scale (FP16, mmap-direct).</summary>
    public nint AttnNormWeight { get; init; }

    /// <summary>Pre-FFN RMSNorm scale (FP16, mmap-direct).</summary>
    public nint FfnNormWeight { get; init; }

    /// <summary>QK-norm scale for Q (Qwen3, Cohere). 0 when absent.</summary>
    public nint QNormWeight { get; init; }

    /// <summary>QK-norm scale for K (Qwen3, Cohere). 0 when absent.</summary>
    public nint KNormWeight { get; init; }

    /// <summary>Attention Q bias (Qwen2). 0 when absent.</summary>
    public nint QBias { get; init; }

    /// <summary>Attention K bias (Qwen2). 0 when absent.</summary>
    public nint KBias { get; init; }

    /// <summary>Attention V bias (Qwen2). 0 when absent.</summary>
    public nint VBias { get; init; }

    /// <summary>Attention O bias. 0 when absent.</summary>
    public nint OBias { get; init; }

    /// <summary>FFN gate bias. 0 when absent.</summary>
    public nint GateBias { get; init; }

    /// <summary>FFN up bias. 0 when absent.</summary>
    public nint UpBias { get; init; }

    /// <summary>FFN down bias. 0 when absent.</summary>
    public nint DownBias { get; init; }
}
