namespace DotLLM.Core.Models;

/// <summary>
/// Per-layer sub-layer kind for hybrid SSM+Transformer models (e.g. Nemotron-H).
/// Each block of the model selects exactly one of these to run between RMSNorm
/// and the residual add — there is no separate FFN pass per layer.
/// </summary>
public enum HybridLayerKind
{
    /// <summary>Mamba2 selective state-space layer.</summary>
    Ssm,

    /// <summary>Multi-head (GQA) attention layer.</summary>
    Attention,

    /// <summary>Position-wise MLP with squared-ReLU activation (no SwiGLU gate).</summary>
    Ffn
}

/// <summary>
/// Per-layer layout for a hybrid model. Captures which sub-layer kind runs at each
/// block index, plus per-layer attention/FFN dimensions sourced from GGUF array
/// metadata (e.g. <c>nemotron_h.attention.head_count_kv</c> as Int32[NumLayers]).
/// </summary>
/// <remarks>
/// For a layer where <see cref="LayerKind"/> is <see cref="HybridLayerKind.Attention"/>,
/// <see cref="HeadCountKv"/>[i] is the GQA KV-head count and <see cref="FeedForwardLength"/>[i]
/// is zero. For an FFN layer the inverse holds. SSM layers have both as zero.
/// </remarks>
public sealed record HybridLayerLayout
{
    /// <summary>Kind of sub-layer at each block index. Length = NumLayers.</summary>
    public required HybridLayerKind[] LayerKind { get; init; }

    /// <summary>Per-layer KV-head count from GGUF (zero for non-attention layers). Length = NumLayers.</summary>
    public required int[] HeadCountKv { get; init; }

    /// <summary>Per-layer FFN intermediate dim from GGUF (zero for non-FFN layers). Length = NumLayers.</summary>
    public required int[] FeedForwardLength { get; init; }
}
