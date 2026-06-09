using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Core.Models;

/// <summary>
/// Complete configuration for a transformer model architecture. Populated from GGUF metadata or explicit construction.
/// A single <see cref="ModelConfig"/> parameterizes the transformer block to handle Llama/Mistral/Phi/Qwen/DeepSeek.
/// </summary>
public record ModelConfig
{
    /// <summary>Model architecture family.</summary>
    public required Architecture Architecture { get; init; }

    /// <summary>Vocabulary size (number of token embeddings).</summary>
    public required int VocabSize { get; init; }

    /// <summary>Hidden size (embedding dimension).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>FFN intermediate dimension.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Number of transformer layers.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of attention heads for queries.</summary>
    public required int NumAttentionHeads { get; init; }

    /// <summary>Number of KV heads. Equal to <see cref="NumAttentionHeads"/> for MHA, 1 for MQA, between for GQA.</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Dimension per attention head. Typically <see cref="HiddenSize"/> / <see cref="NumAttentionHeads"/>.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Maximum supported sequence length.</summary>
    public required int MaxSequenceLength { get; init; }

    /// <summary>Attention mechanism type (GQA or MLA).</summary>
    public AttentionType AttentionType { get; init; } = AttentionType.GQA;

    /// <summary>Positional encoding type.</summary>
    public PositionEncodingType PositionEncodingType { get; init; } = PositionEncodingType.RoPE;

    /// <summary>RoPE-specific configuration. Null when not using RoPE.</summary>
    public RoPEConfig? RoPEConfig { get; init; }

    /// <summary>Activation function used in FFN layers.</summary>
    public ActivationFunction ActivationFunction { get; init; } = ActivationFunction.SiLU;

    /// <summary>Normalization layer type.</summary>
    public NormType NormType { get; init; } = NormType.RMSNorm;

    /// <summary>Epsilon for normalization layers.</summary>
    public float NormEpsilon { get; init; } = 1e-5f;

    /// <summary>Whether input and output embeddings share weights.</summary>
    public bool TiedEmbeddings { get; init; }

    /// <summary>Sliding window size for local attention. Null = full attention.</summary>
    public int? SlidingWindowSize { get; init; }

    /// <summary>
    /// Per-layer sliding-window override (length = <see cref="NumLayers"/>). Each
    /// entry is the per-layer window size, or <see langword="null"/> for a full-attention
    /// layer. Null at the model level means "no per-layer override — every layer uses
    /// <see cref="SlidingWindowSize"/>". Populated for Gemma 3's interleaved local/global
    /// attention pattern (<c>sliding_window_pattern</c>); ignored for every architecture
    /// where every layer behaves the same.
    /// </summary>
    public IReadOnlyList<int?>? PerLayerSlidingWindow { get; init; }

    /// <summary>
    /// Optional attention-logit soft-cap (Gemma 2 / Gemma 3 <c>attn_logit_softcapping</c>).
    /// When non-null, attention scores <c>z</c> are transformed in-place between scaling
    /// and softmax as <c>z' = tanh(z / cap) * cap</c>. Gemma 2 sets this to 50.0;
    /// Gemma 3 leaves it null but the field is wired regardless.
    /// </summary>
    public float? AttnLogitSoftcap { get; init; }

    /// <summary>
    /// Optional final-logit soft-cap (Gemma 2 / Gemma 3 <c>final_logit_softcapping</c>).
    /// When non-null, the LM-head logits <c>z</c> are transformed as
    /// <c>z' = tanh(z / cap) * cap</c> after the LM-head projection and before sampling.
    /// Gemma 2 sets this to 30.0; Gemma 3 leaves it null but the field is wired
    /// regardless.
    /// </summary>
    public float? FinalLogitSoftcap { get; init; }

    /// <summary>
    /// Optional attention-score scale multiplier override (Gemma's
    /// <c>query_pre_attn_scalar</c>). When non-null the kernel uses
    /// <c>1 / sqrt(query_pre_attn_scalar)</c> instead of the default
    /// <c>1 / sqrt(<see cref="HeadDim"/>)</c>. Gemma 3 ships this as 256 (matching the
    /// pre-attn-scalar value used when training the 2.6B/9B/27B SKUs).
    /// </summary>
    public float? QueryPreAttnScalar { get; init; }

    /// <summary>MLA configuration. Only set for DeepSeek-style MLA attention.</summary>
    public MlaConfig? MlaConfig { get; init; }

    /// <summary>
    /// Mixture-of-Experts configuration. Non-null when the per-layer FFN is
    /// replaced by top-k dense routing over <see cref="MoeConfig.NumExperts"/>
    /// experts. Present on <see cref="DotLLM.Core.Configuration.Architecture.Mixtral"/>
    /// today; extensible to Qwen*-MoE and Phi-3.5-MoE via the same record.
    /// </summary>
    public MoeConfig? Moe { get; init; }

    /// <summary>Jinja2 chat template from model metadata. Null if not present.</summary>
    public string? ChatTemplate { get; init; }

    /// <summary>
    /// Layer indices that skip RoPE entirely (NoPE — "no positional encoding").
    /// Null or empty means every layer applies RoPE per the standard
    /// <see cref="RoPEConfig"/>. SmolLM3 ships a sparse pattern
    /// (every 4th layer — indices 3, 7, 11, ... on the 3B SKU). The forward
    /// pass tests <see cref="IsNoRopeLayer(int)"/> per layer and conditionally
    /// skips the RoPE rotation while leaving the rest of the GQA pipeline
    /// (projections, attention, output) intact. Non-RoPE architectures
    /// (Mamba-3, MLA decoupled-rope, ...) ignore this field.
    /// </summary>
    public IReadOnlyList<int>? NoRopeLayers { get; init; }

    /// <summary>
    /// Returns true when <paramref name="layerIdx"/> should skip the per-layer
    /// RoPE rotation (NoPE behaviour). Defaults to false when
    /// <see cref="NoRopeLayers"/> is null or empty — every layer applies RoPE.
    /// </summary>
    public bool IsNoRopeLayer(int layerIdx)
    {
        if (NoRopeLayers is null || NoRopeLayers.Count == 0)
            return false;
        return NoRopeLayers.Contains(layerIdx);
    }
}
