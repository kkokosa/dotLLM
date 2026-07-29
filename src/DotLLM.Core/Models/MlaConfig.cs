namespace DotLLM.Core.Models;

/// <summary>
/// Configuration for Multi-head Latent Attention (MLA), used by DeepSeek-V2 and
/// DeepSeek-V3 (and their Lite / MoE variants). MLA factorises Q and KV through
/// low-rank bottlenecks (<see cref="QLoraRank"/>, <see cref="KvLoraRank"/>) and
/// carries positional information on a decoupled RoPE sub-dimension
/// (<see cref="QkRopeHeadDim"/>) while the bulk of Q·K runs on a larger
/// no-position sub-dimension (<see cref="QkNopeHeadDim"/>). The value side
/// has its own head dimension (<see cref="VHeadDim"/>) that may differ from
/// the Q·K head dimension.
/// </summary>
/// <remarks>
/// <para>
/// <b>Projection topology (per DeepSeek-V2/V3).</b>
/// <list type="bullet">
///   <item><c>q_a_proj</c> : <c>hidden → q_lora_rank</c> (omitted when
///     <see cref="QLoraRank"/> is 0; the model then carries a monolithic
///     <c>q_proj</c>: <c>hidden → n_heads * (qk_nope + qk_rope)</c>).</item>
///   <item><c>q_a_layernorm</c> : RMSNorm over <c>q_lora_rank</c>.</item>
///   <item><c>q_b_proj</c> : <c>q_lora_rank → n_heads * (qk_nope + qk_rope)</c>.</item>
///   <item><c>kv_a_proj_with_mqa</c> : <c>hidden → kv_lora_rank + qk_rope_head_dim</c>
///     where the last <c>qk_rope_head_dim</c> components are the MQA-shared
///     <c>k_pe</c> (a single rope-K that broadcasts across all heads).</item>
///   <item><c>kv_a_layernorm</c> : RMSNorm over <c>kv_lora_rank</c>.</item>
///   <item><c>kv_b_proj</c> : <c>kv_lora_rank → n_heads * (qk_nope_head_dim + v_head_dim)</c>.
///     Per-head layout: first <c>qk_nope_head_dim</c> entries are the
///     position-free K (broadcast-free, one per Q head), followed by
///     <c>v_head_dim</c> entries for V.</item>
///   <item><c>o_proj</c> : <c>n_heads * v_head_dim → hidden</c>.</item>
/// </list>
/// </para>
/// <para>
/// <b>Attention math.</b> Per head, Q = concat(Q_nope, Q_rope) and K =
/// concat(K_nope_per_head, broadcast(K_rope_shared)); attention scores use
/// <c>1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)</c>. The weighted sum runs
/// over V whose per-head dim is <see cref="VHeadDim"/> (may be
/// <c>≠ qk_head_dim</c>). The aggregated output has shape
/// <c>[seq, n_heads * v_head_dim]</c> and feeds <c>o_proj</c>.
/// </para>
/// <para>
/// <b>First-PoC simplification.</b> We do NOT perform the "absorption"
/// optimisation (fusing <c>W_q_nope @ W_k_nope^T</c>). Full Q/K/V are
/// materialised at runtime and standard scaled dot-product attention is used.
/// </para>
/// </remarks>
public sealed record MlaConfig
{
    /// <summary>
    /// Latent rank for the KV compression bottleneck. Typically 512 on
    /// DeepSeek-V2-Lite and DeepSeek-V2. The compressed KV tensor has shape
    /// <c>[kv_lora_rank]</c> per token before <c>kv_b_proj</c> expands it.
    /// Must be positive.
    /// </summary>
    public required int KvLoraRank { get; init; }

    /// <summary>
    /// Latent rank for the Q compression bottleneck. Typically 1536 on
    /// DeepSeek-V2-Lite / V2. Zero (or null in source config) indicates that
    /// the model skips the Q factorisation — in that case a single monolithic
    /// <c>q_proj: hidden → n_heads * (qk_nope + qk_rope)</c> is used, and the
    /// <c>q_a_proj</c> / <c>q_a_layernorm</c> / <c>q_b_proj</c> tensors are
    /// absent. DeepSeek-V3 uses the factorised form with non-zero rank on all
    /// sizes the team has published; the zero case exists primarily as a
    /// forward-compat / unit-test hook.
    /// </summary>
    public int QLoraRank { get; init; }

    /// <summary>
    /// Non-rope portion of the Q·K head dimension. Typically 128 on
    /// DeepSeek-V2-Lite / V2. Applied without any positional encoding —
    /// supplies the bulk of the attention score via
    /// <c>Q_nope · K_nope</c>.
    /// </summary>
    public required int QkNopeHeadDim { get; init; }

    /// <summary>
    /// Rope portion of the Q·K head dimension. Typically 64 on
    /// DeepSeek-V2-Lite / V2. Carries RoPE positional rotation. Must be even
    /// (RoPE rotates adjacent element pairs). The K side is MQA-shared — a
    /// single <c>qk_rope_head_dim</c>-wide rope-K broadcasts across all heads.
    /// </summary>
    public required int QkRopeHeadDim { get; init; }

    /// <summary>
    /// Head dimension for V. Typically 128 on DeepSeek-V2 — equal to
    /// <see cref="QkNopeHeadDim"/> in the V2 Lite config but not required to
    /// be. The attention output per head has this dim; the final
    /// <c>o_proj</c> input dim is <c>n_heads * v_head_dim</c>.
    /// </summary>
    public required int VHeadDim { get; init; }

    /// <summary>
    /// RoPE base frequency used for the decoupled rope sub-dimension. Mirrors
    /// <c>rope_theta</c> in the HF config. DeepSeek-V2 publishes <c>10000</c>
    /// on Lite and larger values with YaRN on the full V2. When YaRN is in
    /// use this is the <i>base</i> theta — the YaRN rescaling is applied on
    /// top via <see cref="RopeScalingFactor"/> and friends; this PoC
    /// implements only the plain-RoPE path and uses YaRN parameters only when
    /// they can be collapsed into a scalar mscale.
    /// </summary>
    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>
    /// Optional YaRN context-length scaling factor (HF <c>rope_scaling.factor</c>).
    /// Null when no rope scaling is configured. Not yet applied in the
    /// forward kernel — surfaced so the loader round-trips config without
    /// data loss, to be consumed once YaRN is wired in.
    /// </summary>
    public float? RopeScalingFactor { get; init; }

    /// <summary>
    /// Optional YaRN mscale (HF <c>rope_scaling.mscale</c>). DeepSeek-V2
    /// applies a softmax scaling correction of
    /// <c>mscale = 0.1 * mscale_all_dim * log(factor) + 1.0</c> to the
    /// attention scale; we expose the raw inputs for follow-up work.
    /// </summary>
    public float? RopeScalingMscale { get; init; }

    /// <summary>
    /// Optional YaRN mscale_all_dim (HF <c>rope_scaling.mscale_all_dim</c>).
    /// Paired with <see cref="RopeScalingMscale"/>.
    /// </summary>
    public float? RopeScalingMscaleAllDim { get; init; }

    /// <summary>
    /// Optional original max-position-embeddings baseline for YaRN
    /// interpolation (HF <c>rope_scaling.original_max_position_embeddings</c>).
    /// </summary>
    public int? RopeScalingOriginalMaxPositionEmbeddings { get; init; }

    /// <summary>
    /// Per-head Q·K total dimension: <c>qk_nope_head_dim + qk_rope_head_dim</c>.
    /// Used for the attention scale <c>1 / sqrt(qk_head_dim)</c>.
    /// </summary>
    public int QkHeadDim => QkNopeHeadDim + QkRopeHeadDim;

    /// <summary>
    /// When <see langword="true"/>, the forward pass uses the latent MLA
    /// KV-cache (<c>MlaLatentKvState</c>) and the absorbed-form attention
    /// kernel — <c>Q_latent[h] = W_UK[h]^T @ Q_nope[h]</c>, scores against
    /// the shared latent, output expanded via <c>W_UV</c>. Storage drops
    /// ~7× vs the Phase A expanded cache (see <c>docs/KV_CACHE.md</c>).
    /// </summary>
    /// <remarks>
    /// Default <see langword="false"/> = Phase A cache (expanded per-head
    /// K_nope/V, the numerical oracle). Flip to <see langword="true"/>
    /// once the Phase B path is validated against Phase A within 1e-3 on
    /// the target checkpoint. Set per-config, not globally — an integration
    /// test can load the same model twice with different settings and
    /// diff the logits.
    /// </remarks>
    public bool UseLatentCache { get; init; }

    /// <summary>
    /// When <see langword="true"/>, the forward pass uses the latent
    /// <c>MlaLatentKvState</c> cache (same ~7× memory win as
    /// <see cref="UseLatentCache"/>) but dispatches the attention kernel by
    /// sequence length: <b>prefill</b> (<c>seqLen &gt; 1</c>) expands the
    /// cached latents into per-head K_nope/V in a local scratch buffer and
    /// runs the standard 192-dim MHA attention loop (compute-bound, cheaper
    /// than the 576-dim absorbed form at long prefill seqKv); <b>decode</b>
    /// (<c>seqLen == 1</c>) uses the Phase B absorbed kernel verbatim
    /// (bandwidth-bound — 576-dim MQA-style read of the compact latent
    /// cache). Mirrors vLLM's production MLA backend split.
    /// </summary>
    /// <remarks>
    /// Mutually exclusive with <see cref="UseLatentCache"/>. The cache
    /// format stored on disk is identical to Phase B
    /// (<c>c_kv + k_pe</c> per token), so a decode step after a Phase C
    /// prefill consumes the same latents a pure-Phase-B prefill would
    /// have produced — Phase A's expanded K_nope/V is scratch only during
    /// the prefill step and is discarded.
    /// </remarks>
    public bool UseHybridMlaCache { get; init; }

    /// <summary>
    /// Compute the YaRN softmax-scale multiplier to fold into the attention
    /// scale: returns <c>mscale² = (yarn_get_mscale(factor, mscale_all_dim))²</c>
    /// when YaRN scaling is configured and active (<c>factor &gt; 1</c>), else
    /// <c>1.0f</c>. The caller applies this as
    /// <c>softmax_scale = 1/sqrt(qk_head_dim) * multiplier</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mirrors the HF reference <c>modeling_deepseek.yarn_get_mscale</c>:
    /// <code>
    ///   def yarn_get_mscale(scale=1, mscale=1):
    ///       if scale &lt;= 1: return 1.0
    ///       return 0.1 * mscale * math.log(scale) + 1.0
    /// </code>
    /// and the softmax correction <c>scale *= mscale * mscale</c>. Uses
    /// <see cref="RopeScalingMscaleAllDim"/>, NOT <see cref="RopeScalingMscale"/> —
    /// the V2 reference applies <c>mscale_all_dim</c> to the softmax scale and
    /// uses <c>mscale</c> only for RoPE frequency scaling (not wired here yet).
    /// </para>
    /// </remarks>
    public float ComputeYarnSoftmaxScaleMultiplier()
    {
        if (RopeScalingFactor is not float factor || factor <= 1.0f)
            return 1.0f;
        if (RopeScalingMscaleAllDim is not float mscaleAllDim || mscaleAllDim == 0.0f)
            return 1.0f;

        float mscale = 0.1f * mscaleAllDim * MathF.Log(factor) + 1.0f;
        return mscale * mscale;
    }
}
