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

    /// <summary>DeepSeek family (pre-V2; legacy placeholder).</summary>
    DeepSeek,

    /// <summary>
    /// DeepSeek-V2 family (<c>model_type=deepseek_v2</c>,
    /// <c>architectures[0]=DeepseekV2ForCausalLM</c>). Multi-head Latent
    /// Attention (MLA) with low-rank Q/KV factorisation + decoupled RoPE,
    /// combined with dense MoE in later layers (governed by
    /// <c>first_k_dense_replace</c>). Lite variant: 16 heads, qk_nope=128,
    /// qk_rope=64, v_head=128, kv_lora_rank=512, q_lora_rank=1536. Carries
    /// optional YaRN rope scaling. See <see cref="DotLLM.Core.Models.MlaConfig"/>.
    /// </summary>
    DeepSeekV2,

    /// <summary>
    /// DeepSeek-V3 family (<c>model_type=deepseek_v3</c>,
    /// <c>architectures[0]=DeepseekV3ForCausalLM</c>). Same MLA attention
    /// mechanism as V2 plus V3-specific MoE refinements (sigmoid router
    /// scoring, node-level aux-loss-free routing) — wired into the same
    /// <see cref="DotLLM.Core.Models.MlaConfig"/> for the attention side.
    /// </summary>
    DeepSeekV3,

    /// <summary>
    /// Mistral Mixtral family — dense transformer with top-k MoE FFN in every
    /// layer. HF <c>model_type</c>: <c>mixtral</c>. Same attention path as
    /// <see cref="Mistral"/> (GQA, RoPE, no sliding window by default); the
    /// MLP is replaced by <c>num_local_experts</c> parallel SwiGLU experts
    /// with <c>num_experts_per_tok</c> active per token. Shared experts are
    /// <b>not</b> a Mixtral thing (DeepSeek-V3 / old Qwen1.5-MoE territory,
    /// tracked separately). See <see cref="DotLLM.Core.Models.MoeConfig"/>.
    /// </summary>
    Mixtral,

    /// <summary>
    /// Alibaba Qwen-MoE family — Qwen1.5-MoE-A2.7B (<c>model_type=qwen2_moe</c>),
    /// Qwen2-MoE, Qwen3-MoE (<c>model_type=qwen3_moe</c>). Shares the Qwen
    /// attention path (GQA, NeoX-pair RoPE, optional sliding window, Qwen3
    /// QK-norm) with the dense <see cref="Qwen"/> variant but replaces the
    /// FFN with a top-k MoE block using HF tensor names
    /// <c>mlp.gate</c> + <c>mlp.experts.{j}.{gate_proj,up_proj,down_proj}</c>
    /// (NOT Mixtral's <c>block_sparse_moe.gate</c> / <c>experts.{j}.w1/w2/w3</c>).
    /// Optional shared-expert branch — a dense SwiGLU MLP running in parallel
    /// on EVERY token, optionally gated by a <c>sigmoid(hidden @ shared_expert_gate)</c>
    /// scalar — is present on Qwen1.5-MoE-A2.7B but absent on Qwen3-MoE.
    /// Qwen3-MoE further interleaves dense-MLP and MoE layers via
    /// <c>decoder_sparse_step</c> and <c>mlp_only_layers</c>. See
    /// <see cref="DotLLM.Core.Models.MoeConfig"/> for the per-layer flags.
    /// </summary>
    QwenMoe,

    /// <summary>
    /// HuggingFace SmolLM3 — Llama-shaped GQA transformer (3B SKU: 36 layers,
    /// 16 Q-heads, 4 KV-heads, head_dim=128, hidden=2048, intermediate=11008,
    /// vocab=128256, max_pos=65536). Distinguishing features vs Llama:
    /// <list type="bullet">
    ///   <item><b>NoPE layers</b> — every 4th layer (indices 3, 7, 11, ...
    ///     in the 3B config, supplied via the HF <c>no_rope_layers</c> 0/1
    ///     mask) skips RoPE entirely, attending on position-free Q/K.
    ///     Threaded as a per-layer boolean inside the forward dispatch;
    ///     see <see cref="DotLLM.Core.Models.ModelConfig.NoRopeLayers"/>.</item>
    ///   <item><b>YaRN context extension</b> — long-context SKUs ship
    ///     <c>rope_scaling</c> (<c>type=yarn</c>, factor &gt; 1) that lifts
    ///     the effective context to 128k. The base 3B checkpoint ships
    ///     <c>rope_scaling=null</c>; YaRN fields are surfaced via
    ///     <see cref="DotLLM.Core.PositionEncoding.RoPEConfig.ScalingType"/>.</item>
    ///   <item>Tool calling — Hermes-compatible
    ///     <c>&lt;tool_call&gt;{...}&lt;/tool_call&gt;</c> XML wrapper or
    ///     Pythonic <c>function_name(arg=value, ...)</c> syntax.</item>
    /// </list>
    /// HF discriminators: <c>architectures[0]=SmolLM3ForCausalLM</c>,
    /// <c>model_type=smollm3</c>. Tensor naming is identical to Llama, so
    /// loading routes through the standard <see cref="Llama"/> safetensors
    /// path with a conditional RoPE per layer.
    /// </summary>
    SmolLM3,

    /// <summary>
    /// Google Gemma 3 family. Text-only checkpoints carry
    /// <c>model_type=gemma3_text</c> + <c>architectures[0]=Gemma3TextForCausalLM</c>;
    /// multimodal checkpoints carry <c>model_type=gemma3</c> +
    /// <c>architectures[0]=Gemma3ForConditionalGeneration</c> and house the text-tower
    /// config under a <c>text_config</c> sub-object (we read only the text tower).
    /// <para>
    /// Distinguishing features vs Llama/Mistral/Qwen:
    /// <list type="bullet">
    ///   <item><b>GeGLU MLP</b> — gate path uses GELU (tanh approximation, HF
    ///     <c>hidden_activation=gelu_pytorch_tanh</c>) instead of SiLU; otherwise
    ///     shape-identical to SwiGLU (<c>down(act(gate(x)) * up(x))</c>).</item>
    ///   <item><b>RMSNorm <c>(1 + weight)</c> convention</b> — every RMSNorm scales by
    ///     <c>(1 + w)</c> rather than <c>w</c>. Absorbed at load time by pre-adding
    ///     <c>1.0</c> to every dequantised norm weight, so the existing kernel runs
    ///     unchanged.</item>
    ///   <item><b>Four RMSNorms per layer</b> — <c>input_layernorm</c>,
    ///     <c>post_attention_layernorm</c>, <c>pre_feedforward_layernorm</c>,
    ///     <c>post_feedforward_layernorm</c>. The post-attn and post-FFN norms run
    ///     <i>between</i> the sublayer output and the residual add (Gemma 2 design,
    ///     retained in Gemma 3).</item>
    ///   <item><b>Per-head Q/K RMSNorms</b> applied before RoPE (dim = <c>head_dim</c>,
    ///     identical plumbing to Qwen3 QK-norm).</item>
    ///   <item><b>Interleaved local/global attention</b> — most layers use
    ///     sliding-window attention of size <c>sliding_window</c>; every
    ///     <c>sliding_window_pattern</c>-th layer (1-indexed) uses full attention.
    ///     Encoded as a per-layer attention-type list on
    ///     <see cref="DotLLM.Core.Models.ModelConfig.PerLayerSlidingWindow"/>.</item>
    ///   <item><b>Query-pre-attn scalar</b> — attention score scale is
    ///     <c>1 / sqrt(query_pre_attn_scalar)</c> instead of the default
    ///     <c>1 / sqrt(head_dim)</c>.</item>
    ///   <item><b>Logit soft-capping</b> — optional <c>tanh(z / cap) * cap</c> applied
    ///     to attention scores (<c>attn_logit_softcapping</c>) and the final LM-head
    ///     logits (<c>final_logit_softcapping</c>). Gemma 2 sets both (50.0 and 30.0);
    ///     Gemma 3 leaves them null but the plumbing is wired regardless via
    ///     <see cref="DotLLM.Core.Models.ModelConfig.AttnLogitSoftcap"/> and
    ///     <see cref="DotLLM.Core.Models.ModelConfig.FinalLogitSoftcap"/>.</item>
    /// </list>
    /// </para>
    /// <para>
    /// Gemma 4 (released 2026-04-02) reuses every mechanism wired through this
    /// enum variant — see <see cref="Gemma4"/> for the per-version dispatch. The
    /// Gemma-family field block in <c>HfConfigExtractor</c> fires for both
    /// variants.
    /// </para>
    /// </summary>
    Gemma3,

    /// <summary>
    /// Google Gemma 4 family (released 2026-04-02 under Apache 2.0). Text-only
    /// checkpoints carry <c>model_type=gemma4_text</c> +
    /// <c>architectures[0]=Gemma4ForCausalLM</c>; multimodal checkpoints carry
    /// <c>model_type=gemma4</c> (or <c>gemma4_unified</c> for the newer unified
    /// vision/audio releases) + <c>architectures[0]=Gemma4ForConditionalGeneration</c>
    /// (or <c>Gemma4UnifiedForConditionalGeneration</c>) and house the text-tower
    /// config under a <c>text_config</c> sub-object (we read only the text tower).
    /// <para>
    /// The architecture reuses every Gemma 2/3 mechanism — same GeGLU activation
    /// (<c>gelu_pytorch_tanh</c>), same RMSNorm <c>(1 + weight)</c> convention,
    /// same four norms per layer, same Q/K per-head RMSNorms, same interleaved
    /// local/global attention pattern (driven by the <c>layer_types</c> array, e.g.
    /// five <c>sliding_attention</c> layers followed by one <c>full_attention</c>),
    /// same <c>final_logit_softcapping=30.0</c>, same <c>tie_word_embeddings=true</c>.
    /// </para>
    /// <para>
    /// New / divergent fields seen on the HF reference (e.g. <c>google/gemma-4-12B-*</c>)
    /// that are NOT yet consumed by this enum's loader path — the dense text-only
    /// forward will still run correctly on the public 12B/31B SKUs because each new
    /// feature is null / disabled by default in those checkpoints:
    /// <list type="bullet">
    ///   <item><c>rope_parameters.{full_attention, sliding_attention}</c> — split-RoPE
    ///     config (Gemma 3 inherited a single <c>rope_theta</c>; Gemma 4 uses different
    ///     theta for full vs sliding layers). The current loader reads the top-level
    ///     <c>rope_theta</c> which the HF Gemma 4 configs still carry as a sensible
    ///     fallback for both layer types.</item>
    ///   <item><c>global_head_dim</c>, <c>num_global_key_value_heads</c> — full-attention
    ///     layers can have different head shape than sliding layers. Disabled on the
    ///     12B SKU; full-attention layers reuse <c>head_dim</c> / <c>num_key_value_heads</c>.</item>
    ///   <item><c>use_double_wide_mlp</c>, <c>attention_k_eq_v</c>,
    ///     <c>num_kv_shared_layers</c>, <c>hidden_size_per_layer_input</c>,
    ///     <c>vocab_size_per_layer_input</c>, <c>enable_moe_block</c>, <c>num_experts</c>,
    ///     <c>top_k_experts</c> — all null / false / 0 on the public 12B/31B SKUs.
    ///     The 26B "A4B" MoE SKU will need <c>enable_moe_block</c>+<c>num_experts</c>
    ///     wired through the existing MoE path in a follow-up.</item>
    /// </list>
    /// </para>
    /// <para>
    /// Loading routes through the standard dense
    /// <c>DotLLM.Models.Architectures.TransformerModel</c> safetensors path,
    /// parameterised by the existing Gemma 2/3 <see cref="DotLLM.Core.Models.ModelConfig"/>
    /// fields (<see cref="DotLLM.Core.Models.ModelConfig.PerLayerSlidingWindow"/>,
    /// <see cref="DotLLM.Core.Models.ModelConfig.AttnLogitSoftcap"/>,
    /// <see cref="DotLLM.Core.Models.ModelConfig.FinalLogitSoftcap"/>,
    /// <see cref="DotLLM.Core.Models.ModelConfig.QueryPreAttnScalar"/>). ROADMAP
    /// Phase 8 Step 57.
    /// </para>
    /// </summary>
    Gemma4
}
