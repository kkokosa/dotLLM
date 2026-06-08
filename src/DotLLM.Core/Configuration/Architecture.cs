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
    QwenMoe
}
