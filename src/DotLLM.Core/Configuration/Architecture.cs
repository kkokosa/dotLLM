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
    DeepSeekV3
}
