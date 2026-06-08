namespace DotLLM.Core.Models;

/// <summary>
/// Dense-routing top-k Mixture-of-Experts configuration. Present on a
/// <see cref="ModelConfig"/> iff the model's FFN is replaced by an MoE block
/// (Mixtral, Qwen*-MoE without shared experts, Phi-3.5-MoE, ...).
/// </summary>
/// <remarks>
/// <para>
/// <b>Semantics (Mixtral convention).</b> For each token the router projects
/// <c>hidden[hidden_size]</c> through <c>gate.weight[num_experts, hidden_size]</c>
/// to produce <c>num_experts</c> logits. Softmax is applied over the full
/// expert set, then the <see cref="NumExpertsPerTok"/> largest entries are
/// gathered. The gathered probabilities are re-normalised by dividing by
/// their own sum (not a second softmax) so the top-k gating weights sum to
/// 1.0 per token. Each selected expert runs an independent SwiGLU MLP over
/// the token's hidden state and its output is scaled by the gating weight
/// and summed.
/// </para>
/// <para>
/// <b>Out of scope for this config.</b> Shared experts (DeepSeek-V3,
/// Qwen1.5-MoE), router aux-loss (training-only), expert parallelism, and
/// fused GroupedGEMM kernels. Those are handled elsewhere in the roadmap.
/// </para>
/// <para>
/// <b>Expert MLP shape.</b> Each expert is a SwiGLU MLP with the same
/// <c>gate_proj</c>/<c>up_proj</c>/<c>down_proj</c> topology as dense Llama
/// — dims <c>[moe_intermediate_size, hidden_size]</c>,
/// <c>[moe_intermediate_size, hidden_size]</c>, and
/// <c>[hidden_size, moe_intermediate_size]</c> respectively. Mixtral reuses
/// the top-level <see cref="ModelConfig.IntermediateSize"/> for the MoE
/// expert width; Phi-3.5-MoE exposes a separate <c>moe_intermediate_size</c>
/// that is surfaced via <see cref="MoeIntermediateSize"/>.
/// </para>
/// </remarks>
public sealed record MoeConfig
{
    /// <summary>
    /// Total number of experts per MoE layer (HF <c>num_local_experts</c> or
    /// <c>num_experts</c>). Typically 8 for Mixtral-8x7B, 16/64/... for others.
    /// Must be &gt; 0 and &gt;= <see cref="NumExpertsPerTok"/>.
    /// </summary>
    public required int NumExperts { get; init; }

    /// <summary>
    /// Number of experts activated per token (HF <c>num_experts_per_tok</c>,
    /// also known as top-k). Typically 2 for Mixtral / Qwen-MoE / Phi-3.5-MoE.
    /// Must satisfy <c>1 &lt;= NumExpertsPerTok &lt;= NumExperts</c>.
    /// </summary>
    public required int NumExpertsPerTok { get; init; }

    /// <summary>
    /// FFN intermediate width per expert. Mixtral reuses the top-level
    /// <see cref="ModelConfig.IntermediateSize"/> for its experts, while
    /// Phi-3.5-MoE exposes a separate <c>moe_intermediate_size</c>. When the
    /// HF config declares both (<c>intermediate_size</c> ≠
    /// <c>moe_intermediate_size</c>) this carries the per-expert value; when
    /// only <c>intermediate_size</c> exists it mirrors that. Callers SHOULD
    /// use this value, not <see cref="ModelConfig.IntermediateSize"/>, when
    /// allocating MoE expert scratch.
    /// </summary>
    public required int MoeIntermediateSize { get; init; }

    /// <summary>
    /// Whether to renormalise the top-k routing probabilities to sum to 1.0
    /// after selection. Mixtral always does this (equivalent to <c>true</c>);
    /// Qwen1.5-MoE-A2.7B ships with <c>norm_topk_prob: false</c> while
    /// Qwen3-MoE ships with <c>norm_topk_prob: true</c>. When <c>false</c>,
    /// the raw softmax-over-all-experts probabilities are carried through as
    /// gating weights (so their sum per token is &lt; 1.0 by construction,
    /// softening the expert-output contribution).
    /// </summary>
    public bool NormTopKProb { get; init; } = true;

    /// <summary>
    /// Optional shared-expert intermediate width <b>per shared expert</b>.
    /// Present on Qwen1.5-MoE-A2.7B (<c>shared_expert_intermediate_size: 5632</c>,
    /// one shared expert) and DeepSeek-V2/V3 (<c>moe_intermediate_size</c> per
    /// shared expert; multiple shared experts summed — see
    /// <see cref="NumSharedExperts"/>). When non-null, the MoE block runs
    /// <see cref="NumSharedExperts"/> dense SwiGLU MLPs (each
    /// <see cref="SharedExpertIntermediateSize"/> wide) in parallel with the
    /// routed top-k path on EVERY token and adds their summed (optionally
    /// sigmoid-gated) output to the routed sum. When null, the layer is
    /// Mixtral-style — routed-only. See <see cref="HasSharedExpertGate"/> for
    /// the optional scalar gate (Qwen1.5-MoE only, single shared).
    /// </summary>
    public int? SharedExpertIntermediateSize { get; init; }

    /// <summary>
    /// Number of parallel shared experts whose outputs are summed into the
    /// shared-expert branch. Defaults to 1 — matches Qwen1.5-MoE's single
    /// <c>mlp.shared_expert.*</c> tensor set. DeepSeek-V2/V3 ship with
    /// <c>n_shared_experts &gt;= 1</c> and plural
    /// <c>mlp.shared_experts.{k}.*</c> tensor naming; each is
    /// <see cref="SharedExpertIntermediateSize"/> wide and they are summed
    /// (equally-weighted, no gating) into the routed-MoE sum. Must be
    /// <c>&gt;= 1</c> whenever <see cref="SharedExpertIntermediateSize"/> is
    /// non-null; ignored otherwise.
    /// </summary>
    public int NumSharedExperts { get; init; } = 1;

    /// <summary>
    /// When <c>true</c> the shared-expert contribution is multiplied by a
    /// per-token sigmoid scalar computed from a dense <c>[hidden_size → 1]</c>
    /// projection (HF: <c>mlp.shared_expert_gate.weight</c>). Qwen1.5-MoE uses
    /// this gate (always with <see cref="NumSharedExperts"/> = 1); DeepSeek-V2/V3
    /// does not. Ignored when <see cref="SharedExpertIntermediateSize"/> is null.
    /// </summary>
    public bool HasSharedExpertGate { get; init; }

    /// <summary>
    /// Qwen-MoE layer-level sparsity stride: only layers where
    /// <c>(layerIdx + 1) % DecoderSparseStep == 0</c> use the MoE FFN; the
    /// others run a dense SwiGLU MLP. Qwen3-MoE tiny-random checkpoints set
    /// this to <c>2</c> (every second layer is MoE). Mixtral / Qwen1.5-MoE /
    /// Phi-3.5-MoE set this to <c>1</c> (every layer is MoE) — the default.
    /// </summary>
    public int DecoderSparseStep { get; init; } = 1;

    /// <summary>
    /// Qwen-MoE per-layer override: layer indices that are FORCED to dense
    /// SwiGLU MLP even if the sparsity stride would otherwise mark them MoE.
    /// Empty for most checkpoints. Null is treated as empty.
    /// </summary>
    public IReadOnlyList<int>? MlpOnlyLayers { get; init; }

    /// <summary>
    /// Returns true if layer <paramref name="layerIdx"/> is a routed-MoE
    /// layer under the current configuration. Checks the
    /// <see cref="MlpOnlyLayers"/> override first (forced dense), then the
    /// <see cref="DecoderSparseStep"/> stride. For Mixtral-style configs
    /// (<c>DecoderSparseStep=1</c>, <c>MlpOnlyLayers=null</c>) this always
    /// returns <c>true</c>.
    /// </summary>
    public bool IsMoeLayer(int layerIdx)
    {
        if (MlpOnlyLayers is not null && MlpOnlyLayers.Contains(layerIdx))
            return false;
        return ((layerIdx + 1) % DecoderSparseStep) == 0;
    }
}
