namespace DotLLM.Core.Lora;

/// <summary>
/// Adapter-level LoRA hyperparameters as parsed from a HuggingFace
/// PEFT <c>adapter_config.json</c>. Carries only the fields required for
/// an inference-side runtime application of the adapter — fine-tuning
/// hyperparameters (learning rate, optimizer state, etc.) are out of scope.
/// </summary>
/// <param name="Rank">
/// LoRA rank <c>r</c>. Typical values 8–64. Determines the inner dimension
/// of the down-up factorisation: <c>B: [d_in, r]</c>, <c>A: [r, d_out]</c>.
/// </param>
/// <param name="Alpha">
/// LoRA scaling parameter — usually applied as <c>scale = alpha / rank</c>
/// when computing the delta. Stored verbatim so callers can choose the
/// scaling convention (plain LoRA, rsLoRA — out of scope this commit).
/// </param>
/// <param name="TargetModules">
/// Canonical projection names the adapter targets (e.g. <c>q_proj</c>,
/// <c>v_proj</c>, <c>k_proj</c>, <c>o_proj</c>). Informational — the actual
/// adapted projections are derived from the tensor names present in
/// <c>adapter_model.safetensors</c>.
/// </param>
/// <param name="Dropout">
/// LoRA dropout rate from training. Set on the config so loaders can
/// surface it for debugging — unused at inference.
/// </param>
public sealed record LoraConfig(
    int Rank,
    float Alpha,
    IReadOnlyList<string> TargetModules,
    float Dropout = 0.0f);
