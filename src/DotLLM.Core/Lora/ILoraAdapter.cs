using DotLLM.Core.Models;

namespace DotLLM.Core.Lora;

/// <summary>
/// A loaded LoRA adapter — a collection of low-rank A/B factor pairs keyed
/// by <c>(layerIndex, projName)</c>. Applied at inference time to compute
/// <c>y += alpha × (x · B) · A</c> in addition to the base <c>y = x · W</c>.
/// </summary>
/// <remarks>
/// <para>
/// Per the dotLLM design (see <c>docs/LORA.md</c>), adapters are NEVER
/// merged into base weights. The cost is a small per-layer overhead
/// (typically &lt;5% for r=16); the gain is instant adapter switching with
/// no copies and concurrent multi-adapter serving.
/// </para>
/// <para>
/// All adapter weight buffers live in CPU native memory aligned to 64 bytes
/// (per project conventions). GPU-side adapter staging is a follow-up
/// (Phase 4b) — when that lands, the same <see cref="ILoraAdapter"/> handle
/// will own both the CPU mirror and the device-side mirror.
/// </para>
/// </remarks>
public interface ILoraAdapter : IDisposable
{
    /// <summary>Adapter name (typically the directory name on disk).</summary>
    string Name { get; }

    /// <summary>LoRA rank — inner dimension of the A/B factorisation.</summary>
    int Rank { get; }

    /// <summary>
    /// LoRA alpha — scaling numerator. The runtime applies
    /// <c>scale = Alpha / Rank</c> when accumulating the delta.
    /// </summary>
    float Alpha { get; }

    /// <summary>
    /// Canonical projection names the adapter declares it targets
    /// (informational — the actual adapted projections live in
    /// the per-layer dictionary).
    /// </summary>
    IReadOnlyList<string> TargetModules { get; }

    /// <summary>
    /// Looks up the (A, B) factor pair for <paramref name="layerIndex"/> /
    /// <paramref name="projName"/>. Returns <c>null</c> when this adapter
    /// does not adapt that projection at that layer.
    /// </summary>
    /// <param name="layerIndex">Zero-based transformer layer index.</param>
    /// <param name="projName">
    /// Canonical projection name: <c>q_proj</c>, <c>k_proj</c>,
    /// <c>v_proj</c>, <c>o_proj</c>, <c>gate_proj</c>, <c>up_proj</c>,
    /// <c>down_proj</c>.
    /// </param>
    /// <returns>
    /// <see cref="LoraLayerWeights"/> when the adapter targets this site,
    /// otherwise <c>null</c>.
    /// </returns>
    LoraLayerWeights? GetLayerWeights(int layerIndex, string projName);

    /// <summary>
    /// Verifies the adapter's per-projection input/output dimensions are
    /// compatible with <paramref name="baseConfig"/>. Returns <c>true</c>
    /// when the adapter can be applied to a model built from that config.
    /// </summary>
    bool IsCompatible(ModelConfig baseConfig);
}

/// <summary>
/// Per-projection LoRA factor pair. Both buffers are row-major F32 in
/// 64-byte-aligned native memory owned by the parent <see cref="ILoraAdapter"/>.
/// Layout matches dotLLM's standard "weight as [output, input]" convention so
/// the existing CPU MatMul kernels consume them directly without transposes.
/// </summary>
/// <remarks>
/// <para>
/// Mapping to / from PEFT (<c>peft.tuners.lora.LoraLayer</c>): PEFT's
/// <c>lora_A.weight</c> has shape <c>[r, in_features]</c> — that is dotLLM's
/// <see cref="BHandle"/> buffer (the down-projection). PEFT's
/// <c>lora_B.weight</c> has shape <c>[out_features, r]</c> — that is dotLLM's
/// <see cref="AHandle"/> buffer (the up-projection). The PEFT loader swaps
/// roles when copying so the runtime kernel sees a uniform layout.
/// </para>
/// <para>
/// Math: <c>y += scale × (x · B) · A</c> where
/// <c>tmp[t, r] = sum_i x[t, i] · B[r, i]</c> and
/// <c>delta[t, o] = sum_r A[o, r] · tmp[t, r]</c>.
/// </para>
/// </remarks>
/// <param name="AHandle">
/// Up-projection pointer — F32 row-major <c>[OutputDim, Rank]</c>.
/// </param>
/// <param name="BHandle">
/// Down-projection pointer — F32 row-major <c>[Rank, InputDim]</c>.
/// </param>
/// <param name="InputDim">Input dimension of the projection (d_in).</param>
/// <param name="OutputDim">Output dimension of the projection (d_out).</param>
public readonly record struct LoraLayerWeights(
    nint AHandle,
    nint BHandle,
    int InputDim,
    int OutputDim);
