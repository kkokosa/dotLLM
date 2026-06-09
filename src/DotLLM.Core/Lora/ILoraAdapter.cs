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
/// Up-projection pointer — row-major <c>[OutputDim, Rank]</c>.
/// </param>
/// <param name="BHandle">
/// Down-projection pointer — row-major <c>[Rank, InputDim]</c>.
/// </param>
/// <param name="InputDim">Input dimension of the projection (d_in).</param>
/// <param name="OutputDim">Output dimension of the projection (d_out).</param>
/// <param name="WeightDType">
/// Element dtype of the B (down-projection) buffer. F32 (default — backward
/// compatible), F16, BF16, or Q8_0 (Phase 4d.4). When the dtype is symmetric
/// (F32/F16/BF16) <see cref="AWeightDType"/> is left at its default and A is
/// implicitly the same dtype. When the dtype is Q8_0 (only valid for B), A
/// is implicitly F16 — see <see cref="LoraWeightDType.Q8_0"/> for why.
/// </param>
/// <param name="AWeightDType">
/// Optional explicit dtype for the A (up-projection) buffer. Defaults to
/// <see cref="LoraWeightDType.F32"/> meaning "use <see cref="WeightDType"/>
/// for both" — the legacy path. Set explicitly when storing A in a different
/// dtype than B (e.g. Q8_0 B + F16 A).
/// </param>
/// <param name="ATransposedHandle">
/// Phase 4d.6 — optional cache of <see cref="AHandle"/> transposed to
/// <c>[rank, OutputDim]</c> row-major F32 layout. When non-zero the
/// stage-2 outer-product fast path
/// (<c>DotLLM.Cpu.Kernels.LoraStage2.ApplyF32_R16</c>) consumes it
/// directly; when zero the runtime either lazy-builds it (via
/// <see cref="ILoraAdapter"/> implementation hooks) or falls back to the
/// per-token GEMV stage-2 path. Loaders may leave this <c>0</c> — the
/// adapter is responsible for materialising it on first use. The buffer,
/// when present, is freed by the adapter at <see cref="IDisposable.Dispose"/>.
/// </param>
public readonly record struct LoraLayerWeights(
    nint AHandle,
    nint BHandle,
    int InputDim,
    int OutputDim,
    LoraWeightDType WeightDType = LoraWeightDType.F32,
    LoraWeightDType AWeightDType = LoraWeightDType.F32,
    nint ATransposedHandle = 0)
{
    /// <summary>
    /// Effective A-buffer dtype. Encodes the "implicit symmetric" rule:
    /// when <see cref="AWeightDType"/> is F32 (the default) the actual
    /// A dtype is whatever <see cref="WeightDType"/> is — except for
    /// <see cref="LoraWeightDType.Q8_0"/> which is B-only and implies F16
    /// for A. Set <see cref="AWeightDType"/> explicitly to override.
    /// </summary>
    public LoraWeightDType ResolvedAWeightDType =>
        AWeightDType != LoraWeightDType.F32 ? AWeightDType
        : WeightDType == LoraWeightDType.Q8_0 ? LoraWeightDType.F16
        : WeightDType;
}

/// <summary>
/// LoRA adapter weight dtype. Most PEFT trainers ship F16; some ship BF16
/// or F32. dotLLM stores adapter buffers in their native dtype to halve
/// memory for typical adapters and avoids an unnecessary up-cast on load.
/// </summary>
/// <remarks>
/// <para>
/// Phase 4d.4 added <see cref="Q8_0"/> for the down-projection (B) buffer
/// only. The asymmetry is deliberate: B has shape <c>[rank, inputDim]</c>
/// where <c>inputDim</c> is the projection input (always a multiple of 32
/// for transformer linear layers), so each B row is a natural Q8_0 row.
/// A has shape <c>[outputDim, rank]</c> where the contracted axis is
/// <c>rank</c> (8–64 typical) — too short for a 32-element Q8_0 block, so
/// A stays F16 / BF16 / F32. This is the layout exercised by the
/// quantised-LoRA bench (see <c>docs/LORA.md</c>).
/// </para>
/// <para>
/// For a Q8_0 B buffer the storage layout is <c>(inputDim / 32)</c>
/// blocks per row, each block 34 bytes (2-byte F16 scale + 32 sbytes),
/// matching the on-disk GGUF Q8_0 layout. Quantisation happens once at
/// adapter load via the helper on <see cref="LoraAdapter"/>.
/// </para>
/// </remarks>
public enum LoraWeightDType : byte
{
    /// <summary>32-bit single-precision float (default).</summary>
    F32 = 0,
    /// <summary>16-bit IEEE half-precision float (PEFT default).</summary>
    F16 = 1,
    /// <summary>16-bit bfloat16 (top 16 bits of an F32).</summary>
    BF16 = 2,
    /// <summary>
    /// Q8_0 block-quantised storage — only valid for the B (down-projection)
    /// buffer, which has rows of length <c>inputDim</c> (multiple of 32).
    /// Each row is <c>(inputDim / 32)</c> blocks × 34 bytes = ~1.0625 B / element.
    /// </summary>
    Q8_0 = 3,
}
