using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Core.Lora;

/// <summary>
/// Default <see cref="ILoraAdapter"/> implementation. Owns native-aligned
/// F32 buffers for every <c>(layerIndex, projName) → (A, B)</c> pair declared
/// by the loader and frees them in <see cref="Dispose"/>.
/// </summary>
/// <remarks>
/// <para>
/// All A/B buffers are allocated via
/// <see cref="NativeMemory.AlignedAlloc(nuint, nuint)"/> with 64-byte
/// alignment so AVX-512-friendly kernels can consume them without copies.
/// The class is sealed so the dispose chain is unambiguous; callers compose
/// adapters via <see cref="ILoraAdapter"/> rather than subclassing.
/// </para>
/// <para>
/// Construction is two-stage: callers <c>new</c> the adapter with its
/// metadata, then call <see cref="AddLayerWeights"/> for each loaded
/// <c>(layerIndex, projName)</c> tensor pair. The loader is responsible for
/// shape-validation against the base <see cref="ModelConfig"/> before
/// adding — see <see cref="IsCompatible"/> for the acceptance criteria.
/// </para>
/// </remarks>
public sealed unsafe class LoraAdapter : ILoraAdapter
{
    private readonly Dictionary<(int Layer, string Proj), LoraLayerWeights> _layers;
    private readonly object _lock = new();
    private bool _disposed;

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public int Rank { get; }

    /// <inheritdoc/>
    public float Alpha { get; }

    /// <inheritdoc/>
    public IReadOnlyList<string> TargetModules { get; }

    /// <summary>
    /// Per-projection layer weights. Exposes the underlying dictionary as a
    /// read-only view for diagnostics; runtime lookups should use
    /// <see cref="GetLayerWeights"/>.
    /// </summary>
    public IReadOnlyDictionary<(int Layer, string Proj), LoraLayerWeights> LayerWeights => _layers;

    /// <summary>
    /// Creates a new adapter shell. Per-layer factors are added with
    /// <see cref="AddLayerWeights"/>.
    /// </summary>
    public LoraAdapter(string name, int rank, float alpha, IReadOnlyList<string> targetModules)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        if (rank <= 0)
            throw new ArgumentOutOfRangeException(nameof(rank), rank, "Rank must be positive.");
        ArgumentNullException.ThrowIfNull(targetModules);

        Name = name;
        Rank = rank;
        Alpha = alpha;
        TargetModules = targetModules;
        _layers = new Dictionary<(int, string), LoraLayerWeights>();
    }

    /// <summary>
    /// Records a freshly-allocated <c>(A, B)</c> pair for
    /// <paramref name="layerIndex"/> / <paramref name="projName"/>. Both
    /// pointers are taken over by the adapter — callers MUST allocate them
    /// via <see cref="AllocAligned"/> (or the equivalent
    /// <c>NativeMemory.AlignedAlloc(_, 64)</c>) so <see cref="Dispose"/> can
    /// safely free them.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown when an entry already exists for that <c>(layer, proj)</c>
    /// key (PEFT shipped duplicate <c>lora_A</c> / <c>lora_B</c> tensors).
    /// </exception>
    public void AddLayerWeights(int layerIndex, string projName, LoraLayerWeights weights)
    {
        ArgumentException.ThrowIfNullOrEmpty(projName);
        if (layerIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(layerIndex), layerIndex, "Layer index must be non-negative.");

        lock (_lock)
        {
            if (!_layers.TryAdd((layerIndex, projName), weights))
            {
                throw new InvalidOperationException(
                    $"LoRA adapter '{Name}' already has weights for layer {layerIndex} projection '{projName}'.");
            }
        }
    }

    /// <inheritdoc/>
    public LoraLayerWeights? GetLayerWeights(int layerIndex, string projName)
    {
        if (string.IsNullOrEmpty(projName)) return null;
        return _layers.TryGetValue((layerIndex, projName), out var w) ? w : null;
    }

    /// <inheritdoc/>
    public bool IsCompatible(ModelConfig baseConfig)
    {
        ArgumentNullException.ThrowIfNull(baseConfig);

        int qOut = baseConfig.NumAttentionHeads * baseConfig.HeadDim;
        int kvOut = baseConfig.NumKvHeads * baseConfig.HeadDim;

        foreach (var ((layer, proj), w) in _layers)
        {
            if ((uint)layer >= (uint)baseConfig.NumLayers)
                return false;

            // Validate the projection's input/output dimensions match the
            // base model's per-projection shape.
            switch (proj)
            {
                case "q_proj":
                    if (w.InputDim != baseConfig.HiddenSize || w.OutputDim != qOut) return false;
                    break;
                case "k_proj":
                case "v_proj":
                    if (w.InputDim != baseConfig.HiddenSize || w.OutputDim != kvOut) return false;
                    break;
                case "o_proj":
                    if (w.InputDim != qOut || w.OutputDim != baseConfig.HiddenSize) return false;
                    break;
                case "gate_proj":
                case "up_proj":
                    if (w.InputDim != baseConfig.HiddenSize || w.OutputDim != baseConfig.IntermediateSize) return false;
                    break;
                case "down_proj":
                    if (w.InputDim != baseConfig.IntermediateSize || w.OutputDim != baseConfig.HiddenSize) return false;
                    break;
                case "q_a_proj":
                    if (baseConfig.MlaConfig is not { } qAMla || qAMla.QLoraRank <= 0) return false;
                    if (w.InputDim != baseConfig.HiddenSize || w.OutputDim != qAMla.QLoraRank) return false;
                    break;
                case "q_b_proj":
                    if (baseConfig.MlaConfig is not { } qBMla || qBMla.QLoraRank <= 0) return false;
                    if (w.InputDim != qBMla.QLoraRank ||
                        w.OutputDim != baseConfig.NumAttentionHeads * (qBMla.QkNopeHeadDim + qBMla.QkRopeHeadDim)) return false;
                    break;
                case "kv_a_proj_with_mqa":
                    if (baseConfig.MlaConfig is not { } kvAMla) return false;
                    if (w.InputDim != baseConfig.HiddenSize ||
                        w.OutputDim != kvAMla.KvLoraRank + kvAMla.QkRopeHeadDim) return false;
                    break;
                case "kv_b_proj":
                    if (baseConfig.MlaConfig is not { } kvBMla) return false;
                    if (w.InputDim != kvBMla.KvLoraRank ||
                        w.OutputDim != baseConfig.NumAttentionHeads * (kvBMla.QkNopeHeadDim + kvBMla.VHeadDim)) return false;
                    break;
                default:
                    if (!TryValidatePerExpertMoeProjection(proj, w, baseConfig))
                        return false;
                    break;
            }
        }
        return true;
    }

    private static bool TryValidatePerExpertMoeProjection(
        string proj,
        LoraLayerWeights weights,
        ModelConfig baseConfig)
    {
        const string prefix = "mlp.experts.";
        if (!proj.StartsWith(prefix, StringComparison.Ordinal)) return false;
        if (baseConfig.Moe is not { } moe) return false;

        ReadOnlySpan<char> rest = proj.AsSpan(prefix.Length);
        int dot = rest.IndexOf('.');
        if (dot <= 0) return false;
        if (!int.TryParse(rest[..dot], out int expert) || (uint)expert >= (uint)moe.NumExperts)
            return false;

        string projection = rest[(dot + 1)..].ToString();
        int intermediate = moe.MoeIntermediateSize > 0 ? moe.MoeIntermediateSize : baseConfig.IntermediateSize;
        return projection switch
        {
            "gate_proj" or "up_proj" =>
                weights.InputDim == baseConfig.HiddenSize && weights.OutputDim == intermediate,
            "down_proj" =>
                weights.InputDim == intermediate && weights.OutputDim == baseConfig.HiddenSize,
            _ => false,
        };
    }

    /// <summary>
    /// Allocates a 64-byte-aligned native F32 buffer of <paramref name="elementCount"/>
    /// elements. Caller is responsible for transferring ownership to a
    /// <see cref="LoraAdapter"/> via <see cref="AddLayerWeights"/> (or freeing
    /// it directly with <see cref="NativeMemory.AlignedFree"/>).
    /// </summary>
    public static nint AllocAligned(long elementCount)
    {
        if (elementCount < 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount), elementCount, "Element count must be non-negative.");
        if (elementCount == 0) return 0;
        return (nint)NativeMemory.AlignedAlloc((nuint)(elementCount * sizeof(float)), 64);
    }

    /// <summary>
    /// Allocates a 64-byte-aligned native byte buffer of <paramref name="byteCount"/>
    /// bytes. Used for non-F32 LoRA weights — Q8_0, F16, BF16 — where the
    /// element size is not 4 bytes. Caller transfers ownership to a
    /// <see cref="LoraAdapter"/> via <see cref="AddLayerWeights"/>.
    /// </summary>
    public static nint AllocAlignedBytes(long byteCount)
    {
        if (byteCount < 0)
            throw new ArgumentOutOfRangeException(nameof(byteCount), byteCount, "Byte count must be non-negative.");
        if (byteCount == 0) return 0;
        return (nint)NativeMemory.AlignedAlloc((nuint)byteCount, 64);
    }

    /// <summary>
    /// Q8_0 block size in bytes — 2-byte F16 scale + 32 sbytes. Matches the
    /// GGUF on-disk Q8_0 layout and the
    /// <c>DotLLM.Cpu.Kernels.MatMul</c> kernel format.
    /// </summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Number of F32 elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = 32;

    /// <summary>
    /// Returns the Q8_0 byte size for a row of <paramref name="elements"/>
    /// F32 values. Throws when the row is not a multiple of
    /// <see cref="Q8_0GroupSize"/>.
    /// </summary>
    public static long Q8_0ByteSize(long elements)
    {
        if (elements < 0)
            throw new ArgumentOutOfRangeException(nameof(elements), elements, "Element count must be non-negative.");
        if (elements % Q8_0GroupSize != 0)
            throw new ArgumentException(
                $"Q8_0 requires element count multiple of {Q8_0GroupSize}, got {elements}.",
                nameof(elements));
        return (elements / Q8_0GroupSize) * Q8_0BlockBytes;
    }

    /// <summary>
    /// Phase 4d.6 — set when the runtime has eagerly built the
    /// <see cref="LoraLayerWeights.ATransposedHandle"/> for every layer-proj
    /// pair this adapter targets. Lets callers (e.g.
    /// <c>DotLLM.Cpu.Kernels.LoraStage2.PrewarmAdapter</c>) early-exit on
    /// repeat invocation — important because the prewarm hook lives on the
    /// per-token <c>IModel.Forward(...)</c> path and any per-call work would
    /// dominate decode-batch=1 throughput.
    /// </summary>
    public bool IsStage2FastPathPrewarmed { get; private set; }

    /// <summary>
    /// Sets <see cref="IsStage2FastPathPrewarmed"/> to <c>true</c>. Called
    /// by the runtime after a successful prewarm walk completes.
    /// </summary>
    public void MarkStage2FastPathPrewarmed()
    {
        IsStage2FastPathPrewarmed = true;
    }

    /// <summary>
    /// Phase 4d.6 — installs a transposed-A handle on the cached
    /// <see cref="LoraLayerWeights"/> entry for <paramref name="layerIndex"/> /
    /// <paramref name="projName"/>. Idempotent: if a handle is already
    /// installed the new one is freed and the existing one is returned. The
    /// adapter takes ownership and frees the buffer at <see cref="Dispose"/>.
    /// </summary>
    /// <remarks>
    /// Used by the runtime LoRA dispatch to lazily materialise the
    /// outer-product stage-2 fast path's <c>[rank, outputDim]</c> A layout
    /// without forcing every adapter loader to pre-compute it.
    /// </remarks>
    /// <returns>
    /// The cached transposed-A handle (either the freshly installed one or
    /// the pre-existing one). Returns <c>0</c> when no entry exists for
    /// <c>(layerIndex, projName)</c>.
    /// </returns>
    public nint InstallATransposedHandle(int layerIndex, string projName, nint aTransposed)
    {
        ArgumentException.ThrowIfNullOrEmpty(projName);
        var key = (layerIndex, projName);

        lock (_lock)
        {
            if (!_layers.TryGetValue(key, out var w)) return 0;
            if (w.ATransposedHandle != 0)
            {
                // Already cached — discard the newly built one. This shouldn't
                // happen if callers check first, but keep the semantics clean.
                if (aTransposed != 0 && aTransposed != w.ATransposedHandle)
                    NativeMemory.AlignedFree((void*)aTransposed);
                return w.ATransposedHandle;
            }

            var updated = w with { ATransposedHandle = aTransposed };
            _layers[key] = updated;
            return aTransposed;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        lock (_lock)
        {
            foreach (var w in _layers.Values)
            {
                if (w.AHandle != 0) NativeMemory.AlignedFree((void*)w.AHandle);
                if (w.BHandle != 0) NativeMemory.AlignedFree((void*)w.BHandle);
                if (w.ATransposedHandle != 0) NativeMemory.AlignedFree((void*)w.ATransposedHandle);
            }
            _layers.Clear();
        }
    }
}
