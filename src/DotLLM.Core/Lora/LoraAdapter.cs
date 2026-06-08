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
            // base model's per-projection shape. MLA-specific projections
            // (q_a_proj, kv_a_proj_with_mqa, etc.) are deferred — only the
            // standard q/k/v/o + gate/up/down sites are covered today.
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
                default:
                    // Unknown projection name — fail compatibility check rather than silently
                    // accept (the runtime would not apply it anyway).
                    return false;
            }
        }
        return true;
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
            }
            _layers.Clear();
        }
    }
}
