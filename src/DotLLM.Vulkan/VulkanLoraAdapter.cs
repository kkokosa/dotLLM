using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Lora;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-resident wrapper around an <see cref="ILoraAdapter"/>: uploads each
/// per-(layer, projection) <c>(B, A)</c> factor pair to Vulkan device memory
/// once and caches the resulting <see cref="VulkanDevice.Buffer"/> handles
/// for subsequent forwards.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the CPU <see cref="LoraAdapter"/> lifecycle: created lazily the
/// first time an adapter is used with a <see cref="VulkanTransformerModel"/>,
/// cached on the model so subsequent forwards with the same adapter pay no
/// upload cost, and disposed with the model.
/// </para>
/// <para>
/// On upload the runtime scaling factor <c>scale = adapter.Alpha / adapter.Rank</c>
/// is folded into the <c>B</c> (down-projection) weight: every element is
/// pre-multiplied by <c>scale</c> so the second matmul of the LoRA delta
/// produces the already-scaled contribution. This avoids needing a new
/// "scaled add" shader to land the delta back into the projection output —
/// the existing <see cref="Kernels.AddKernel"/> can do an unscaled add of
/// <c>y + delta</c> into a scratch buffer and a copy lands the result back
/// in <c>y</c>.
/// </para>
/// <para>
/// Convention (matches <see cref="LoraLayerWeights"/>):
/// <list type="bullet">
///   <item><c>B</c> on host = <c>[rank, inputDim]</c> row-major F32 (down-projection).</item>
///   <item><c>A</c> on host = <c>[outputDim, rank]</c> row-major F32 (up-projection).</item>
/// </list>
/// On device both buffers are F32 row-major with the same shape — scaling
/// is folded into <c>B</c> only. Quantised LoRA weights (F16 / BF16 / Q8_0)
/// are deferred to Phase 4d.
/// </para>
/// </remarks>
internal sealed class VulkanLoraAdapter : IDisposable
{
    /// <summary>Per-(layer, projection) device buffers + dimensions.</summary>
    public readonly record struct LayerBuffers(
        VulkanDevice.Buffer B,
        VulkanDevice.Buffer A,
        int InputDim,
        int OutputDim,
        int Rank);

    private readonly Dictionary<(int Layer, string Proj), LayerBuffers> _layers;
    private bool _disposed;

    /// <summary>Source adapter — kept for diagnostics + identity equality.</summary>
    public ILoraAdapter Source { get; }

    /// <summary>LoRA rank — used to size shared scratch buffers.</summary>
    public int Rank { get; }

    /// <summary>Pre-folded scale = Alpha / Rank.</summary>
    public float Scale { get; }

    /// <summary>Largest output-dim across every uploaded entry; sized scratch users read this.</summary>
    public int MaxOutputDim { get; }

    private VulkanLoraAdapter(
        ILoraAdapter source, int rank, float scale, int maxOutputDim,
        Dictionary<(int Layer, string Proj), LayerBuffers> layers)
    {
        Source = source;
        Rank = rank;
        Scale = scale;
        MaxOutputDim = maxOutputDim;
        _layers = layers;
    }

    /// <summary>
    /// Uploads every <c>(layer, proj)</c> entry from <paramref name="adapter"/>
    /// to <paramref name="device"/> as a pair of F32 row-major buffers, with
    /// <c>alpha / rank</c> pre-folded into the <c>B</c> weight.
    /// </summary>
    public static unsafe VulkanLoraAdapter Upload(VulkanDevice device, ILoraAdapter adapter)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(adapter);
        if (adapter.Rank <= 0)
            throw new ArgumentException("LoRA adapter rank must be positive.", nameof(adapter));

        int rank = adapter.Rank;
        float scale = adapter.Alpha / adapter.Rank;

        var layers = new Dictionary<(int, string), LayerBuffers>();
        int maxOutputDim = 0;

        try
        {
            // Probe the adapter for every (layer, proj) entry.
            // ILoraAdapter does not expose its dictionary directly, so we
            // walk the canonical projection names per layer up to a
            // generous cap. For the concrete LoraAdapter the cost is one
            // dictionary probe per (layer, name) — cheap. We stop scanning
            // when we've gone two layers past the last populated layer
            // (covers TinyLlama where only the last few layers are
            // adapted, plus typical full-coverage adapters).
            string[] canonicalNames = ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"];

            // If the adapter is the concrete LoraAdapter we can iterate the
            // dictionary directly — exact, no scan cap.
            if (adapter is LoraAdapter concrete)
            {
                foreach (var kvp in concrete.LayerWeights)
                {
                    string proj = kvp.Key.Proj;
                    if (!IsStandardTransformerProj(proj)) continue;

                    var lb = UploadOne(device, kvp.Value, rank, scale);
                    layers[kvp.Key] = lb;
                    if (lb.OutputDim > maxOutputDim) maxOutputDim = lb.OutputDim;
                }
            }
            else
            {
                int consecutiveEmpty = 0;
                bool everPopulated = false;
                for (int layer = 0; layer < 4096; layer++)
                {
                    bool any = false;
                    foreach (var name in canonicalNames)
                    {
                        var w = adapter.GetLayerWeights(layer, name);
                        if (w is null) continue;

                        var lb = UploadOne(device, w.Value, rank, scale);
                        layers[(layer, name)] = lb;
                        if (lb.OutputDim > maxOutputDim) maxOutputDim = lb.OutputDim;
                        any = true;
                    }
                    if (any) { consecutiveEmpty = 0; everPopulated = true; }
                    else if (everPopulated)
                    {
                        consecutiveEmpty++;
                        if (consecutiveEmpty >= 2) break;
                    }
                }
            }
        }
        catch
        {
            foreach (var lb in layers.Values)
            {
                lb.B.Dispose();
                lb.A.Dispose();
            }
            throw;
        }

        return new VulkanLoraAdapter(adapter, rank, scale, maxOutputDim, layers);
    }

    private static unsafe LayerBuffers UploadOne(
        VulkanDevice device, LoraLayerWeights w, int rank, float scale)
    {
        long bElems = (long)rank * w.InputDim;
        long aElems = (long)w.OutputDim * rank;
        long bBytes = bElems * sizeof(float);
        long aBytes = aElems * sizeof(float);

        // Scaffold path: VulkanDevice.Allocate returns host-visible
        // host-coherent storage. A real Vulkan perf pass would migrate
        // adapter weights to device-local + staging, but the scaffold has
        // not yet introduced AllocateDeviceLocal — keep adapter upload on
        // the host-mapped path used by every other weight buffer.
        var bBuf = device.Allocate(bBytes);
        VulkanDevice.Buffer? aBuf = null;
        try
        {
            // Stage B into an F32 scratch with scale pre-folded. When the
            // source dtype is non-F32 (F16 / BF16 / Q8_0 — Phase 4d.1 +
            // Phase 4d.5 / Gap 1) we dequantise on the host once at upload
            // time so the existing fused F32 device-side kernel runs
            // unchanged. Adapter upload is one-shot per adapter (cached in
            // VulkanLoraAdapterCache), so a per-element multiply or per-block
            // Q8_0 dequant here has zero impact on inference throughput.
            var pool = ArrayPool<float>.Shared;
            float[] scratch = pool.Rent((int)bElems);
            try
            {
                var dst = new Span<float>(scratch, 0, (int)bElems);
                DequantAndScale(
                    src: (void*)w.BHandle,
                    dtype: w.WeightDType,
                    elementsPerRow: w.InputDim,
                    rows: rank,
                    scale: scale,
                    dst: dst);
                device.Upload(new ReadOnlySpan<float>(scratch, 0, (int)bElems), bBuf);
            }
            finally
            {
                pool.Return(scratch);
            }

            aBuf = device.Allocate(aBytes);
            // A: convert to F32 in a scratch and upload. Scale is folded
            // into B only, so A goes verbatim (Q8_0 is invalid for A per
            // the LoraWeightDType contract — A's contracted axis is rank,
            // below the Q8_0 block size of 32 — so we only handle
            // F32 / F16 / BF16 here).
            var aDType = w.ResolvedAWeightDType;
            if (aDType == LoraWeightDType.F32)
            {
                var aSrc = new ReadOnlySpan<float>((void*)w.AHandle, (int)aElems);
                device.Upload(aSrc, aBuf);
            }
            else
            {
                float[] aScratch = pool.Rent((int)aElems);
                try
                {
                    var dst = new Span<float>(aScratch, 0, (int)aElems);
                    DequantAndScale(
                        src: (void*)w.AHandle,
                        dtype: aDType,
                        elementsPerRow: rank, // A has shape [outputDim, rank]; row layout doesn't matter for the unscaled F32/F16/BF16 dequant
                        rows: w.OutputDim,
                        scale: 1.0f,
                        dst: dst);
                    device.Upload(new ReadOnlySpan<float>(aScratch, 0, (int)aElems), aBuf);
                }
                finally
                {
                    pool.Return(aScratch);
                }
            }
        }
        catch
        {
            bBuf.Dispose();
            aBuf?.Dispose();
            throw;
        }

        return new LayerBuffers(bBuf, aBuf, w.InputDim, w.OutputDim, rank);
    }

    /// <summary>
    /// Host-side dequant of a LoRA factor buffer into F32, multiplying every
    /// element by <paramref name="scale"/>. F32 is a straight copy + scale;
    /// F16 / BF16 use the standard <see cref="TensorPrimitives"/> / scalar
    /// paths; Q8_0 routes through <see cref="LoraDelta.DequantizeRowToF32"/>
    /// row-by-row (Q8_0 is row-blocked so a contiguous range of rows is a
    /// contiguous range of bytes).
    /// </summary>
    /// <remarks>
    /// Adapter upload is one-shot per adapter (cached in
    /// <see cref="VulkanLoraAdapterCache"/>) so we deliberately don't
    /// SIMD-tune this path — clarity wins over throughput. The Q8_0 case is
    /// the only one that exists post-Phase-4d.5 / Gap 1; F16 / BF16 / F32
    /// paths existed pre-Phase-4d.5 but were silently routed through a
    /// "read as F32" path that would have produced junk values for any
    /// non-F32 adapter. This method makes those paths correct as well.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DequantAndScale(
        void* src, LoraWeightDType dtype, int elementsPerRow, int rows, float scale, Span<float> dst)
    {
        long count = (long)rows * elementsPerRow;
        if (dst.Length < count)
            throw new ArgumentException(
                $"Dequant destination span too small: {dst.Length} < {count}.", nameof(dst));

        switch (dtype)
        {
            case LoraWeightDType.F32:
                {
                    var srcSpan = new ReadOnlySpan<float>(src, (int)count);
                    if (scale == 1.0f)
                    {
                        srcSpan.CopyTo(dst);
                    }
                    else
                    {
                        TensorPrimitives.Multiply(srcSpan, scale, dst);
                    }
                    break;
                }
            case LoraWeightDType.F16:
                {
                    var srcSpan = new ReadOnlySpan<Half>(src, (int)count);
                    TensorPrimitives.ConvertToSingle(srcSpan, dst);
                    if (scale != 1.0f)
                    {
                        TensorPrimitives.Multiply((ReadOnlySpan<float>)dst, scale, dst);
                    }
                    break;
                }
            case LoraWeightDType.BF16:
                {
                    byte* p = (byte*)src;
                    for (long i = 0; i < count; i++)
                    {
                        ushort raw = BinaryPrimitives.ReadUInt16LittleEndian(
                            new ReadOnlySpan<byte>(p + i * 2, 2));
                        uint asF32 = (uint)raw << 16;
                        dst[(int)i] = BitConverter.UInt32BitsToSingle(asF32) * scale;
                    }
                    break;
                }
            case LoraWeightDType.Q8_0:
                {
                    if (elementsPerRow % 32 != 0)
                        throw new ArgumentException(
                            $"Q8_0 LoRA factor requires elementsPerRow multiple of 32, got {elementsPerRow}.",
                            nameof(elementsPerRow));
                    byte* srcQ8 = (byte*)src;
                    int rowBytes = (elementsPerRow / 32) * LoraAdapter.Q8_0BlockBytes;
                    fixed (float* dstPtr = dst)
                    {
                        for (int r = 0; r < rows; r++)
                        {
                            LoraDelta.DequantizeRowToF32(
                                srcQ8 + (long)r * rowBytes,
                                dstPtr + (long)r * elementsPerRow,
                                elementsPerRow);
                        }
                    }
                    // Fold scale once over the dequantised rows.
                    if (scale != 1.0f)
                    {
                        TensorPrimitives.Multiply((ReadOnlySpan<float>)dst[..(int)count], scale, dst[..(int)count]);
                    }
                    break;
                }
            default:
                throw new NotSupportedException(
                    $"LoRA weight dtype {dtype} is not supported by VulkanLoraAdapter upload.");
        }
    }


    private static bool IsStandardTransformerProj(string proj) =>
        proj is "q_proj" or "k_proj" or "v_proj" or "o_proj"
             or "gate_proj" or "up_proj" or "down_proj";

    /// <summary>
    /// Looks up the device buffers for <paramref name="layerIndex"/> /
    /// <paramref name="projName"/>. Returns <c>null</c> when the adapter
    /// does not target that site (no LoRA delta to apply).
    /// </summary>
    public LayerBuffers? Get(int layerIndex, string projName)
    {
        return _layers.TryGetValue((layerIndex, projName), out var lb) ? lb : null;
    }

    /// <summary>Iterates every uploaded entry — diagnostics-only.</summary>
    public IReadOnlyDictionary<(int Layer, string Proj), LayerBuffers> Layers => _layers;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var lb in _layers.Values)
        {
            lb.B.Dispose();
            lb.A.Dispose();
        }
        _layers.Clear();
    }
}

/// <summary>
/// Per-model device-side LoRA cache. Maps an <see cref="ILoraAdapter"/> to
/// its uploaded <see cref="VulkanLoraAdapter"/>, so the same forward-time
/// adapter is uploaded once and reused on every subsequent call.
/// </summary>
/// <remarks>
/// Caching is by reference identity on <see cref="ILoraAdapter"/> — same
/// instance, same upload. If the host disposes the source adapter without
/// removing it from the cache, the next forward against that adapter will
/// throw at the lookup; the constraint is documented at the model level.
/// </remarks>
internal sealed class VulkanLoraAdapterCache : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly ConcurrentDictionary<ILoraAdapter, VulkanLoraAdapter> _entries = new();
    private readonly object _uploadLock = new();
    private bool _disposed;

    public VulkanLoraAdapterCache(VulkanDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
    }

    /// <summary>Resolves an upload for <paramref name="adapter"/>, uploading on first use.</summary>
    public VulkanLoraAdapter GetOrAdd(ILoraAdapter adapter)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_entries.TryGetValue(adapter, out var existing)) return existing;

        // Serialise uploads — the upload path uses a transient ArrayPool
        // scratch + per-buffer Allocate that we'd rather not race on.
        lock (_uploadLock)
        {
            if (_entries.TryGetValue(adapter, out existing)) return existing;
            var fresh = VulkanLoraAdapter.Upload(_device, adapter);
            if (!_entries.TryAdd(adapter, fresh))
            {
                fresh.Dispose();
                return _entries[adapter];
            }
            return fresh;
        }
    }

    /// <summary>Number of cached entries — diagnostics-only.</summary>
    public int Count => _entries.Count;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var v in _entries.Values) v.Dispose();
        _entries.Clear();
    }
}
