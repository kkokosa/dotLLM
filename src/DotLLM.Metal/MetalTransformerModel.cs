using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Metal.Interop;
using DotLLM.Metal.Kernels;
using DotLLM.Metal.Weights;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Convert = DotLLM.Metal.Kernels.Convert;

namespace DotLLM.Metal;

/// <summary>
/// Metal implementation of <see cref="IModel"/>.
/// </summary>
public sealed unsafe class MetalTransformerModel : IModel
{
    private readonly MetalWeights _weights;
    private readonly MetalForwardState _state;
    private readonly MetalContext _context;
    private readonly GgufFile _gguf;
    private readonly int _deviceId;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _ropeType;
    private bool _disposed;

    private MetalTransformerModel(MetalWeights weights, MetalForwardState state, MetalContext context, GgufFile gguf, int deviceId, float ropeTheta, int ropeDim, int ropeType, ModelConfig config)
    {
        _weights = weights;
        _state = state;
        _context = context;
        _gguf = gguf;
        _deviceId = deviceId;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _ropeType = ropeType;
        Config = config;
    }

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    /// <summary>
    /// Provides access to the underlying <see cref="MetalContext"/> associated with this model.
    /// </summary>
    public MetalContext Context => _context;

    /// <summary>
    /// Loads a transformer model onto the GPU from an opened GGUF file.
    /// </summary>
    /// <param name="gguf">Opened GGUF file (must remain alive for model lifetime).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="strategy"></param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. If null, auto-detects from assembly location.</param>
    public static MetalTransformerModel LoadFromGguf(
        GgufFile gguf,
        ModelConfig config,
        IWeightLoadStrategy strategy,
        int deviceId = 0,
        string? ptxDir = null)
    {
        // Initialize CUDA
        var context = new MetalContext();

        // Resolve PTX directory
        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");

        // Check VRAM before loading — warn if model likely exceeds available memory.
        // Estimate: sum of quantized byte sizes for all GGUF tensors.
        long estimatedWeightBytes = 0;
        foreach (var t in gguf.TensorsByName.Values)
        {
            int innerDim = t.Shape[0];
            long outerDim = (long)t.Shape.ElementCount / innerDim;
            estimatedWeightBytes += Cpu.Kernels.Dequantize.RowByteSize(innerDim, t.QuantizationType) * outerDim;
        }

        long totalAvailableMemoryBytes = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;

        string? vramWarning = null;
        if (totalAvailableMemoryBytes > 0 && estimatedWeightBytes > totalAvailableMemoryBytes)
        {
            long modelMb = estimatedWeightBytes / (1024 * 1024);
            long totalMb = totalAvailableMemoryBytes / (1024 * 1024);
            vramWarning = $"Model weights (~{modelMb} MB) exceed available RAM ({totalMb} MB). " +
                          $"Performance will be degraded due to PCIe memory paging. " +
                          $"Consider a smaller model or quantization format.";
        }

        // Upload weights to GPU
        var weights = MetalWeights.LoadFromGguf(gguf, context, strategy);

        // Create scratch buffers
        MetalForwardState state = new MetalForwardState(
            context,
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize);

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        int ropeType = (int)(config.RoPEConfig?.Type ?? RoPEType.Norm);

        return new MetalTransformerModel(weights, state, context, gguf, deviceId, ropeTheta, ropeDim, ropeType, config);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        int seqLen          = tokenIds.Length;
        int hiddenSize      = Config.HiddenSize;
        int numHeads        = Config.NumAttentionHeads;
        int numKvHeads      = Config.NumKvHeads;
        int headDim         = Config.HeadDim;
        int vocabSize       = Config.VocabSize;
        float eps           = Config.NormEpsilon;

        _state.EnsureCapacity(seqLen);

        // 1. Copy inputs (unified memory → simple copy)
        CopyInputToState(tokenIds, _state.TokenIds);
        CopyInputToState(positions,  _state.Positions);

        // ── Open ONE command buffer for the entire forward pass ─────────────
        // All kernels, all intra-forward buffer copies, and all KV-cache
        // writes are encoded into this single command buffer and only
        // committed/waited at the end. Every pointer that crosses this
        // boundary MUST be GPU-resident (registered in shared_buffers) or
        // BufferCopy will reject it.
        MetalNative.BeginForward(_context.Handle);

        // 2. Embedding
        var embedding = _weights.TokenEmbedding;
        nint tablePtr = embedding.Fp16Pointer != 0
            ? embedding.Fp16Pointer       // FP16 si la stratégie l'a produit
            : embedding.QuantizedPointer; // sinon, format quantisé du mmap
        QuantizationType tableFormat = embedding.Fp16Pointer != 0
            ? QuantizationType.F16
            : embedding.QuantizedFormat;
        EmbeddingLookup.ExecuteF16Out(
            _context, tablePtr, tableFormat,
            _state.TokenIds, _state.HiddenState,
            seqLen, Config.HiddenSize, Config.VocabSize);

        // 3. Layer 0 setup — GPU blit (was CPU memcpy)
        BufferCopy(_state.Residual, _state.HiddenState, seqLen * hiddenSize);

        RmsNormF16.Execute(_context, _state.HiddenState, _weights.Layers[0].AttnNormWeight,
                           _state.NormOutput, hiddenSize, seqLen, eps);

        // 4. Boucle des layers — everything stays inside the active command buffer.
        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // Q/K/V
            Project(lw.Q, _state.NormOutput, _state.Q, seqLen);
            Project(lw.K, _state.NormOutput, _state.K, seqLen);
            Project(lw.V, _state.NormOutput, _state.V, seqLen);
            if (layer == 0) {
                Check("L0: Q after proj", _state.Q, seqLen * lw.Q.OutputDim);  // ← #4
                Check("L0: K after proj", _state.K, seqLen * lw.K.OutputDim);
                Check("L0: V after proj", _state.V, seqLen * lw.V.OutputDim);
            }

            // Biases optionnels
            if (lw.QBias != 0) BiasAddF16.Execute(_context, _state.Q, lw.QBias, lw.Q.OutputDim, seqLen);
            // … idem K, V

            // QK-norm optionnels
            if (lw.QNormWeight != 0)
                PerHeadRmsNormF16.Execute(_context, _state.Q, lw.QNormWeight, numHeads, headDim, seqLen, eps);
            if (lw.KNormWeight != 0)
                PerHeadRmsNormF16.Execute(_context, _state.K, lw.KNormWeight, numKvHeads, headDim, seqLen, eps);

            // RoPE
            RoPEF16.Execute(_context, _state.Q, _state.K, _state.Positions,
                             numHeads, numKvHeads, headDim,
                            _ropeDim, _ropeTheta, seqLen, (RoPEType)_ropeType);
            if (layer == 0) {
                Check("L0: Q after RoPE", _state.Q, seqLen * lw.Q.OutputDim); // ← #5
                Check("L0: K after RoPE", _state.K, seqLen * lw.K.OutputDim);
            }

            // Write current K/V into the persistent cache. Inside the active
            // forward this is a GPU blit (no CPU sync), so attention below
            // reads the freshly written K/V without any wait.
            if (kvCache is MetalKvCache metalCache)
            {
                int kvStride = Config.NumKvHeads * Config.HeadDim;
                var kRef = new TensorRef(positions.Length, kvStride, DType.Float16, -1, _state.K);
                var vRef = new TensorRef(positions.Length, kvStride, DType.Float16, -1, _state.V);

                metalCache.Update(kRef, vRef, positions, layer);

                if (seqLen > 1)
                {
                    // Prefill: full causal self-attention over the current tokens.
                    // We still use _state.K/_state.V (already in shared memory) to avoid
                    // a round-trip through the MTLBuffer for the prefill pass.
                    AttentionF16.Execute(_context, _state.Q, _state.K, _state.V, _state.AttnOutput,
                        seqLen, seqLen, numHeads, numKvHeads, headDim,
                        positionOffset: 0, slidingWindow: 0);
                }
                else
                {
                    // Decode: Q attends over the full accumulated cache.
                    // K/V are read directly from the persistent MTLBuffers — zero copies.
                    AttentionF16.ExecuteWithKvCache(_context, metalCache, layer,
                        _state.Q, _state.AttnOutput,
                        seqLen, numHeads, numKvHeads, headDim,
                        positionOffset: positions[0]);
                }
            }
            else
            {
                // No kvcache — full self-attention over the current tokens only.
                AttentionF16.Execute(_context, _state.Q, _state.K, _state.V, _state.AttnOutput,
                    seqLen, seqLen, numHeads, numKvHeads, headDim,
                    positionOffset: 0, slidingWindow: 0);
            }
            if (layer == 0) {
                Check("L0: AttnOutput", _state.AttnOutput, seqLen * lw.Q.OutputDim); // ← #6
            }

            // O projection
            Project(lw.O, _state.AttnOutput, _state.NormOutput, seqLen);
            if (layer == 0) {
                Check("L0: after O proj", _state.NormOutput, seqLen * hiddenSize); // ← #7
            }

            // Residual + FFN-norm
            FusedAddRmsNorm.Execute(_context, _state.Residual, _state.NormOutput,
                lw.FfnNormWeight, _state.HiddenState,
                hiddenSize, seqLen, eps);
            if (layer == 0) {
                Check("L0: after FusedAddRmsNorm (FFN-norm)", _state.HiddenState, seqLen * hiddenSize); // ← #8
                Check("L0: residual after fused-add", _state.Residual, seqLen * hiddenSize);
            }

            // GPU blit: copy HiddenState → NormOutput for the FFN input.
            BufferCopy(_state.NormOutput, _state.HiddenState, seqLen * hiddenSize);

            // FFN
            Project(lw.Gate, _state.NormOutput, _state.FfnGate, seqLen);
            Project(lw.Up,   _state.NormOutput, _state.FfnUp,   seqLen);
            SwigluF16.Execute(_context, _state.FfnGate, _state.FfnUp, _state.SiluOutput, seqLen * Config.IntermediateSize);
            if (layer == 0) {
                Check("L0: after SwiGLU", _state.SiluOutput, seqLen * Config.IntermediateSize); // ← #9
            }

            Project(lw.Down, _state.SiluOutput, _state.NormOutput, seqLen);
            if (layer == 0) {
                Check("L0: after Down proj", _state.NormOutput, seqLen * hiddenSize); // ← #10
            }

            // Residual + norme attn du layer suivant.
            // On the final layer, produce final hidden state = residual + FFN output.
            if (layer + 1 < Config.NumLayers)
            {
                ref readonly var nextLw = ref _weights.Layers[layer + 1];

                FusedAddRmsNorm.Execute(
                    _context,
                    _state.Residual,
                    _state.NormOutput,
                    nextLw.AttnNormWeight,
                    _state.HiddenState,
                    hiddenSize,
                    seqLen,
                    eps);

                BufferCopy(_state.NormOutput, _state.HiddenState, seqLen * hiddenSize);
            }
            else
            {
                AddF16.Execute(
                    _context,
                    _state.Residual,
                    _state.NormOutput,
                    _state.HiddenState,
                    (uint)(seqLen * hiddenSize));
            }
        }

        // 5. Final RmsNorm — last token only
        nint lastHidden = _state.HiddenState + (nint)((seqLen - 1) * hiddenSize * sizeof(ushort));
        RmsNormF16.Execute(_context, lastHidden, _weights.OutputNormWeight,
            _state.NormOutput, hiddenSize, 1, eps);

        // 6. LM head → vocab logits
        Project(_weights.LmHead, _state.NormOutput, _state.LogitsF16, seqLen: 1);

        // 7. F16 → F32 pour le sampler (vocabSize logits, dernier token uniquement)
        Convert.F16ToF32(_context, _state.LogitsF16, _state.LogitsF32, vocabSize);

        // ── Close the batched command buffer and wait for the GPU ───────────
        MetalNative.EndForward(_context.Handle);

        Check("after layer loop: Residual", _state.Residual, seqLen * hiddenSize); // ← #11
        Check("after final RmsNorm", _state.NormOutput, hiddenSize);
        Check("LogitsF16", _state.LogitsF16, vocabSize); // ← #13
        Check("LogitsF32", _state.LogitsF32, vocabSize, isFp16: false); // ← #14

        // 8. Wrap en ITensor (host-readable directement, pas de D2H grâce à la mémoire unifiée)
        return new MetalTensor(
            shape: new TensorShape(1, vocabSize),
            dtype: DType.Float32,
            deviceId: _deviceId,
            ptr: _state.LogitsF32,
            ownsMemory: false);
    }

    // Dans MetalTransformerModel
    private static void CopyInputToState(ReadOnlySpan<int> src, nint dst)
    {
        fixed (int* p = src)
        {
            NativeMemory.Copy(p, (void*)dst, (nuint)(src.Length * sizeof(int)));
        }
    }

    private void Memcpy(IntPtr stateResidual, IntPtr stateHiddenState, int size)
    {
        NativeMemory.Copy(
            source:      (void*)stateHiddenState,
            destination: (void*)stateResidual,
            byteCount:   (nuint)(size * sizeof(ushort)));
    }

    /// <summary>
    /// Copies <paramref name="elementCount"/> FP16 elements from <paramref name="src"/>
    /// to <paramref name="dst"/>. Inside a batched forward this is a GPU blit (no CPU sync);
    /// outside it falls back to CPU memcpy via the native helper.
    /// </summary>
    private void BufferCopy(nint dst, nint src, int elementCount)
    {
        int rc = MetalNative.BufferCopy(
            _context.Handle, dst, src, (nuint)((long)elementCount * sizeof(ushort)));
        if (rc != 0)
            throw new InvalidOperationException(
                $"MetalNative.BufferCopy failed (rc={rc}). The source/destination pointers " +
                "must be registered with the context (alloc_shared or register_buffer).");
    }

    private void Project(in LoadedWeight w, nint input, nint output, int seqLen)
    {
        int n = w.OutputDim;   // sortie
        int k = w.InputDim;    // entrée

        if (seqLen > 1)
        {
            // ── Prefill : matrice × matrice ──
            // Besoin d'inputs FP16. Si la stratégie a fourni Fp16Pointer, direct.
            // Sinon (MmapOnly + format quantisé), dequant dans DequantScratch.
            nint weightFp16 = w.Fp16Pointer;
            if (weightFp16 == 0)
            {
                DequantInto(_state.DequantScratch, w);   // dequant on-the-fly
                weightFp16 = _state.DequantScratch;
            }
            Gemm.ExecuteF16(_context, input, weightFp16, output,
                m: seqLen, n: n, k: k,
                transposeA: false, transposeB: true);
        }
        else if (w.QuantizedPointer != 0 && QuantizedGemv.Supports(w.QuantizedFormat))
        {
            // ── Decode + format quantisé supporté : GEMV quantisé ──
            QuantizedGemv.Dispatch(_context, w.QuantizedFormat, w.QuantizedPointer,
                input, output, n, k);
        }
        else
        {
            // ── Decode FP16 : MPS GEMM avec m=1 ──
            Gemm.ExecuteF16(_context, input, w.Fp16Pointer, output,
                m: 1, n: n, k: k,
                transposeA: false, transposeB: true);
        }
    }

    private void DequantInto(nint destinationFp16, in LoadedWeight weight)
    {
        if (destinationFp16 == 0)
            throw new ArgumentException("Destination FP16 pointer cannot be null.", nameof(destinationFp16));

        if (weight.QuantizedPointer == 0)
            throw new InvalidOperationException("Cannot dequantize a weight without a quantized source pointer.");

        int totalElements = checked(weight.OutputDim * weight.InputDim);
        long sourceByteLength = checked(Dequantize.RowByteSize(weight.InputDim, weight.QuantizedFormat) * weight.OutputDim);

        if (sourceByteLength > int.MaxValue)
            throw new NotSupportedException(
                $"Metal on-the-fly dequantization source is too large for Span<byte>: {sourceByteLength} bytes.");

        var source = new ReadOnlySpan<byte>((void*)weight.QuantizedPointer, (int)sourceByteLength);
        var destination = new Span<Half>((void*)destinationFp16, totalElements);

        switch (weight.QuantizedFormat)
        {
            case QuantizationType.Q8_0:
                EnsureDivisible(totalElements, 32, weight.QuantizedFormat);
                Dequant.Q8_0ToF16(_context, source, destination, totalElements / 32);
                break;

            case QuantizationType.Q4_0:
                EnsureDivisible(totalElements, 32, weight.QuantizedFormat);
                Dequant.Q4_0ToF16(_context, source, destination, totalElements / 32);
                break;

            case QuantizationType.Q5_0:
                EnsureDivisible(totalElements, 32, weight.QuantizedFormat);
                Dequant.Q5_0ToF16(_context, source, destination, totalElements / 32);
                break;

            case QuantizationType.Q4_K:
                EnsureDivisible(totalElements, 256, weight.QuantizedFormat);
                Dequant.Q4_KToF16(_context, source, destination, totalElements / 256);
                break;

            case QuantizationType.Q5_K:
                EnsureDivisible(totalElements, 256, weight.QuantizedFormat);
                Dequant.Q5_KToF16(_context, source, destination, totalElements / 256);
                break;

            case QuantizationType.Q6_K:
                EnsureDivisible(totalElements, 256, weight.QuantizedFormat);
                Dequant.Q6_KToF16(_context, source, destination, totalElements / 256);
                break;

            case QuantizationType.F16:
                NativeMemory.Copy(
                    source: (void*)weight.QuantizedPointer,
                    destination: (void*)destinationFp16,
                    byteCount: (nuint)(totalElements * sizeof(ushort)));
                break;

            default:
                throw new NotSupportedException(
                    $"On-the-fly Metal dequantization from {weight.QuantizedFormat} to FP16 is not supported.");
        }
    }

    private static void EnsureDivisible(int value, int divisor, QuantizationType format)
    {
        if (value % divisor != 0)
        {
            throw new InvalidOperationException(
                $"Cannot dequantize {format}: element count {value} must be divisible by {divisor}.");
        }
    }

    [Conditional("DEBUG")]
    private static unsafe void Check(string label, nint ptr, int count, bool isFp16 = true)
    {
        int nans = 0, infs = 0, zeros = 0;
        float min = float.MaxValue, max = float.MinValue;
        double sumAbs = 0;

        if (isFp16)
        {
            var span = new ReadOnlySpan<Half>((void*)ptr, count);
            foreach (var h in span)
            {
                float v = (float)h;
                if (float.IsNaN(v))      nans++;
                else if (float.IsInfinity(v)) infs++;
                else
                {
                    if (v == 0f) zeros++;
                    if (v < min) min = v;
                    if (v > max) max = v;
                    sumAbs += Math.Abs(v);
                }
            }
        }
        else
        {
            var span = new ReadOnlySpan<float>((void*)ptr, count);
            foreach (var v in span)
            {
                if (float.IsNaN(v))      nans++;
                else if (float.IsInfinity(v)) infs++;
                else
                {
                    if (v == 0f) zeros++;
                    if (v < min) min = v;
                    if (v > max) max = v;
                    sumAbs += Math.Abs(v);
                }
            }
        }

        bool isBad = nans > 0 || infs > 0;
        if (!isBad) return;

        string status = isBad ? "💥 BAD" : "ok";
        Console.WriteLine(
            $"[{label,-30}] {status}  nans={nans,-5} infs={infs,-5} zeros={zeros,-5} " +
            $"range=[{min,8:F3}, {max,8:F3}]  meanAbs={sumAbs/Math.Max(1, count - nans - infs):F3}");
    }

    /// <summary>
    /// Releases all resources used by the <c>MetalTransformerModel</c> instance.
    /// </summary>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            _state.Dispose();
            _weights.Dispose();
            _context.Dispose();
        }

        _disposed = true;
    }

}
