using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-accelerated transformer forward pass using CUDA. All operations execute on a single
/// CUDA stream with no host synchronization until the final logits D2H transfer.
/// Mirrors <see cref="TransformerModel"/> structure but uses cuBLAS GEMM/GEMV and custom PTX kernels.
/// </summary>
public sealed unsafe class CudaTransformerModel : IModel
{
    private readonly CudaWeights _weights;
    private readonly CudaForwardState _state;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaContext _context;
    private readonly CudaKernels _kernels;
    private readonly GgufFile _gguf;
    private readonly int _deviceId;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _ropeType;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    /// <summary>Non-null when model weights exceed available VRAM. Caller should display after loading.</summary>
    public string? VramWarning { get; }

    /// <summary>Debug: limit the number of transformer layers processed. 0 = all layers (default). -1 = skip all layers (embedding + LM head only).</summary>
    internal int DebugMaxLayers { get; set; }

    /// <summary>Debug: override RoPE type. -1 = use model's type (default).</summary>
    internal int DebugRopeTypeOverride { get; set; } = -1;

    /// <summary>Debug: skip bias add operations.</summary>
    internal bool DebugSkipBias { get; set; }

    private CudaTransformerModel(
        ModelConfig config, CudaWeights weights, CudaForwardState state,
        CudaStream stream, CudaCublasHandle cublas, CudaContext context,
        CudaKernels kernels, GgufFile gguf, int deviceId,
        float ropeTheta, int ropeDim, int ropeType, string? vramWarning)
    {
        Config = config;
        _weights = weights;
        _state = state;
        _stream = stream;
        _cublas = cublas;
        _context = context;
        _kernels = kernels;
        _gguf = gguf;
        _deviceId = deviceId;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        VramWarning = vramWarning;
        _ropeType = ropeType;
    }

    /// <summary>
    /// Loads a transformer model onto the GPU from an opened GGUF file.
    /// </summary>
    /// <param name="gguf">Opened GGUF file (must remain alive for model lifetime).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. If null, auto-detects from assembly location.</param>
    public static CudaTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config,
                                                       int deviceId = 0, string? ptxDir = null)
    {
        // Load CPU weights (mmap references only, no heavy allocation)
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);

        // Initialize CUDA
        var context = CudaContext.Create(deviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        // Resolve PTX directory
        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        // Check VRAM before loading — warn if model likely exceeds available memory.
        // Estimate: sum of quantized byte sizes for all GGUF tensors.
        long estimatedWeightBytes = 0;
        foreach (var t in gguf.TensorsByName.Values)
        {
            int innerDim = t.Shape[0];
            long outerDim = (long)t.Shape.ElementCount / innerDim;
            estimatedWeightBytes += Cpu.Kernels.Dequantize.RowByteSize(innerDim, t.QuantizationType) * outerDim;
        }

        string? vramWarning = null;
        if (CudaDriverApi.cuMemGetInfo_v2(out nuint freeBefore, out nuint totalVram) == 0
            && totalVram > 0 && estimatedWeightBytes > (long)freeBefore)
        {
            long modelMb = estimatedWeightBytes / (1024 * 1024);
            long freeMb = (long)freeBefore / (1024 * 1024);
            long totalMb = (long)totalVram / (1024 * 1024);
            vramWarning = $"Model weights (~{modelMb} MB) exceed available VRAM ({freeMb}/{totalMb} MB free). " +
                          $"Performance will be degraded due to PCIe memory paging. " +
                          $"Consider a smaller model or quantization format.";
        }

        // Upload weights to GPU
        var weights = CudaWeights.LoadFromGguf(cpuWeights, config, kernels, stream.Handle);

        // Create scratch buffers
        var state = new CudaForwardState(
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize);

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        int ropeType = (int)(config.RoPEConfig?.Type ?? RoPEType.Norm);

        return new CudaTransformerModel(config, weights, state, stream, cublas, context,
            kernels, gguf, deviceId, ropeTheta, ropeDim, ropeType, vramWarning);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        _context.MakeCurrent();
        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;
        int slidingWindow = Config.SlidingWindowSize ?? 0;
        int h = sizeof(ushort); // FP16 element size

        nint s = _stream.Handle;
        nint cublasH = _cublas.Handle;

        _state.EnsureCapacity(seqLen);

        // 1. Upload tokenIds + positions to device
        fixed (int* tokenPtr = tokenIds)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.TokenIdsDevice, (nint)tokenPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();
        fixed (int* posPtr = positions)
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();

        // 2. Embedding lookup → FP16 HiddenState
        _kernels.LaunchEmbeddingLookup(
            _weights.TokenEmbedDevice, _weights.TokenEmbedQuantType,
            _state.TokenIdsDevice, _state.HiddenState,
            seqLen, hiddenSize, s);

        // 3. Layer 0 setup: copy hidden→residual, RmsNorm→NormOutput
        long hiddenBytes = (long)seqLen * hiddenSize * h;
        CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState, (nuint)hiddenBytes, s).ThrowOnError();
        _kernels.LaunchRmsNorm(_state.HiddenState, _weights.Layers[0].AttnNormWeight, _state.NormOutput,
            hiddenSize, eps, seqLen, s);

        // 4. Transformer layers — FP16 activations, cuBLAS GEMM for prefill, quantized GEMV for decode,
        //    FusedAddRmsNorm at residual junctions to avoid FP16 truncation.
        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,   // skip all layers (embedding + LM head only)
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        // When skipping all layers, treat embedding output as final hidden state
        if (numLayers == 0)
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.HiddenState, _state.Residual, (nuint)hiddenBytes, s).ThrowOnError();
        }

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // ── ATTENTION BLOCK (NormOutput has normalized input) ──

            // Q/K/V projections: prefill → cuBLAS HGEMM, decode → quantized GEMV.
            // ProjectQkv picks the fused 1-launch path when all three weights qualify
            // and falls back to 3 sequential Project() calls otherwise.
            ProjectQkv(in lw, _state.NormOutput, _state.Q, _state.K, _state.V, seqLen);

            // Optional biases (FP16)
            if (lw.QBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.Q, lw.QBias, lw.QOutputDim, seqLen, s);
            if (lw.KBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.K, lw.KBias, lw.KOutputDim, seqLen, s);
            if (lw.VBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAdd(_state.V, lw.VBias, lw.VOutputDim, seqLen, s);

            // Optional QK-norms (FP16)
            if (lw.QNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_state.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            if (lw.KNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_state.K, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

            // RoPE (FP16, in-place on Q and K)
            int effectiveRopeType = DebugRopeTypeOverride >= 0 ? DebugRopeTypeOverride : _ropeType;
            _kernels.LaunchRoPE(_state.Q, _state.K, _state.PositionsDevice,
                seqLen, numHeads, numKvHeads, headDim,
                _ropeDim, _ropeTheta, effectiveRopeType, s);

            // KV-cache update + Attention (FP16)
            if (kvCache is CudaQuantizedKvCache cudaQKvCache)
            {
                cudaQKvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer, s, _kernels);
                int seqKv = cudaQKvCache.CurrentLength;

                // Dequant quantized region + copy window → scratch, then regular attention
                var (kPtr, vPtr) = cudaQKvCache.PrepareAttentionScratch(layer, s, _kernels);
                _kernels.LaunchAttention(_state.Q, kPtr, vPtr, _state.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim,
                    positions[0], slidingWindow, s);
            }
            else if (kvCache is CudaKvCache cudaKvCache)
            {
                cudaKvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer, s);
                int seqKv = cudaKvCache.CurrentLength;

                _kernels.LaunchAttention(_state.Q, cudaKvCache.GetKeysPtr(layer),
                    cudaKvCache.GetValuesPtr(layer), _state.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim,
                    positions[0], slidingWindow, s);
            }
            else
            {
                _kernels.LaunchAttention(_state.Q, _state.K, _state.V, _state.AttnOutput,
                    seqLen, seqLen, numHeads, numKvHeads, headDim,
                    0, slidingWindow, s);
            }

            // O projection → NormOutput
            Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);
            if (lw.OBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

            // ── FUSED: attention residual + FFN norm ──
            // residual = residual + NormOutput (via FP32), NormOutput = rmsnorm(new_residual, ffnNormWeight)
            _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                hiddenSize, eps, seqLen, s);

            // ── FFN BLOCK (NormOutput has FFN-normalized input) ──

            // Gate/Up projections. Same fusion logic as Q/K/V above.
            ProjectGateUp(in lw, _state.NormOutput, _state.FfnGate, _state.FfnUp, seqLen);

            if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_state.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_state.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

            // SwiGLU (FP16)
            _kernels.LaunchSwiGLU(_state.FfnGate, _state.FfnUp, _state.SiluOutput,
                intermediateSize, seqLen, s);

            // Down projection → NormOutput
            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

            // ── FUSED: FFN residual + next layer's attention norm ──
            if (layer < numLayers - 1)
            {
                ref readonly var nextLw = ref _weights.Layers[layer + 1];
                _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);
            }
            else
            {
                // Last processed layer: plain add → HiddenState for final norm
                _kernels.LaunchAdd(_state.Residual, _state.NormOutput, _state.HiddenState,
                    seqLen * hiddenSize, s);
            }
        }

        // 5. Final RmsNorm (last token only)
        nint lastHidden = _state.HiddenState + (nint)((seqLen - 1) * hiddenSize * h);
        _kernels.LaunchRmsNorm(lastHidden, _weights.OutputNormWeight, _state.NormOutput,
            hiddenSize, eps, 1, s);

        // 6. LM head (last token only) → FP16 logits, then convert to FP32
        Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
            _state.NormOutput, _state.LogitsF16,
            _weights.OutputOutputDim, _weights.OutputInputDim, 1);
        _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);

        // 7. Stream sync (single sync point for entire forward pass)
        _stream.Synchronize();

        // 8. D2H copy FP32 logits to CPU UnmanagedTensor
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.LogitsF32,
            (nuint)(vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }

    /// <summary>
    /// Projects Q/K/V from a shared input. Picks the fused 1-launch kernel when all
    /// three weights qualify (same fused-capable quant type, no FP16 fallback, matching
    /// input dim, decode-only); otherwise issues three sequential <see cref="Project"/>
    /// calls — numerically identical, just two extra <c>cuLaunchKernel</c> dispatches.
    /// </summary>
    /// <remarks>
    /// Fusion saves ~5.5 ms / session at 30 layers × 32 tokens (verified on
    /// SmolLM-135M Q8_0). Disqualifying conditions:
    /// <list type="bullet">
    /// <item>Prefill (<paramref name="seqLen"/> &gt; 1) — uses cuBLAS HGEMM, not GEMV.</item>
    /// <item>Mixed quant types across Q, K, V (e.g. Q8_0 attention + K-quants elsewhere).</item>
    /// <item>FP16 fallback weights present (<c>QQuant == 0</c>) — would need dequant scratch.</item>
    /// <item>Mismatched input dim across Q/K/V (would need separate K-loops).</item>
    /// <item>Quant type without a fused kernel (<see cref="CudaKernels.HasFusedQuantizedGemv"/>).</item>
    /// </list>
    /// </remarks>
    private void ProjectQkv(ref readonly CudaLayerWeights lw,
                             nint input, nint qOut, nint kOut, nint vOut, int seqLen)
    {
        bool fused = seqLen == 1
            && lw.QQuantType == lw.KQuantType && lw.QQuantType == lw.VQuantType
            && CudaKernels.HasFusedQuantizedGemv(lw.QQuantType)
            && lw.QQuant != 0 && lw.KQuant != 0 && lw.VQuant != 0
            && lw.QInputDim == lw.KInputDim && lw.QInputDim == lw.VInputDim;

        if (fused)
        {
            _kernels.LaunchFusedQuantizedGemv3(
                lw.QQuant, lw.KQuant, lw.VQuant,
                qOut, kOut, vOut,
                input, lw.QQuantType,
                lw.QOutputDim, lw.KOutputDim, lw.VOutputDim, lw.QInputDim,
                _stream.Handle);
            return;
        }

        Project(lw.QQuant, lw.QQuantType, lw.Q, input, qOut, lw.QOutputDim, lw.QInputDim, seqLen);
        Project(lw.KQuant, lw.KQuantType, lw.K, input, kOut, lw.KOutputDim, lw.KInputDim, seqLen);
        Project(lw.VQuant, lw.VQuantType, lw.V, input, vOut, lw.VOutputDim, lw.VInputDim, seqLen);
    }

    /// <summary>
    /// Projects Gate/Up from a shared input. Companion to <see cref="ProjectQkv"/>
    /// with the same fused-vs-sequential decision logic.
    /// </summary>
    private void ProjectGateUp(ref readonly CudaLayerWeights lw,
                                nint input, nint gateOut, nint upOut, int seqLen)
    {
        bool fused = seqLen == 1
            && lw.GateQuantType == lw.UpQuantType
            && CudaKernels.HasFusedQuantizedGemv(lw.GateQuantType)
            && lw.GateQuant != 0 && lw.UpQuant != 0
            && lw.GateInputDim == lw.UpInputDim;

        if (fused)
        {
            _kernels.LaunchFusedQuantizedGemv2(
                lw.GateQuant, lw.UpQuant,
                gateOut, upOut,
                input, lw.GateQuantType,
                lw.GateOutputDim, lw.UpOutputDim, lw.GateInputDim,
                _stream.Handle);
            return;
        }

        Project(lw.GateQuant, lw.GateQuantType, lw.Gate, input, gateOut, lw.GateOutputDim, lw.GateInputDim, seqLen);
        Project(lw.UpQuant, lw.UpQuantType, lw.Up, input, upOut, lw.UpOutputDim, lw.UpInputDim, seqLen);
    }

    /// <summary>
    /// Dispatches projection as cuBLAS HGEMM (prefill) or quantized/cuBLAS GEMV (decode).
    /// For quantized weights with no persistent FP16 copy (<paramref name="fp16Weight"/> == 0),
    /// dequantizes on-the-fly into <see cref="CudaForwardState.DequantScratch"/> before calling cuBLAS.
    /// </summary>
    private void Project(nint quantWeight, QuantizationType qt, nint fp16Weight,
                          nint input, nint output, int outputDim, int inputDim, int seqLen)
    {
        nint s = _stream.Handle;

        if (seqLen > 1) // Prefill: cuBLAS HGEMM
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                // Quantized: dequant into scratch, then GEMM
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                    outputDim * inputDim, s);
                w = _state.DequantScratch;
            }
            CudaGemm.LinearF16(_cublas.Handle, input, w, output, seqLen, inputDim, outputDim, s);
        }
        else if (quantWeight != 0 && CudaKernels.HasQuantizedGemv(qt)) // Decode: quantized GEMV
        {
            _kernels.LaunchQuantizedGemv(quantWeight, qt, input, output, outputDim, inputDim, s);
        }
        else // Decode fallback: cuBLAS GEMV (F16/F32 weights or unsupported quant)
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                    outputDim * inputDim, s);
                w = _state.DequantScratch;
            }
            CudaGemm.GemvF16(_cublas.Handle, w, input, output, outputDim, inputDim, s);
        }
    }

    /// <summary>
    /// Creates a <see cref="CudaKvCache"/> for this model.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length for the cache.</param>
    public CudaKvCache CreateKvCache(int maxSeqLen)
    {
        _context.MakeCurrent();
        return new CudaKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);
    }

    /// <summary>
    /// Creates a KV-cache with optional quantization for this model.
    /// Returns <see cref="CudaQuantizedKvCache"/> when quantization is configured,
    /// otherwise a standard <see cref="CudaKvCache"/>.
    /// </summary>
    public Core.Attention.IKvCache CreateKvCache(int maxSeqLen, Core.Configuration.KvCacheConfig config)
    {
        _context.MakeCurrent();
        if (!config.IsQuantized)
            return new CudaKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);
        return new CudaQuantizedKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen, config);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _state.Dispose();
        _weights.Dispose();
        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
    }
}
