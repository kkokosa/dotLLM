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
        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // ── ATTENTION BLOCK (NormOutput has normalized input) ──

            // Q/K/V projections: prefill → cuBLAS HGEMM, decode → quantized GEMV
            Project(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
            Project(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutput, _state.K, lw.KOutputDim, lw.KInputDim, seqLen);
            Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, _state.V, lw.VOutputDim, lw.VInputDim, seqLen);

            // Optional biases (FP16)
            if (lw.QBias != 0) _kernels.LaunchBiasAdd(_state.Q, lw.QBias, lw.QOutputDim, seqLen, s);
            if (lw.KBias != 0) _kernels.LaunchBiasAdd(_state.K, lw.KBias, lw.KOutputDim, seqLen, s);
            if (lw.VBias != 0) _kernels.LaunchBiasAdd(_state.V, lw.VBias, lw.VOutputDim, seqLen, s);

            // Optional QK-norms (FP16)
            if (lw.QNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_state.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            if (lw.KNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_state.K, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

            // RoPE (FP16, in-place on Q and K)
            _kernels.LaunchRoPE(_state.Q, _state.K, _state.PositionsDevice,
                seqLen, numHeads, numKvHeads, headDim,
                _ropeDim, _ropeTheta, _ropeType, s);

            // KV-cache update + Attention (FP16)
            var cudaKvCache = kvCache as CudaKvCache;
            if (cudaKvCache != null)
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

            // Gate/Up projections
            Project(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutput, _state.FfnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
            Project(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutput, _state.FfnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);

            if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_state.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_state.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

            // SwiGLU (FP16)
            _kernels.LaunchSwiGLU(_state.FfnGate, _state.FfnUp, _state.SiluOutput,
                intermediateSize, seqLen, s);

            // Down projection → NormOutput
            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

            // ── FUSED: FFN residual + next layer's attention norm ──
            if (layer < Config.NumLayers - 1)
            {
                ref readonly var nextLw = ref _weights.Layers[layer + 1];
                _kernels.LaunchFusedAddRmsNorm(_state.Residual, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);
            }
            else
            {
                // Last layer: plain add → HiddenState for final norm
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
        return new CudaKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);
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
