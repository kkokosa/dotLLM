using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda;

/// <summary>
/// Hybrid CPU/GPU transformer model: first N layers run on GPU (FP16, cuBLAS/PTX kernels),
/// remaining layers run on CPU (FP32, SIMD kernels). Hidden state is transferred D2H at
/// the layer boundary with FP16-to-FP32 conversion. Useful when a model doesn't fully fit
/// in VRAM.
/// </summary>
public sealed unsafe class HybridTransformerModel : IModel
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int Q8_1GroupSize = 32;
    private const int InterleavedMinRowBytes = 1024;

    // ── GPU resources ──
    private readonly CudaWeights _gpuWeights;
    private readonly CudaForwardState _gpuState;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaContext _context;
    private readonly CudaKernels _kernels;

    // ── CPU resources ──
    private readonly TransformerWeights _cpuWeights;
    private readonly TransformerForwardState _cpuState;
    private readonly ComputeThreadPool? _threadPool;
    private readonly bool _ownsThreadPool;

    // ── Shared ──
    private readonly GgufFile _gguf;
    private readonly int _numGpuLayers;
    private readonly int _deviceId;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _gpuRopeType;
    private readonly RoPEType _cpuRopeType;
    private readonly int? _slidingWindowSize;
    private nint _fp16TransferBuffer;
    private int _fp16TransferCapacity; // in elements

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _gpuState.AllocatedBytes + _cpuState.AllocatedBytes;

    /// <summary>Non-null when GPU-side weights exceed available VRAM.</summary>
    public string? VramWarning { get; }

    /// <summary>Number of transformer layers running on GPU.</summary>
    public int NumGpuLayers => _numGpuLayers;

    /// <summary>Debug: limit the number of transformer layers processed. 0 = all layers (default). -1 = skip all layers.</summary>
    internal int DebugMaxLayers { get; set; }

    private HybridTransformerModel(
        ModelConfig config, CudaWeights gpuWeights, CudaForwardState gpuState,
        CudaStream stream, CudaCublasHandle cublas, CudaContext context,
        CudaKernels kernels, TransformerWeights cpuWeights, TransformerForwardState cpuState,
        ComputeThreadPool? threadPool, bool ownsPool, GgufFile gguf,
        int numGpuLayers, int deviceId, float ropeTheta, int ropeDim,
        int gpuRopeType, RoPEType cpuRopeType, int? slidingWindowSize,
        string? vramWarning)
    {
        Config = config;
        _gpuWeights = gpuWeights;
        _gpuState = gpuState;
        _stream = stream;
        _cublas = cublas;
        _context = context;
        _kernels = kernels;
        _cpuWeights = cpuWeights;
        _cpuState = cpuState;
        _threadPool = threadPool;
        _ownsThreadPool = ownsPool;
        _gguf = gguf;
        _numGpuLayers = numGpuLayers;
        _deviceId = deviceId;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _gpuRopeType = gpuRopeType;
        _cpuRopeType = cpuRopeType;
        _slidingWindowSize = slidingWindowSize;
        VramWarning = vramWarning;

        // Initial transfer buffer for decode (1 token)
        _fp16TransferCapacity = config.HiddenSize;
        _fp16TransferBuffer = (nint)NativeMemory.AlignedAlloc(
            (nuint)(_fp16TransferCapacity * sizeof(ushort)), 64);
    }

    /// <summary>
    /// Loads a hybrid transformer model from an opened GGUF file.
    /// </summary>
    /// <param name="gguf">Opened GGUF file (must remain alive for model lifetime).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="numGpuLayers">Number of layers to run on GPU (must be &gt; 0 and &lt; config.NumLayers).</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="threading">CPU threading configuration for CPU-side layers.</param>
    public static HybridTransformerModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int numGpuLayers,
        int deviceId, ThreadingConfig threading)
    {
        if (numGpuLayers <= 0 || numGpuLayers >= config.NumLayers)
            throw new ArgumentOutOfRangeException(nameof(numGpuLayers),
                $"numGpuLayers must be between 1 and {config.NumLayers - 1} for hybrid mode. " +
                $"Use TransformerModel for pure CPU or CudaTransformerModel for pure GPU.");

        // 1. Load CPU weights (mmap references only)
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
        cpuWeights.RepackWeights();

        // 2. Initialize CUDA
        var context = CudaContext.Create(deviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        string? ptxDir = Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        // 3. Upload only GPU layers to VRAM
        var gpuWeights = CudaWeights.LoadFromGguf(cpuWeights, config, kernels, stream.Handle, numGpuLayers);

        // 4. VRAM estimation and warning
        string? vramWarning = null;
        if (CudaDriverApi.cuMemGetInfo_v2(out nuint freeAfter, out nuint totalVram) == 0
            && totalVram > 0)
        {
            // Rough check: if less than 10% free after loading, warn
            double freePercent = (double)freeAfter / totalVram;
            if (freePercent < 0.10)
            {
                long freeMb = (long)freeAfter / (1024 * 1024);
                long totalMb = (long)totalVram / (1024 * 1024);
                vramWarning = $"VRAM nearly full after loading {numGpuLayers}/{config.NumLayers} layers " +
                              $"({freeMb}/{totalMb} MB free). Consider reducing --gpu-layers.";
            }
        }

        // 5. GPU scratch buffers
        var gpuState = new CudaForwardState(
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize);

        // 6. CPU scratch buffers (with RoPE tables)
        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        RoPEType cpuRopeType = config.RoPEConfig?.Type ?? RoPEType.Norm;
        int gpuRopeType = (int)cpuRopeType;

        var cpuState = new TransformerForwardState(
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize,
            config.MaxSequenceLength, ropeDim, ropeTheta);

        // 7. ComputeThreadPool for CPU layers
        ComputeThreadPool? pool = null;
        if (threading.IsParallel)
        {
            int effectiveThreads = threading.EffectiveThreadCount;
            if (threading.EnableNumaPinning || threading.EnablePCorePinning)
            {
                var topology = NumaTopology.Detect();
                if (threading.EnablePCorePinning && topology.IsHybrid)
                    effectiveThreads = Math.Min(effectiveThreads, topology.PerformanceCoreIds.Count);
                pool = new ComputeThreadPool(effectiveThreads, topology, threading);
            }
            else
            {
                pool = new ComputeThreadPool(effectiveThreads, topology: null, threading);
            }
        }

        return new HybridTransformerModel(
            config, gpuWeights, gpuState, stream, cublas, context, kernels,
            cpuWeights, cpuState, pool, ownsPool: pool is not null, gguf,
            numGpuLayers, deviceId, ropeTheta, ropeDim, gpuRopeType, cpuRopeType,
            config.SlidingWindowSize, vramWarning);
    }

    /// <summary>Creates a <see cref="HybridKvCache"/> for this model.</summary>
    public HybridKvCache CreateKvCache(int maxSeqLen)
    {
        _context.MakeCurrent();
        return new HybridKvCache(
            new CudaKvCache(_numGpuLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen),
            new SimpleKvCache(Config.NumLayers - _numGpuLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen),
            _numGpuLayers);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    /// <remarks>
    /// SYNC WARNING: The GPU phase (layers 0..N-1) replicates logic from
    /// CudaTransformerModel.Forward and the CPU phase (layers N..L-1) replicates logic
    /// from TransformerModel.Forward. Bug fixes to attention, FFN, or norm logic may
    /// need to be applied in all three locations.
    /// </remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        _context.MakeCurrent();
        int maxSeq = Config.MaxSequenceLength;
        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        int kvStride = numKvHeads * headDim;
        float eps = Config.NormEpsilon;
        int slidingWindow = Config.SlidingWindowSize ?? 0;
        int h = sizeof(ushort); // FP16 element size

        int totalLayers = Config.NumLayers;
        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,
            0 => totalLayers,
            _ => Math.Min(DebugMaxLayers, totalLayers)
        };

        int gpuLayers = Math.Min(_numGpuLayers, numLayers);
        int cpuLayers = numLayers - gpuLayers;

        var hybridKvCache = kvCache as HybridKvCache;

        // ═══════════════════════════════════════════════════
        //  PHASE 1: GPU — Embedding + Layers 0..gpuLayers-1
        // ═══════════════════════════════════════════════════

        nint s = _stream.Handle;
        nint cublasH = _cublas.Handle;

        _gpuState.EnsureCapacity(seqLen);

        // 1. Upload tokenIds + positions
        fixed (int* tokenPtr = tokenIds)
            CudaDriverApi.cuMemcpyHtoD_v2(_gpuState.TokenIdsDevice, (nint)tokenPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();
        fixed (int* posPtr = positions)
            CudaDriverApi.cuMemcpyHtoD_v2(_gpuState.PositionsDevice, (nint)posPtr,
                (nuint)(seqLen * sizeof(int))).ThrowOnError();

        // 2. GPU Embedding lookup → FP16 HiddenState
        _kernels.LaunchEmbeddingLookup(
            _gpuWeights.TokenEmbedDevice, _gpuWeights.TokenEmbedQuantType,
            _gpuState.TokenIdsDevice, _gpuState.HiddenState,
            seqLen, hiddenSize, s);

        // 3. Layer 0 setup: copy hidden→residual, RmsNorm→NormOutput
        long hiddenBytes = (long)seqLen * hiddenSize * h;

        if (gpuLayers > 0)
        {
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_gpuState.Residual, _gpuState.HiddenState,
                (nuint)hiddenBytes, s).ThrowOnError();
            _kernels.LaunchRmsNorm(_gpuState.HiddenState, _gpuWeights.Layers[0].AttnNormWeight,
                _gpuState.NormOutput, hiddenSize, eps, seqLen, s);
        }

        // 4. GPU layer loop
        for (int layer = 0; layer < gpuLayers; layer++)
        {
            ref readonly var lw = ref _gpuWeights.Layers[layer];

            // ── ATTENTION BLOCK ──
            ProjectGpu(lw.QQuant, lw.QQuantType, lw.Q, _gpuState.NormOutput, _gpuState.Q,
                lw.QOutputDim, lw.QInputDim, seqLen);
            ProjectGpu(lw.KQuant, lw.KQuantType, lw.K, _gpuState.NormOutput, _gpuState.K,
                lw.KOutputDim, lw.KInputDim, seqLen);
            ProjectGpu(lw.VQuant, lw.VQuantType, lw.V, _gpuState.NormOutput, _gpuState.V,
                lw.VOutputDim, lw.VInputDim, seqLen);

            if (lw.QBias != 0) _kernels.LaunchBiasAdd(_gpuState.Q, lw.QBias, lw.QOutputDim, seqLen, s);
            if (lw.KBias != 0) _kernels.LaunchBiasAdd(_gpuState.K, lw.KBias, lw.KOutputDim, seqLen, s);
            if (lw.VBias != 0) _kernels.LaunchBiasAdd(_gpuState.V, lw.VBias, lw.VOutputDim, seqLen, s);

            if (lw.QNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_gpuState.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            if (lw.KNormWeight != 0)
                _kernels.LaunchPerHeadRmsNorm(_gpuState.K, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

            _kernels.LaunchRoPE(_gpuState.Q, _gpuState.K, _gpuState.PositionsDevice,
                seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, _gpuRopeType, s);

            // KV-cache update + Attention
            var gpuKvCache = hybridKvCache?.GpuCache;
            if (gpuKvCache != null)
            {
                gpuKvCache.UpdateDevice(_gpuState.K, _gpuState.V, positions, seqLen, layer, s);
                int seqKv = gpuKvCache.CurrentLength;
                _kernels.LaunchAttention(_gpuState.Q, gpuKvCache.GetKeysPtr(layer),
                    gpuKvCache.GetValuesPtr(layer), _gpuState.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], slidingWindow, s);
            }
            else
            {
                _kernels.LaunchAttention(_gpuState.Q, _gpuState.K, _gpuState.V, _gpuState.AttnOutput,
                    seqLen, seqLen, numHeads, numKvHeads, headDim, 0, slidingWindow, s);
            }

            // O projection → NormOutput
            ProjectGpu(lw.OQuant, lw.OQuantType, lw.O, _gpuState.AttnOutput, _gpuState.NormOutput,
                lw.OOutputDim, lw.OInputDim, seqLen);
            if (lw.OBias != 0) _kernels.LaunchBiasAdd(_gpuState.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

            // ── FUSED: attention residual + FFN norm ──
            _kernels.LaunchFusedAddRmsNorm(_gpuState.Residual, _gpuState.NormOutput,
                lw.FfnNormWeight, _gpuState.NormOutput, hiddenSize, eps, seqLen, s);

            // ── FFN BLOCK ──
            ProjectGpu(lw.GateQuant, lw.GateQuantType, lw.Gate, _gpuState.NormOutput, _gpuState.FfnGate,
                lw.GateOutputDim, lw.GateInputDim, seqLen);
            ProjectGpu(lw.UpQuant, lw.UpQuantType, lw.Up, _gpuState.NormOutput, _gpuState.FfnUp,
                lw.UpOutputDim, lw.UpInputDim, seqLen);

            if (lw.GateBias != 0) _kernels.LaunchBiasAdd(_gpuState.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
            if (lw.UpBias != 0) _kernels.LaunchBiasAdd(_gpuState.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

            _kernels.LaunchSwiGLU(_gpuState.FfnGate, _gpuState.FfnUp, _gpuState.SiluOutput,
                intermediateSize, seqLen, s);

            ProjectGpu(lw.DownQuant, lw.DownQuantType, lw.Down, _gpuState.SiluOutput, _gpuState.NormOutput,
                lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAdd(_gpuState.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

            // ── FUSED: FFN residual + next layer setup ──
            if (layer < gpuLayers - 1)
            {
                // Not last GPU layer: FusedAddRmsNorm for next GPU layer
                ref readonly var nextLw = ref _gpuWeights.Layers[layer + 1];
                _kernels.LaunchFusedAddRmsNorm(_gpuState.Residual, _gpuState.NormOutput,
                    nextLw.AttnNormWeight, _gpuState.NormOutput, hiddenSize, eps, seqLen, s);
            }
            else
            {
                // Last GPU layer: plain Add → HiddenState (boundary transfer source)
                _kernels.LaunchAdd(_gpuState.Residual, _gpuState.NormOutput, _gpuState.HiddenState,
                    seqLen * hiddenSize, s);
            }
        }

        // ═══════════════════════════════════════════════════
        //  PHASE 2: Boundary Transfer (GPU → CPU)
        // ═══════════════════════════════════════════════════

        if (gpuLayers > 0 && cpuLayers > 0)
        {
            _stream.Synchronize();

            // D2H: HiddenState [seqLen, hiddenSize] as FP16
            int transferElements = seqLen * hiddenSize;
            EnsureTransferCapacity(transferElements);

            long transferBytes = (long)transferElements * h;
            CudaDriverApi.cuMemcpyDtoH_v2(_fp16TransferBuffer, _gpuState.HiddenState,
                (nuint)transferBytes).ThrowOnError();

            // FP16 → FP32 conversion into CPU state
            _cpuState.EnsureCapacity(seqLen);
            ConvertFp16ToFp32(_fp16TransferBuffer, _cpuState.HiddenState, transferElements);
        }

        // ═══════════════════════════════════════════════════
        //  PHASE 3: CPU — Layers gpuLayers..totalLayers-1
        //           + Final Norm + LM Head
        // ═══════════════════════════════════════════════════

        if (cpuLayers > 0)
        {
            _cpuState.EnsureCapacity(seqLen);

            // Adaptive dispatch mode
            _threadPool?.SetDispatchMode(seqLen == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);

            float* hidden = (float*)_cpuState.HiddenState;
            float* residual = (float*)_cpuState.Residual;
            float* normOut = (float*)_cpuState.NormOutput;
            float* q = (float*)_cpuState.Q;
            float* k = (float*)_cpuState.K;
            float* v = (float*)_cpuState.V;
            float* attnOut = (float*)_cpuState.AttnOutput;
            float* ffnGate = (float*)_cpuState.FfnGate;
            float* ffnUp = (float*)_cpuState.FfnUp;
            float* siluOut = (float*)_cpuState.SiluOutput;
            float* logits = (float*)_cpuState.Logits;

            // If no GPU layers, we need to do embedding lookup on CPU
            if (gpuLayers == 0)
                EmbeddingLookupCpu(tokenIds, hidden, hiddenSize);

            var repackedLayers = _cpuWeights.RepackedLayers;

            for (int layer = gpuLayers; layer < numLayers; layer++)
            {
                ref readonly var lw = ref _cpuWeights.Layers[layer];
                var rl = repackedLayers?[layer];
                int cpuCacheLayer = layer; // HybridKvCache handles remapping

                // a. Copy hiddenState → residual
                new Span<float>(hidden, seqLen * hiddenSize)
                    .CopyTo(new Span<float>(residual, seqLen * hiddenSize));

                // b. RMSNorm + Pre-quantize + Q/K/V projections
                byte* inputQ8Scratch = (byte*)_cpuState.InputQ8Scratch;

                if (seqLen == 1 && _threadPool != null)
                {
                    // Decode path: try fused RmsNorm+Quantize
                    byte* preQuantNorm = null;
                    if (IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType)
                        && IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType))
                    {
                        preQuantNorm = FusedOps.RmsNormQuantize(hidden, lw.AttnNormWeight, eps,
                            inputQ8Scratch, hiddenSize, lw.QQuantType);
                    }

                    if (preQuantNorm == null)
                    {
                        RmsNorm.Execute(
                            new ReadOnlySpan<float>(hidden, hiddenSize),
                            lw.AttnNormWeight, eps,
                            new Span<float>(normOut, hiddenSize));
                        preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, 1, lw.QQuantType);
                    }

                    FusedQkvDecode(in lw, normOut, preQuantNorm, q, k, v);
                }
                else
                {
                    // Prefill path: unfused
                    for (int t = 0; t < seqLen; t++)
                    {
                        RmsNorm.Execute(
                            new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                            lw.AttnNormWeight, eps,
                            new Span<float>(normOut + t * hiddenSize, hiddenSize));
                    }

                    byte* preQuantNorm = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, lw.QQuantType);

                    var rwQ = rl?.Q ?? default;
                    var rwK = rl?.K ?? default;
                    var rwV = rl?.V ?? default;
                    GemmInterleaved(lw.QWeight, lw.QQuantType, normOut, q, lw.QOutputDim, lw.QInputDim, seqLen,
                        preQuantNorm, in rwQ);
                    GemmInterleaved(lw.KWeight, lw.KQuantType, normOut, k, lw.KOutputDim, lw.KInputDim, seqLen,
                        IsCompatiblePreQuant(lw.QQuantType, lw.KQuantType) ? preQuantNorm : null, in rwK);
                    GemmInterleaved(lw.VWeight, lw.VQuantType, normOut, v, lw.VOutputDim, lw.VInputDim, seqLen,
                        IsCompatiblePreQuant(lw.QQuantType, lw.VQuantType) ? preQuantNorm : null, in rwV);
                }

                // Optional bias
                AddBias(lw.QBias, q, lw.QOutputDim, seqLen);
                AddBias(lw.KBias, k, lw.KOutputDim, seqLen);
                AddBias(lw.VBias, v, lw.VOutputDim, seqLen);

                // Optional QK-norms
                if (lw.QNormWeight is not null)
                    ApplyPerHeadNorm(lw.QNormWeight, q, numHeads, headDim, seqLen, eps);
                if (lw.KNormWeight is not null)
                    ApplyPerHeadNorm(lw.KNormWeight, k, numKvHeads, headDim, seqLen, eps);

                // RoPE
                RoPE.Execute(
                    new Span<float>(q, seqLen * numHeads * headDim),
                    new Span<float>(k, seqLen * kvStride),
                    positions, numHeads, numKvHeads, headDim, _ropeDim,
                    _cpuState.CosTable, _cpuState.SinTable, _cpuRopeType);

                // Attention with KV-cache
                if (hybridKvCache is not null)
                {
                    var kRef = new TensorRef(seqLen, kvStride, DType.Float32, -1, (nint)k);
                    var vRef = new TensorRef(seqLen, kvStride, DType.Float32, -1, (nint)v);
                    hybridKvCache.Update(kRef, vRef, positions, cpuCacheLayer);

                    int seqKv = hybridKvCache.CpuCache.CurrentLength;
                    var cachedK = hybridKvCache.CpuCache.GetKeysRef(cpuCacheLayer - _numGpuLayers);
                    var cachedV = hybridKvCache.CpuCache.GetValuesRef(cpuCacheLayer - _numGpuLayers);

                    Attention.Execute(q, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, attnOut,
                        seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], _threadPool,
                        _slidingWindowSize);
                }
                else
                {
                    Attention.Execute(q, k, v, attnOut,
                        seqLen, seqLen, numHeads, numKvHeads, headDim, 0, _threadPool,
                        _slidingWindowSize);
                }

                // O projection
                byte* preQuantAttn = QuantizeInput(attnOut, inputQ8Scratch, numHeads * headDim, seqLen, lw.OQuantType);
                var rwO = rl?.O ?? default;
                GemmInterleaved(lw.OWeight, lw.OQuantType, attnOut, normOut, lw.OOutputDim, lw.OInputDim, seqLen,
                    preQuantAttn, in rwO);
                AddBias(lw.OBias, normOut, lw.OOutputDim, seqLen);

                // Residual add
                for (int t = 0; t < seqLen; t++)
                {
                    Add.Execute(
                        new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        new Span<float>(hidden + t * hiddenSize, hiddenSize));
                }

                // Copy hiddenState → residual
                new Span<float>(hidden, seqLen * hiddenSize)
                    .CopyTo(new Span<float>(residual, seqLen * hiddenSize));

                // FFN: RMSNorm + Pre-quantize + Gate/Up projections
                if (seqLen == 1 && _threadPool != null)
                {
                    byte* preQuantFfn = null;
                    if (IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType))
                    {
                        preQuantFfn = FusedOps.RmsNormQuantize(hidden, lw.FfnNormWeight, eps,
                            inputQ8Scratch, hiddenSize, lw.GateQuantType);
                    }

                    if (preQuantFfn == null)
                    {
                        RmsNorm.Execute(
                            new ReadOnlySpan<float>(hidden, hiddenSize),
                            lw.FfnNormWeight, eps,
                            new Span<float>(normOut, hiddenSize));
                        preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, 1, lw.GateQuantType);
                    }

                    FusedGateUpDecode(in lw, normOut, preQuantFfn, ffnGate, ffnUp);
                }
                else
                {
                    for (int t = 0; t < seqLen; t++)
                    {
                        RmsNorm.Execute(
                            new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                            lw.FfnNormWeight, eps,
                            new Span<float>(normOut + t * hiddenSize, hiddenSize));
                    }

                    byte* preQuantFfn = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, lw.GateQuantType);

                    var rwGate = rl?.Gate ?? default;
                    var rwUp = rl?.Up ?? default;
                    GemmInterleaved(lw.GateWeight, lw.GateQuantType, normOut, ffnGate, lw.GateOutputDim, lw.GateInputDim, seqLen,
                        preQuantFfn, in rwGate);
                    GemmInterleaved(lw.UpWeight, lw.UpQuantType, normOut, ffnUp, lw.UpOutputDim, lw.UpInputDim, seqLen,
                        IsCompatiblePreQuant(lw.GateQuantType, lw.UpQuantType) ? preQuantFfn : null, in rwUp);
                }
                AddBias(lw.GateBias, ffnGate, lw.GateOutputDim, seqLen);
                AddBias(lw.UpBias, ffnUp, lw.UpOutputDim, seqLen);

                // Fused SwiGLU
                for (int t = 0; t < seqLen; t++)
                {
                    FusedOps.SwiGLU(
                        new ReadOnlySpan<float>(ffnGate + t * intermediateSize, intermediateSize),
                        new ReadOnlySpan<float>(ffnUp + t * intermediateSize, intermediateSize),
                        new Span<float>(siluOut + t * intermediateSize, intermediateSize));
                }

                // Down projection
                byte* preQuantSilu = QuantizeInput(siluOut, inputQ8Scratch, intermediateSize, seqLen, lw.DownQuantType);
                var rwDown = rl?.Down ?? default;
                GemmInterleaved(lw.DownWeight, lw.DownQuantType, siluOut, normOut, lw.DownOutputDim, lw.DownInputDim, seqLen,
                    preQuantSilu, in rwDown);
                AddBias(lw.DownBias, normOut, lw.DownOutputDim, seqLen);

                // Residual add
                for (int t = 0; t < seqLen; t++)
                {
                    Add.Execute(
                        new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                        new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                        new Span<float>(hidden + t * hiddenSize, hiddenSize));
                }
            }

            // Final RMSNorm
            for (int t = 0; t < seqLen; t++)
            {
                float* hiddenT = hidden + t * hiddenSize;
                float* normOutT = normOut + t * hiddenSize;
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hiddenT, hiddenSize),
                    _cpuWeights.OutputNormWeight, eps,
                    new Span<float>(normOutT, hiddenSize));
                new Span<float>(normOutT, hiddenSize).CopyTo(new Span<float>(hiddenT, hiddenSize));
            }

            // LM Head — last token only
            float* lastHidden = hidden + (seqLen - 1) * hiddenSize;
            {
                var rwOutput = _cpuWeights.RepackedOutput ?? default;
                GemvInterleaved(_cpuWeights.OutputWeight, _cpuWeights.OutputQuantType,
                    lastHidden, logits, _cpuWeights.OutputOutputDim, _cpuWeights.OutputInputDim, in rwOutput);
            }

            // Return [1, vocabSize]
            var shape = new TensorShape(1, vocabSize);
            var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
            new Span<float>(logits, vocabSize).CopyTo(
                new Span<float>((void*)result.DataPointer, vocabSize));

            return result;
        }
        else
        {
            // All layers on GPU, no CPU layers — but this shouldn't happen
            // (constructor validates numGpuLayers < totalLayers).
            // Handle gracefully: sync GPU, produce logits from GPU.
            throw new InvalidOperationException(
                "HybridTransformerModel requires at least one CPU layer. " +
                "Use CudaTransformerModel for full GPU execution.");
        }
    }

    // ═══════════════════════════════════════════════════
    //  GPU helper: Project (same as CudaTransformerModel.Project)
    // ═══════════════════════════════════════════════════

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ProjectGpu(nint quantWeight, QuantizationType qt, nint fp16Weight,
                            nint input, nint output, int outputDim, int inputDim, int seqLen)
    {
        nint s = _stream.Handle;

        if (seqLen > 1)
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _gpuState.DequantScratch,
                    outputDim * inputDim, s);
                w = _gpuState.DequantScratch;
            }
            CudaGemm.LinearF16(_cublas.Handle, input, w, output, seqLen, inputDim, outputDim, s);
        }
        else if (quantWeight != 0 && CudaKernels.HasQuantizedGemv(qt))
        {
            _kernels.LaunchQuantizedGemv(quantWeight, qt, input, output, outputDim, inputDim, s);
        }
        else
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _gpuState.DequantScratch,
                    outputDim * inputDim, s);
                w = _gpuState.DequantScratch;
            }
            CudaGemm.GemvF16(_cublas.Handle, w, input, output, outputDim, inputDim, s);
        }
    }

    // ═══════════════════════════════════════════════════
    //  CPU helpers (same as TransformerModel private methods)
    // ═══════════════════════════════════════════════════

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemv(nint weights, QuantizationType qt, float* x, float* y, int m, int k)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.GemvQ8_0((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q5_0)
            MatMul.GemvQ5_0((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q4_K)
            MatMul.GemvQ4_K((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q5_K)
            MatMul.GemvQ5_K((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.Q6_K)
            MatMul.GemvQ6_K((byte*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.F32)
            MatMul.GemvF32((float*)weights, x, y, m, k, _threadPool);
        else if (qt == QuantizationType.F16)
            MatMul.GemvF16(weights, x, y, m, k, _threadPool);
        else
            GemvDequantFallback(weights, qt, x, y, m, k);
    }

    private static void GemvDequantFallback(nint weights, QuantizationType qt, float* x, float* y, int m, int k)
    {
        long rowBytes = Dequantize.RowByteSize(k, qt);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            var xSpan = new ReadOnlySpan<float>(x, k);
            for (int i = 0; i < m; i++)
            {
                Dequantize.ToFloat32(weights + i * (nint)rowBytes, k, qt, rowSpan);
                y[i] = TensorPrimitives.Dot(new ReadOnlySpan<float>(rowBuf, 0, k), xSpan);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Gemm(nint weights, QuantizationType qt, float* b, float* c,
                      int m, int k, int n, byte* preQuantizedInput = null)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q5_0)
            MatMul.GemmQ5_0((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q4_K)
            MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q5_K)
            MatMul.GemmQ5_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.Q6_K)
            MatMul.GemmQ6_K((byte*)weights, b, c, m, k, n, _threadPool, preQuantizedInput);
        else if (qt == QuantizationType.F32)
            MatMul.GemmF32((float*)weights, b, c, m, k, n, _threadPool);
        else if (qt == QuantizationType.F16)
            MatMul.GemmF16(weights, b, c, m, k, n, _threadPool);
        else
            GemmDequantFallback(weights, qt, b, c, m, k, n);
    }

    private static void GemmDequantFallback(nint weights, QuantizationType qt, float* b, float* c,
                                            int m, int k, int n)
    {
        for (int t = 0; t < n; t++)
            GemvDequantFallback(weights, qt, b + t * k, c + t * m, m, k);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GemvInterleaved(nint origWeights, QuantizationType qt, float* x, float* y,
                                 int m, int k, in WeightRepacking.RepackedWeight rw)
    {
        if (rw.Ptr == 0 || rw.RowBytes < InterleavedMinRowBytes)
        {
            Gemv(origWeights, qt, x, y, m, k);
            return;
        }

        byte* inputQ8Scratch = (byte*)_cpuState.InputQ8Scratch;
        if (qt == QuantizationType.Q8_0)
        {
            int blockCount = k / Q8_0GroupSize;
            int xQ8Bytes = blockCount * Q8_0BlockBytes;
            byte* xQ8 = (byte*)(_threadPool?.GetWorkerScratch(0, xQ8Bytes) ?? (nint)inputQ8Scratch);
            MatMul.QuantizeF32ToQ8_0(x, xQ8, k);
            MatMul.ComputeRowsQ8_0Interleaved((byte*)rw.Ptr, xQ8, y, rw.FullGroupCount, rw.TailRows, blockCount, _threadPool);
        }
        else if (qt == QuantizationType.Q5_0)
        {
            int blockCount = k / Q8_0GroupSize;
            int xQ8Bytes = blockCount * MatMul.Q8_1BlockBytes;
            byte* xQ8 = (byte*)(_threadPool?.GetWorkerScratch(0, xQ8Bytes) ?? (nint)inputQ8Scratch);
            MatMul.QuantizeF32ToQ8_1(x, xQ8, k);
            MatMul.ComputeRowsQ5_0Interleaved((byte*)rw.Ptr, xQ8, y, rw.FullGroupCount, rw.TailRows, blockCount, _threadPool);
        }
        else if (qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K)
        {
            int superBlockCount = k / 256;
            int xQ8KBytes = superBlockCount * MatMul.Q8_K_BlockBytes;
            byte* xQ8K = (byte*)(_threadPool?.GetWorkerScratch(0, xQ8KBytes) ?? (nint)inputQ8Scratch);
            MatMul.QuantizeF32ToQ8_K(x, xQ8K, k);
            if (qt == QuantizationType.Q4_K)
                MatMul.ComputeRowsQ4_KInterleaved((byte*)rw.Ptr, xQ8K, y, rw.FullGroupCount, rw.TailRows, superBlockCount, _threadPool);
            else if (qt == QuantizationType.Q5_K)
                MatMul.ComputeRowsQ5_KInterleaved((byte*)rw.Ptr, xQ8K, y, rw.FullGroupCount, rw.TailRows, superBlockCount, _threadPool);
            else
                MatMul.ComputeRowsQ6_KInterleaved((byte*)rw.Ptr, xQ8K, y, rw.FullGroupCount, rw.TailRows, superBlockCount, _threadPool);
        }
        else
        {
            Gemv(origWeights, qt, x, y, m, k);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GemmInterleaved(nint origWeights, QuantizationType qt, float* b, float* c,
                                 int m, int k, int n, byte* preQuantizedInput,
                                 in WeightRepacking.RepackedWeight rw)
    {
        if (rw.Ptr == 0 || n > 1 || rw.RowBytes < InterleavedMinRowBytes)
        {
            Gemm(origWeights, qt, b, c, m, k, n, preQuantizedInput);
            return;
        }

        if (preQuantizedInput != null)
        {
            DispatchInterleavedComputeRows(qt, (byte*)rw.Ptr, preQuantizedInput, c,
                rw.FullGroupCount, rw.TailRows, k);
            return;
        }

        GemvInterleaved(origWeights, qt, b, c, m, k, in rw);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void DispatchInterleavedComputeRows(QuantizationType qt, byte* repackedWeights,
        byte* preQuantInput, float* result, int fullGroups, int tailRows, int k)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.ComputeRowsQ8_0Interleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 32, _threadPool);
        else if (qt == QuantizationType.Q5_0)
            MatMul.ComputeRowsQ5_0Interleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 32, _threadPool);
        else if (qt == QuantizationType.Q4_K)
            MatMul.ComputeRowsQ4_KInterleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 256, _threadPool);
        else if (qt == QuantizationType.Q5_K)
            MatMul.ComputeRowsQ5_KInterleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 256, _threadPool);
        else if (qt == QuantizationType.Q6_K)
            MatMul.ComputeRowsQ6_KInterleaved(repackedWeights, preQuantInput, result,
                fullGroups, tailRows, k / 256, _threadPool);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsCompatiblePreQuant(QuantizationType preQuantSource, QuantizationType target)
    {
        if (preQuantSource == target) return true;
        bool sourceIsKQuant = preQuantSource is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;
        bool targetIsKQuant = target is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;
        if (sourceIsKQuant && targetIsKQuant) return true;
        return false;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte* QuantizeInput(float* input, byte* scratch, int dim, int seqLen, QuantizationType qt)
    {
        if (qt == QuantizationType.Q4_K || qt == QuantizationType.Q5_K || qt == QuantizationType.Q6_K)
        {
            int blockCount = dim / 256;
            int q8kRowBytes = blockCount * MatMul.Q8_K_BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_K(input + t * dim, scratch + t * q8kRowBytes, dim);
            return scratch;
        }
        if (qt == QuantizationType.Q5_0)
        {
            int blockCount = dim / Q8_1GroupSize;
            int q8_1RowBytes = blockCount * MatMul.Q8_1BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_1(input + t * dim, scratch + t * q8_1RowBytes, dim);
            return scratch;
        }
        if (qt == QuantizationType.Q8_0)
        {
            int blockCount = dim / Q8_0GroupSize;
            int q8RowBytes = blockCount * Q8_0BlockBytes;
            for (int t = 0; t < seqLen; t++)
                MatMul.QuantizeF32ToQ8_0(input + t * dim, scratch + t * q8RowBytes, dim);
            return scratch;
        }
        return null;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyPerHeadNorm(float[] normWeight, float* qk,
        int numHeads, int headDim, int seqLen, float eps)
    {
        int stride = numHeads * headDim;
        for (int t = 0; t < seqLen; t++)
            for (int hh = 0; hh < numHeads; hh++)
            {
                float* head = qk + t * stride + hh * headDim;
                RmsNorm.Execute(new ReadOnlySpan<float>(head, headDim), normWeight, eps,
                    new Span<float>(head, headDim));
            }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddBias(float[]? bias, float* output, int outputDim, int seqLen)
    {
        if (bias is null) return;
        for (int t = 0; t < seqLen; t++)
        {
            var row = new Span<float>(output + t * outputDim, outputDim);
            TensorPrimitives.Add((ReadOnlySpan<float>)row, bias, row);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FusedQkvDecode(ref readonly TransformerLayerWeights lw,
        float* normOut, byte* preQuantNorm, float* q, float* k, float* v)
    {
        MatMul.FusedDecodeGemv3(
            (byte*)lw.QWeight, lw.QQuantType, q, lw.QOutputDim,
            (byte*)lw.KWeight, lw.KQuantType, k, lw.KOutputDim,
            (byte*)lw.VWeight, lw.VQuantType, v, lw.VOutputDim,
            normOut, preQuantNorm, lw.QInputDim, _threadPool!);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FusedGateUpDecode(ref readonly TransformerLayerWeights lw,
        float* normOut, byte* preQuantFfn, float* ffnGate, float* ffnUp)
    {
        MatMul.FusedDecodeGemv2(
            (byte*)lw.GateWeight, lw.GateQuantType, ffnGate, lw.GateOutputDim,
            (byte*)lw.UpWeight, lw.UpQuantType, ffnUp, lw.UpOutputDim,
            normOut, preQuantFfn, lw.GateInputDim, _threadPool!);
    }

    /// <summary>
    /// CPU embedding lookup — used only when gpuLayers == 0 (shouldn't happen in practice
    /// since the constructor validates numGpuLayers > 0, but kept for completeness).
    /// </summary>
    private void EmbeddingLookupCpu(ReadOnlySpan<int> tokenIds, float* hidden, int hiddenSize)
    {
        nint embPtr = _cpuWeights.TokenEmbedWeight;
        var qt = _cpuWeights.TokenEmbedQuantType;

        for (int t = 0; t < tokenIds.Length; t++)
        {
            int tokenId = tokenIds[t];
            float* dest = hidden + t * hiddenSize;
            var destSpan = new Span<float>(dest, hiddenSize);

            if (qt == QuantizationType.F32)
            {
                float* src = (float*)embPtr + (long)tokenId * hiddenSize;
                new ReadOnlySpan<float>(src, hiddenSize).CopyTo(destSpan);
            }
            else
            {
                long rowBytes = Dequantize.RowByteSize(hiddenSize, qt);
                nint rowPtr = embPtr + (nint)((long)tokenId * rowBytes);
                Dequantize.ToFloat32(rowPtr, hiddenSize, qt, destSpan);
            }
        }
    }

    // ═══════════════════════════════════════════════════
    //  Boundary transfer helpers
    // ═══════════════════════════════════════════════════

    /// <summary>Ensures the FP16 transfer buffer can hold at least <paramref name="elements"/> Half values.</summary>
    private void EnsureTransferCapacity(int elements)
    {
        if (elements <= _fp16TransferCapacity) return;

        // Power-of-2 growth
        int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)elements);
        if (_fp16TransferBuffer != 0)
            NativeMemory.AlignedFree((void*)_fp16TransferBuffer);
        _fp16TransferBuffer = (nint)NativeMemory.AlignedAlloc(
            (nuint)(newCapacity * sizeof(ushort)), 64);
        _fp16TransferCapacity = newCapacity;
    }

    /// <summary>
    /// Converts FP16 data to FP32 on the CPU using vectorized TensorPrimitives.
    /// Executed once per forward pass at the GPU/CPU boundary.
    /// </summary>
    private static void ConvertFp16ToFp32(nint srcFp16, nint dstFp32, int count)
    {
        var src = new ReadOnlySpan<Half>((Half*)srcFp16, count);
        var dst = new Span<float>((float*)dstFp32, count);
        TensorPrimitives.ConvertToSingle(src, dst);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _gpuState.Dispose();
        _gpuWeights.Dispose();
        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();

        if (_ownsThreadPool)
            _threadPool?.Dispose();
        _cpuState.Dispose();
        _cpuWeights.Dispose();

        if (_fp16TransferBuffer != 0)
        {
            NativeMemory.AlignedFree((void*)_fp16TransferBuffer);
            _fp16TransferBuffer = 0;
        }
    }
}
