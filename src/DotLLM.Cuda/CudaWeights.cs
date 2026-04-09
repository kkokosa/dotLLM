using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda;

/// <summary>
/// Per-layer GPU weight pointers. All linear projections stored as FP16 on device.
/// </summary>
internal readonly struct CudaLayerWeights
{
    // FP16 dequantized weights on device [outputDim, inputDim] (for prefill GEMM)
    public readonly nint Q, K, V, O, Gate, Up, Down;
    // Original quantized weights on device (for decode quantized GEMV)
    public readonly nint QQuant, KQuant, VQuant, OQuant, GateQuant, UpQuant, DownQuant;
    public readonly QuantizationType QQuantType, KQuantType, VQuantType, OQuantType;
    public readonly QuantizationType GateQuantType, UpQuantType, DownQuantType;
    public readonly int QOutputDim, QInputDim, KOutputDim, KInputDim;
    public readonly int VOutputDim, VInputDim, OOutputDim, OInputDim;
    public readonly int GateOutputDim, GateInputDim, UpOutputDim, UpInputDim;
    public readonly int DownOutputDim, DownInputDim;

    // Norm weights on device (FP16)
    public readonly nint AttnNormWeight, FfnNormWeight;
    public readonly nint QNormWeight, KNormWeight; // 0 when absent

    // Bias on device (FP16, 0 when absent)
    public readonly nint QBias, KBias, VBias, OBias;
    public readonly nint GateBias, UpBias, DownBias;

    public CudaLayerWeights(
        nint q, int qOut, int qIn, nint k, int kOut, int kIn,
        nint v, int vOut, int vIn, nint o, int oOut, int oIn,
        nint gate, int gateOut, int gateIn, nint up, int upOut, int upIn,
        nint down, int downOut, int downIn,
        nint attnNorm, nint ffnNorm,
        nint qNorm, nint kNorm,
        nint qBias, nint kBias, nint vBias, nint oBias,
        nint gateBias, nint upBias, nint downBias,
        nint qQuant, QuantizationType qQt, nint kQuant, QuantizationType kQt,
        nint vQuant, QuantizationType vQt, nint oQuant, QuantizationType oQt,
        nint gateQuant, QuantizationType gateQt, nint upQuant, QuantizationType upQt,
        nint downQuant, QuantizationType downQt)
    {
        Q = q; QOutputDim = qOut; QInputDim = qIn;
        K = k; KOutputDim = kOut; KInputDim = kIn;
        V = v; VOutputDim = vOut; VInputDim = vIn;
        O = o; OOutputDim = oOut; OInputDim = oIn;
        Gate = gate; GateOutputDim = gateOut; GateInputDim = gateIn;
        Up = up; UpOutputDim = upOut; UpInputDim = upIn;
        Down = down; DownOutputDim = downOut; DownInputDim = downIn;
        AttnNormWeight = attnNorm; FfnNormWeight = ffnNorm;
        QNormWeight = qNorm; KNormWeight = kNorm;
        QBias = qBias; KBias = kBias; VBias = vBias; OBias = oBias;
        GateBias = gateBias; UpBias = upBias; DownBias = downBias;
        QQuant = qQuant; QQuantType = qQt; KQuant = kQuant; KQuantType = kQt;
        VQuant = vQuant; VQuantType = vQt; OQuant = oQuant; OQuantType = oQt;
        GateQuant = gateQuant; GateQuantType = gateQt;
        UpQuant = upQuant; UpQuantType = upQt;
        DownQuant = downQuant; DownQuantType = downQt;
    }
}

/// <summary>
/// Manages all model weights on GPU. Uploads from GGUF mmap, dequantizes to FP16 on device.
/// </summary>
internal sealed class CudaWeights : IDisposable
{
    public CudaLayerWeights[] Layers { get; }
    public nint TokenEmbedDevice { get; }
    public QuantizationType TokenEmbedQuantType { get; }
    public nint OutputNormWeight { get; }
    public nint OutputWeight { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }
    public nint OutputWeightQuant { get; }
    public QuantizationType OutputQuantType { get; }

    private readonly List<nint> _allAllocations = new();

    private CudaWeights(CudaLayerWeights[] layers, nint tokenEmbed, QuantizationType tokenEmbedQt,
                          nint outputNorm, nint outputWeight, int outputOutDim, int outputInDim,
                          nint outputWeightQuant, QuantizationType outputQt,
                          List<nint> allocs)
    {
        Layers = layers;
        TokenEmbedDevice = tokenEmbed;
        TokenEmbedQuantType = tokenEmbedQt;
        OutputNormWeight = outputNorm;
        OutputWeight = outputWeight;
        OutputOutputDim = outputOutDim;
        OutputInputDim = outputInDim;
        OutputWeightQuant = outputWeightQuant;
        OutputQuantType = outputQt;
        _allAllocations = allocs;
    }

    /// <summary>
    /// Uploads weights from CPU (GGUF mmap) to GPU. Quantized weights are
    /// dequantized to FP16 on-device to avoid transferring the larger FP16 data over PCIe.
    /// </summary>
    /// <param name="cpuWeights">CPU-side weights (mmap'd from GGUF).</param>
    /// <param name="config">Model configuration.</param>
    /// <param name="kernels">Loaded PTX kernels for dequantization.</param>
    /// <param name="stream">CUDA stream for async uploads.</param>
    /// <param name="numGpuLayers">Number of layers to upload. -1 = all layers.
    /// When less than total layers (hybrid mode), output norm and LM head are skipped
    /// since the CPU handles final projection.</param>
    public static CudaWeights LoadFromGguf(TransformerWeights cpuWeights, ModelConfig config,
                                              CudaKernels kernels, nint stream,
                                              int numGpuLayers = -1)
    {
        int layerCount = numGpuLayers < 0
            ? config.NumLayers
            : Math.Min(numGpuLayers, config.NumLayers);
        bool isHybrid = layerCount < config.NumLayers;

        var allocs = new List<nint>();

        // Token embeddings — upload in original format if the embedding kernel supports it,
        // otherwise dequant to FP16 at load time (one-time cost).
        nint tokenEmbed;
        var tokenEmbedQt = cpuWeights.TokenEmbedQuantType;
        if (tokenEmbedQt is QuantizationType.F32 or QuantizationType.F16 or QuantizationType.Q8_0)
        {
            long embedBytes = Dequantize.RowByteSize(config.HiddenSize, tokenEmbedQt) * config.VocabSize;
            tokenEmbed = AllocAndUpload(cpuWeights.TokenEmbedWeight, embedBytes, allocs);
        }
        else
        {
            // Q4_K, Q5_K, Q6_K, Q4_0 — no per-row embedding kernel; dequant entire table to FP16
            tokenEmbed = UploadAndDequant(cpuWeights.TokenEmbedWeight, tokenEmbedQt,
                config.VocabSize, config.HiddenSize, allocs, kernels, stream);
            tokenEmbedQt = QuantizationType.F16;
        }

        // Output norm + LM head: skip in hybrid mode (CPU handles final norm + LM head)
        nint outputNorm = 0;
        nint outputWeight = 0;
        nint outputWeightQuant = 0;

        if (!isHybrid)
        {
            // Output norm (float[] → FP16)
            outputNorm = UploadNormWeight(cpuWeights.OutputNormWeight, allocs, kernels, stream);

            // LM head — too large for the per-projection dequant scratch (vocabSize × hiddenSize).
            // Create a persistent FP16 copy unless it has a custom quantized GEMV kernel
            // (Q8_0/Q4_K/Q6_K can use quantized GEMV directly → no FP16 copy needed).
            bool lmHeadHasGemv = CudaKernels.HasQuantizedGemv(cpuWeights.OutputQuantType);
            outputWeight = (!IsQuantized(cpuWeights.OutputQuantType) || !lmHeadHasGemv)
                ? UploadAndDequant(cpuWeights.OutputWeight, cpuWeights.OutputQuantType,
                    cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, allocs, kernels, stream)
                : 0;
        }

        // Per-layer weights — skip persistent FP16 copies only for types with custom
        // quantized GEMV kernels (Q8_0, Q4_K, Q6_K). These can dequant on-the-fly into
        // a scratch buffer for prefill GEMM, and use the GEMV kernel directly for decode.
        // All other types (Q5_0, Q4_0, Q5_K, F16, F32) keep a persistent FP16 copy.
        // In hybrid mode, only upload the first layerCount layers.
        var layers = new CudaLayerWeights[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            ref readonly var lw = ref cpuWeights.Layers[i];

            nint q = SkipFp16(lw.QQuantType) ? 0 : UploadAndDequant(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim, allocs, kernels, stream);
            nint k = SkipFp16(lw.KQuantType) ? 0 : UploadAndDequant(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim, allocs, kernels, stream);
            nint v = SkipFp16(lw.VQuantType) ? 0 : UploadAndDequant(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim, allocs, kernels, stream);
            nint o = SkipFp16(lw.OQuantType) ? 0 : UploadAndDequant(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim, allocs, kernels, stream);
            nint gate = SkipFp16(lw.GateQuantType) ? 0 : UploadAndDequant(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim, allocs, kernels, stream);
            nint up = SkipFp16(lw.UpQuantType) ? 0 : UploadAndDequant(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim, allocs, kernels, stream);
            nint down = SkipFp16(lw.DownQuantType) ? 0 : UploadAndDequant(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim, allocs, kernels, stream);

            // Upload raw quantized weights (single copy — for decode quantized GEMV and prefill on-the-fly dequant)
            nint qQuant = UploadQuantized(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim, allocs);
            nint kQuant = UploadQuantized(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim, allocs);
            nint vQuant = UploadQuantized(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim, allocs);
            nint oQuant = UploadQuantized(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim, allocs);
            nint gateQuant = UploadQuantized(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim, allocs);
            nint upQuant = UploadQuantized(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim, allocs);
            nint downQuant = UploadQuantized(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim, allocs);

            nint attnNorm = UploadNormWeight(lw.AttnNormWeight, allocs, kernels, stream);
            nint ffnNorm = UploadNormWeight(lw.FfnNormWeight, allocs, kernels, stream);

            nint qNorm = lw.QNormWeight is not null ? UploadNormWeight(lw.QNormWeight, allocs, kernels, stream) : 0;
            nint kNorm = lw.KNormWeight is not null ? UploadNormWeight(lw.KNormWeight, allocs, kernels, stream) : 0;

            nint qBias = UploadBias(lw.QBias, allocs, kernels, stream);
            nint kBias = UploadBias(lw.KBias, allocs, kernels, stream);
            nint vBias = UploadBias(lw.VBias, allocs, kernels, stream);
            nint oBias = UploadBias(lw.OBias, allocs, kernels, stream);
            nint gateBias = UploadBias(lw.GateBias, allocs, kernels, stream);
            nint upBias = UploadBias(lw.UpBias, allocs, kernels, stream);
            nint downBias = UploadBias(lw.DownBias, allocs, kernels, stream);

            layers[i] = new CudaLayerWeights(
                q, lw.QOutputDim, lw.QInputDim, k, lw.KOutputDim, lw.KInputDim,
                v, lw.VOutputDim, lw.VInputDim, o, lw.OOutputDim, lw.OInputDim,
                gate, lw.GateOutputDim, lw.GateInputDim, up, lw.UpOutputDim, lw.UpInputDim,
                down, lw.DownOutputDim, lw.DownInputDim,
                attnNorm, ffnNorm, qNorm, kNorm,
                qBias, kBias, vBias, oBias, gateBias, upBias, downBias,
                qQuant, lw.QQuantType, kQuant, lw.KQuantType,
                vQuant, lw.VQuantType, oQuant, lw.OQuantType,
                gateQuant, lw.GateQuantType, upQuant, lw.UpQuantType,
                downQuant, lw.DownQuantType);
        }

        // Sync to ensure all uploads are complete
        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();

        // LM head quantized copy for decode (skip in hybrid mode — CPU handles LM head)
        if (!isHybrid)
        {
            outputWeightQuant = UploadQuantized(cpuWeights.OutputWeight, cpuWeights.OutputQuantType,
                cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, allocs);
        }

        return new CudaWeights(layers, tokenEmbed, tokenEmbedQt,
            outputNorm, outputWeight, cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim,
            outputWeightQuant, cpuWeights.OutputQuantType, allocs);
    }

    /// <summary>Upload raw quantized weight bytes to GPU (no dequant). For decode quantized GEMV.</summary>
    private static nint UploadQuantized(nint hostPtr, QuantizationType qt,
                                          int outputDim, int inputDim, List<nint> allocs)
    {
        if (qt is QuantizationType.F16 or QuantizationType.F32)
            return 0; // Non-quantized weights don't need a separate quantized copy

        long quantBytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;
        return AllocAndUpload(hostPtr, quantBytes, allocs);
    }

    /// <summary>Upload quantized weight to GPU, then dequantize to FP16 on device.</summary>
    private static nint UploadAndDequant(nint hostPtr, QuantizationType qt,
                                           int outputDim, int inputDim,
                                           List<nint> allocs, CudaKernels kernels, nint stream)
    {
        int totalElements = outputDim * inputDim;

        if (qt == QuantizationType.F16)
        {
            // Already FP16 — just upload
            long bytes = (long)totalElements * sizeof(ushort);
            return AllocAndUpload(hostPtr, bytes, allocs);
        }

        if (qt == QuantizationType.F32)
        {
            // Upload F32, convert to F16 on device
            long f32Bytes = (long)totalElements * sizeof(float);
            nint devF32 = AllocAndUpload(hostPtr, f32Bytes, allocs);
            long f16Bytes = (long)totalElements * sizeof(ushort);
            CudaDriverApi.cuMemAlloc_v2(out nint devF16, (nuint)f16Bytes).ThrowOnError();
            allocs.Add(devF16);
            kernels.LaunchConvertF32ToF16(devF32, devF16, totalElements, stream);
            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
            allocs.Remove(devF32);
            CudaDriverApi.cuMemFree_v2(devF32);
            return devF16;
        }

        // Quantized: upload raw bytes, dequant to FP16 on device
        long quantBytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;
        nint devQuant = AllocAndUpload(hostPtr, quantBytes, allocs);

        long fp16Bytes = (long)totalElements * sizeof(ushort);
        CudaDriverApi.cuMemAlloc_v2(out nint devFp16, (nuint)fp16Bytes).ThrowOnError();
        allocs.Add(devFp16);

        kernels.LaunchDequantToF16(devQuant, qt, devFp16, totalElements, stream);
        return devFp16;
    }

    /// <summary>Upload float[] norm weight → FP16 on device (F32→F16 conversion via GPU kernel).</summary>
    private static unsafe nint UploadNormWeight(float[] weight, List<nint> allocs,
                                                  CudaKernels kernels, nint stream)
    {
        int n = weight.Length;

        // Upload F32 to temp buffer, then convert to FP16 on device
        long f32Bytes = (long)n * sizeof(float);
        long f16Bytes = (long)n * sizeof(ushort);
        CudaDriverApi.cuMemAlloc_v2(out nint devF32, (nuint)f32Bytes).ThrowOnError();
        allocs.Add(devF32);
        fixed (float* ptr = weight)
            CudaDriverApi.cuMemcpyHtoD_v2(devF32, (nint)ptr, (nuint)f32Bytes).ThrowOnError();

        CudaDriverApi.cuMemAlloc_v2(out nint devF16, (nuint)f16Bytes).ThrowOnError();
        allocs.Add(devF16);
        kernels.LaunchConvertF32ToF16(devF32, devF16, n, stream);

        return devF16;
    }

    /// <summary>Upload optional float[] bias → FP16 on device. Returns 0 if bias is null.</summary>
    private static nint UploadBias(float[]? bias, List<nint> allocs,
                                     CudaKernels kernels, nint stream)
    {
        if (bias is null) return 0;
        return UploadNormWeight(bias, allocs, kernels, stream);
    }

    private static bool IsQuantized(QuantizationType qt) =>
        qt is not QuantizationType.F16 and not QuantizationType.F32;

    /// <summary>
    /// Whether to skip the persistent FP16 copy for this quant type.
    /// Only skip when we have BOTH a custom quantized GEMV kernel (for decode)
    /// AND a dequant-to-F16 kernel (for on-the-fly prefill GEMM via scratch buffer).
    /// Types without a custom GEMV (Q5_0, Q4_0, Q5_K) keep persistent FP16
    /// because the scratch buffer approach requires cuBLAS fallback.
    /// </summary>
    private static bool SkipFp16(QuantizationType qt) =>
        CudaKernels.HasQuantizedGemv(qt); // Q8_0, Q4_K, Q6_K only

    /// <summary>Allocate device memory and copy host data.</summary>
    private static nint AllocAndUpload(nint hostPtr, long bytes, List<nint> allocs)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)bytes).ThrowOnError();
        allocs.Add(devPtr);
        CudaDriverApi.cuMemcpyHtoD_v2(devPtr, hostPtr, (nuint)bytes).ThrowOnError();
        return devPtr;
    }

    public void Dispose()
    {
        foreach (nint ptr in _allAllocations)
        {
            if (ptr != 0)
                CudaDriverApi.cuMemFree_v2(ptr);
        }
        _allAllocations.Clear();
    }
}
