using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-layer F32 weight buffers on a Vulkan device. Mirrors
/// <c>DotLLM.Cuda.CudaWeights</c> but with a simpler storage model:
/// all weights are dequantized to FP32 at load time (the Vulkan kernel set
/// is F32-only in this wave — no quantized GEMV yet). Bias tensors are
/// uploaded as FP32 buffers; norm weights become FP32 device buffers.
/// </summary>
internal sealed class VulkanWeights : IDisposable
{
    internal readonly struct LayerBuffers
    {
        public readonly VulkanDevice.Buffer AttnNormWeight;

        public readonly VulkanDevice.Buffer Q;
        public readonly VulkanDevice.Buffer K;
        public readonly VulkanDevice.Buffer V;
        public readonly VulkanDevice.Buffer O;
        public readonly int QOutputDim, QInputDim;
        public readonly int KOutputDim, KInputDim;
        public readonly int VOutputDim, VInputDim;
        public readonly int OOutputDim, OInputDim;

        public readonly VulkanDevice.Buffer? QBias, KBias, VBias, OBias;

        public readonly VulkanDevice.Buffer FfnNormWeight;

        public readonly VulkanDevice.Buffer Gate;
        public readonly VulkanDevice.Buffer Up;
        public readonly VulkanDevice.Buffer Down;
        public readonly int GateOutputDim, GateInputDim;
        public readonly int UpOutputDim, UpInputDim;
        public readonly int DownOutputDim, DownInputDim;

        public readonly VulkanDevice.Buffer? GateBias, UpBias, DownBias;

        public LayerBuffers(
            VulkanDevice.Buffer attnNorm,
            VulkanDevice.Buffer q, int qM, int qK,
            VulkanDevice.Buffer k, int kM, int kK,
            VulkanDevice.Buffer v, int vM, int vK,
            VulkanDevice.Buffer o, int oM, int oK,
            VulkanDevice.Buffer? qBias, VulkanDevice.Buffer? kBias, VulkanDevice.Buffer? vBias, VulkanDevice.Buffer? oBias,
            VulkanDevice.Buffer ffnNorm,
            VulkanDevice.Buffer gate, int gateM, int gateK,
            VulkanDevice.Buffer up, int upM, int upK,
            VulkanDevice.Buffer down, int downM, int downK,
            VulkanDevice.Buffer? gateBias, VulkanDevice.Buffer? upBias, VulkanDevice.Buffer? downBias)
        {
            AttnNormWeight = attnNorm;
            Q = q; QOutputDim = qM; QInputDim = qK;
            K = k; KOutputDim = kM; KInputDim = kK;
            V = v; VOutputDim = vM; VInputDim = vK;
            O = o; OOutputDim = oM; OInputDim = oK;
            QBias = qBias; KBias = kBias; VBias = vBias; OBias = oBias;
            FfnNormWeight = ffnNorm;
            Gate = gate; GateOutputDim = gateM; GateInputDim = gateK;
            Up = up; UpOutputDim = upM; UpInputDim = upK;
            Down = down; DownOutputDim = downM; DownInputDim = downK;
            GateBias = gateBias; UpBias = upBias; DownBias = downBias;
        }

        public void Dispose()
        {
            AttnNormWeight.Dispose();
            Q.Dispose(); K.Dispose(); V.Dispose(); O.Dispose();
            QBias?.Dispose(); KBias?.Dispose(); VBias?.Dispose(); OBias?.Dispose();
            FfnNormWeight.Dispose();
            Gate.Dispose(); Up.Dispose(); Down.Dispose();
            GateBias?.Dispose(); UpBias?.Dispose(); DownBias?.Dispose();
        }
    }

    private readonly VulkanDevice _device;
    private readonly LayerBuffers[] _layers;

    public LayerBuffers[] Layers => _layers;
    public VulkanDevice.Buffer TokenEmbedding { get; }
    public int VocabSize { get; }
    public int HiddenSize { get; }

    public VulkanDevice.Buffer OutputNormWeight { get; }
    public VulkanDevice.Buffer OutputWeight { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }

    public long AllocatedBytes { get; private set; }

    private VulkanWeights(
        VulkanDevice device,
        VulkanDevice.Buffer tokenEmbed, int vocabSize, int hiddenSize,
        LayerBuffers[] layers,
        VulkanDevice.Buffer outputNormWeight,
        VulkanDevice.Buffer outputWeight, int outputM, int outputK,
        long allocatedBytes)
    {
        _device = device;
        TokenEmbedding = tokenEmbed;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        _layers = layers;
        OutputNormWeight = outputNormWeight;
        OutputWeight = outputWeight;
        OutputOutputDim = outputM;
        OutputInputDim = outputK;
        AllocatedBytes = allocatedBytes;
    }

    /// <summary>
    /// Uploads the given CPU-resident <see cref="TransformerWeights"/> to the
    /// Vulkan device as immutable device-local buffers. Weights are staged
    /// through a single reusable host-visible staging buffer (sized to the
    /// largest single matrix), dequantized to FP32 as needed, and copied
    /// via <c>vkCmdCopyBuffer</c> to VRAM.
    /// </summary>
    /// <remarks>
    /// <para>
    /// On both discrete and UMA parts, a DEVICE_LOCAL-only memory type lets
    /// the driver pick a tiled / swizzled layout that is substantially
    /// faster to read from a compute shader than the host-coherent linear
    /// memory the scaffold used. See <see cref="VulkanDevice.AllocateDeviceLocal"/>.
    /// </para>
    /// <para>
    /// The staging buffer is sized at construction to fit the widest matrix
    /// (typically the LM head at <c>vocab × hidden</c>) so weight upload is
    /// one staging copy per matrix — no malloc/free loop per row.
    /// </para>
    /// </remarks>
    public static VulkanWeights Upload(VulkanDevice device, TransformerWeights weights, int numLayers)
    {
        long totalBytes = 0;

        // Size the reusable staging buffer to the largest single weight upload.
        long stagingBytes = ComputeMaxMatrixBytes(weights, numLayers);
        using var staging = device.Allocate(stagingBytes);

        // Token embedding table: [vocabSize, hiddenSize] FP32.
        var tokenEmbed = UploadMatrix(device, staging, weights.TokenEmbedWeight, weights.TokenEmbedQuantType,
            weights.VocabSize, weights.HiddenSize);
        totalBytes += (long)weights.VocabSize * weights.HiddenSize * sizeof(float);

        var layerBuffers = new LayerBuffers[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            ref readonly var lw = ref weights.Layers[i];

            var attnNorm = UploadNormVec(device, staging, lw.AttnNormWeight);

            var q = UploadMatrix(device, staging, lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim);
            var k = UploadMatrix(device, staging, lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim);
            var v = UploadMatrix(device, staging, lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim);
            var o = UploadMatrix(device, staging, lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim);

            var qBias = UploadOptionalVec(device, staging, lw.QBias);
            var kBias = UploadOptionalVec(device, staging, lw.KBias);
            var vBias = UploadOptionalVec(device, staging, lw.VBias);
            var oBias = UploadOptionalVec(device, staging, lw.OBias);

            var ffnNorm = UploadNormVec(device, staging, lw.FfnNormWeight);

            var gate = UploadMatrix(device, staging, lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim);
            var up = UploadMatrix(device, staging, lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim);
            var down = UploadMatrix(device, staging, lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim);

            var gateBias = UploadOptionalVec(device, staging, lw.GateBias);
            var upBias = UploadOptionalVec(device, staging, lw.UpBias);
            var downBias = UploadOptionalVec(device, staging, lw.DownBias);

            layerBuffers[i] = new LayerBuffers(
                attnNorm,
                q, lw.QOutputDim, lw.QInputDim,
                k, lw.KOutputDim, lw.KInputDim,
                v, lw.VOutputDim, lw.VInputDim,
                o, lw.OOutputDim, lw.OInputDim,
                qBias, kBias, vBias, oBias,
                ffnNorm,
                gate, lw.GateOutputDim, lw.GateInputDim,
                up, lw.UpOutputDim, lw.UpInputDim,
                down, lw.DownOutputDim, lw.DownInputDim,
                gateBias, upBias, downBias);

            totalBytes += (long)lw.QOutputDim * lw.QInputDim * sizeof(float);
            totalBytes += (long)lw.KOutputDim * lw.KInputDim * sizeof(float);
            totalBytes += (long)lw.VOutputDim * lw.VInputDim * sizeof(float);
            totalBytes += (long)lw.OOutputDim * lw.OInputDim * sizeof(float);
            totalBytes += (long)lw.GateOutputDim * lw.GateInputDim * sizeof(float);
            totalBytes += (long)lw.UpOutputDim * lw.UpInputDim * sizeof(float);
            totalBytes += (long)lw.DownOutputDim * lw.DownInputDim * sizeof(float);
        }

        var outputNorm = UploadNormVec(device, staging, weights.OutputNormWeight);
        var outputWeight = UploadMatrix(device, staging, weights.OutputWeight, weights.OutputQuantType,
            weights.OutputOutputDim, weights.OutputInputDim);
        totalBytes += (long)weights.OutputOutputDim * weights.OutputInputDim * sizeof(float);

        return new VulkanWeights(
            device, tokenEmbed, weights.VocabSize, weights.HiddenSize,
            layerBuffers,
            outputNorm, outputWeight, weights.OutputOutputDim, weights.OutputInputDim,
            totalBytes);
    }

    private static long ComputeMaxMatrixBytes(TransformerWeights weights, int numLayers)
    {
        long max = (long)weights.VocabSize * weights.HiddenSize;
        max = Math.Max(max, (long)weights.OutputOutputDim * weights.OutputInputDim);
        for (int i = 0; i < numLayers; i++)
        {
            ref readonly var lw = ref weights.Layers[i];
            max = Math.Max(max, (long)lw.QOutputDim * lw.QInputDim);
            max = Math.Max(max, (long)lw.KOutputDim * lw.KInputDim);
            max = Math.Max(max, (long)lw.VOutputDim * lw.VInputDim);
            max = Math.Max(max, (long)lw.OOutputDim * lw.OInputDim);
            max = Math.Max(max, (long)lw.GateOutputDim * lw.GateInputDim);
            max = Math.Max(max, (long)lw.UpOutputDim * lw.UpInputDim);
            max = Math.Max(max, (long)lw.DownOutputDim * lw.DownInputDim);
        }
        return max * sizeof(float);
    }

    private static unsafe VulkanDevice.Buffer UploadMatrix(
        VulkanDevice device, VulkanDevice.Buffer staging,
        nint srcPtr, QuantizationType qt, int outputDim, int inputDim)
    {
        long elems = (long)outputDim * inputDim;
        long bytes = elems * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);

        // 1. Write FP32 bytes into the host-visible staging buffer (dequantizing if needed).
        DotLLM.Vulkan.Interop.VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanWeights.UploadMatrix staging");
        try
        {
            float* d = (float*)mapped;
            if (qt == QuantizationType.F32)
            {
                // Bulk copy from mmap → staging.
                new ReadOnlySpan<float>((void*)srcPtr, checked((int)elems))
                    .CopyTo(new Span<float>(d, checked((int)elems)));
            }
            else
            {
                long rowBytes = Dequantize.RowByteSize(inputDim, qt);
                for (int row = 0; row < outputDim; row++)
                {
                    nint rowSrc = srcPtr + (nint)(row * rowBytes);
                    Dequantize.ToFloat32(rowSrc, inputDim, qt,
                        new Span<float>(d + (long)row * inputDim, inputDim));
                }
            }
        }
        finally
        {
            DotLLM.Vulkan.Interop.VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }

        // 2. Record + submit vkCmdCopyBuffer(staging → device-local), wait on fence.
        device.CopyBufferSynchronous(staging, buf, (ulong)bytes);
        return buf;
    }

    private static unsafe VulkanDevice.Buffer UploadNormVec(
        VulkanDevice device, VulkanDevice.Buffer staging, float[] normWeight)
    {
        long bytes = (long)normWeight.Length * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);

        DotLLM.Vulkan.Interop.VulkanApi.vkMapMemory(device.Handle, staging.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory VulkanWeights.UploadNormVec staging");
        try
        {
            normWeight.AsSpan().CopyTo(new Span<float>((void*)mapped, normWeight.Length));
        }
        finally
        {
            DotLLM.Vulkan.Interop.VulkanApi.vkUnmapMemory(device.Handle, staging.Memory);
        }

        device.CopyBufferSynchronous(staging, buf, (ulong)bytes);
        return buf;
    }

    private static VulkanDevice.Buffer? UploadOptionalVec(
        VulkanDevice device, VulkanDevice.Buffer staging, float[]? vec)
    {
        if (vec is null) return null;
        return UploadNormVec(device, staging, vec);
    }

    public void Dispose()
    {
        TokenEmbedding.Dispose();
        OutputNormWeight.Dispose();
        OutputWeight.Dispose();
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].Dispose();
    }
}
