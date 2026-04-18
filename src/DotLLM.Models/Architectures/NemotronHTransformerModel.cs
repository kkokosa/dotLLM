using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Nemotron-H (<c>nemotron_h</c>) hybrid SSM + Transformer model.
///
/// Each layer is exactly one of:
///   - Mamba2 selective state-space (SSM),
///   - GQA multi-head attention, or
///   - squared-ReLU MLP (no gate).
/// with a single RMSNorm before and one residual add after.
///
/// This class owns weight loading and layout classification; the Forward pass is
/// implemented in stages — see DESIGN.md and the per-branch TODOs inside
/// <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>.
/// </summary>
public sealed unsafe class NemotronHTransformerModel : IModel
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int Q8_1GroupSize = 32;

    private readonly GgufFile _gguf; // keep alive
    private readonly NemotronHLayerWeights[] _layers;
    private readonly float[] _outputNormWeight;

    // Global weights (embeddings + LM head). Stored as raw pointers.
    private readonly nint _tokenEmbedWeight;
    private readonly QuantizationType _tokenEmbedQuantType;
    private readonly nint _outputWeight;
    private readonly QuantizationType _outputQuantType;
    private readonly int _outputOutputDim;
    private readonly int _outputInputDim;

    private readonly HybridLayerLayout _layout;
    private readonly MambaSsmConfig _ssm;
    private readonly NemotronHForwardState _state;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    private NemotronHTransformerModel(
        ModelConfig config,
        GgufFile gguf,
        NemotronHLayerWeights[] layers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim)
    {
        Config = config;
        _gguf = gguf;
        _layers = layers;
        _outputNormWeight = outputNormWeight;
        _tokenEmbedWeight = tokenEmbedWeight;
        _tokenEmbedQuantType = tokenEmbedQuantType;
        _outputWeight = outputWeight;
        _outputQuantType = outputQuantType;
        _outputOutputDim = outputOutputDim;
        _outputInputDim = outputInputDim;
        _layout = config.HybridLayout!;
        _ssm = config.SsmConfig!.Value;

        int maxIntermediate = 0;
        for (int i = 0; i < _layers.Length; i++)
        {
            var ffn = _layers[i].Ffn;
            if (ffn is not null && ffn.UpOutputDim > maxIntermediate)
                maxIntermediate = ffn.UpOutputDim;
        }
        // Intermediate scratch must be non-zero even for all-SSM debug builds.
        if (maxIntermediate == 0) maxIntermediate = config.HiddenSize;

        _state = new NemotronHForwardState(config.HiddenSize, maxIntermediate, config.VocabSize);
    }

    /// <summary>
    /// Loads a Nemotron-H model from an opened GGUF file. The <paramref name="gguf"/> must
    /// remain alive for the lifetime of the returned model.
    /// </summary>
    public static NemotronHTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config)
    {
        if (config.Architecture != Architecture.NemotronH)
            throw new ArgumentException(
                $"NemotronHTransformerModel requires Architecture.NemotronH, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("NemotronH config must have HybridLayout populated.", nameof(config));
        if (config.SsmConfig is null)
            throw new ArgumentException("NemotronH config must have SsmConfig populated.", nameof(config));

        nint dataBase = gguf.DataBasePointer;
        var tensors = gguf.TensorsByName;
        var layout = config.HybridLayout;
        var ssm = config.SsmConfig.Value;

        // --- global tensors ---
        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase + (nint)embDesc.DataOffset;

        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormWeight = DequantizeNorm(dataBase, outNormDesc, config.HiddenSize);

        nint outputPtr;
        QuantizationType outputQt;
        int outputM, outputK;
        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            outputPtr = dataBase + (nint)outDesc.DataOffset;
            outputQt = outDesc.QuantizationType;
            outputK = outDesc.Shape[0];
            outputM = outDesc.Shape[1];
        }
        else
        {
            // Tied — not observed in the 4B checkpoint, but support it for completeness.
            outputPtr = embPtr;
            outputQt = embDesc.QuantizationType;
            outputK = embDesc.Shape[0];
            outputM = embDesc.Shape[1];
        }

        // --- per-layer ---
        var layers = new NemotronHLayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayer(i, dataBase, tensors, config, layout, ssm);
        }

        return new NemotronHTransformerModel(
            config, gguf, layers, outputNormWeight,
            embPtr, embDesc.QuantizationType,
            outputPtr, outputQt, outputM, outputK);
    }

    private static NemotronHLayerWeights LoadLayer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config,
        HybridLayerLayout layout,
        MambaSsmConfig ssm)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;

        // Pre-sublayer RMSNorm is always named attn_norm in GGUF, even on FFN and SSM layers.
        var attnNormDesc = tensors[$"{prefix}.attn_norm.weight"];
        float[] attnNormWeight = DequantizeNorm(dataBase, attnNormDesc, hiddenSize);

        return layout.LayerKind[layerIdx] switch
        {
            HybridLayerKind.Ssm => new NemotronHLayerWeights
            {
                AttnNormWeight = attnNormWeight,
                Ssm = LoadSsmLayer(prefix, dataBase, tensors, ssm),
            },
            HybridLayerKind.Attention => new NemotronHLayerWeights
            {
                AttnNormWeight = attnNormWeight,
                Attention = LoadAttentionLayer(prefix, dataBase, tensors, config, layout.HeadCountKv[layerIdx]),
            },
            HybridLayerKind.Ffn => new NemotronHLayerWeights
            {
                AttnNormWeight = attnNormWeight,
                Ffn = LoadFfnLayer(prefix, dataBase, tensors, layout.FeedForwardLength[layerIdx]),
            },
            _ => throw new InvalidOperationException(
                $"Unknown HybridLayerKind {layout.LayerKind[layerIdx]} at layer {layerIdx}."),
        };
    }

    private static NemotronHSsmWeights LoadSsmLayer(
        string prefix, nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        MambaSsmConfig ssm)
    {
        var inDesc = tensors[$"{prefix}.ssm_in.weight"];
        var conv1dWDesc = tensors[$"{prefix}.ssm_conv1d.weight"]; // [d_conv, conv_dim]
        var conv1dBDesc = tensors[$"{prefix}.ssm_conv1d.bias"];   // [conv_dim]
        var aDesc = tensors[$"{prefix}.ssm_a"];     // [1, n_head]
        var dDesc = tensors[$"{prefix}.ssm_d"];     // [1, n_head]
        var dtBDesc = tensors[$"{prefix}.ssm_dt.bias"]; // [n_head]
        var normDesc = tensors[$"{prefix}.ssm_norm.weight"]; // [d_inner/n_group, n_group]
        var outDesc = tensors[$"{prefix}.ssm_out.weight"];   // [d_inner, hidden]

        // Dimension sanity — mirror llama.cpp assertions so we fail loudly, not silently.
        if (inDesc.Shape[1] != ssm.InputProjectionDim)
            throw new InvalidDataException(
                $"{prefix}.ssm_in.weight shape[1]={inDesc.Shape[1]}, expected {ssm.InputProjectionDim}.");
        if (conv1dWDesc.Shape[0] != ssm.DConv || conv1dWDesc.Shape[1] != ssm.ConvDim)
            throw new InvalidDataException(
                $"{prefix}.ssm_conv1d.weight shape [{conv1dWDesc.Shape[0]},{conv1dWDesc.Shape[1]}], expected [{ssm.DConv},{ssm.ConvDim}].");
        if (conv1dBDesc.Shape.ElementCount != ssm.ConvDim)
            throw new InvalidDataException(
                $"{prefix}.ssm_conv1d.bias length {conv1dBDesc.Shape.ElementCount}, expected {ssm.ConvDim}.");
        if (aDesc.Shape.ElementCount != ssm.NHead)
            throw new InvalidDataException(
                $"{prefix}.ssm_a length {aDesc.Shape.ElementCount}, expected {ssm.NHead}.");
        if (dDesc.Shape.ElementCount != ssm.NHead)
            throw new InvalidDataException(
                $"{prefix}.ssm_d length {dDesc.Shape.ElementCount}, expected {ssm.NHead}.");
        if (dtBDesc.Shape.ElementCount != ssm.NHead)
            throw new InvalidDataException(
                $"{prefix}.ssm_dt.bias length {dtBDesc.Shape.ElementCount}, expected {ssm.NHead}.");

        return new NemotronHSsmWeights
        {
            InWeight = dataBase + (nint)inDesc.DataOffset,
            InQuantType = inDesc.QuantizationType,
            InInputDim = inDesc.Shape[0],
            InOutputDim = inDesc.Shape[1],
            Conv1dWeight = DequantizeF32(dataBase, conv1dWDesc, ssm.DConv * ssm.ConvDim),
            Conv1dBias = DequantizeF32(dataBase, conv1dBDesc, ssm.ConvDim),
            A = DequantizeF32(dataBase, aDesc, ssm.NHead),
            D = DequantizeF32(dataBase, dDesc, ssm.NHead),
            DtBias = DequantizeF32(dataBase, dtBDesc, ssm.NHead),
            NormWeight = DequantizeF32(dataBase, normDesc, ssm.DInner),
            OutWeight = dataBase + (nint)outDesc.DataOffset,
            OutQuantType = outDesc.QuantizationType,
            OutInputDim = outDesc.Shape[0],
            OutOutputDim = outDesc.Shape[1],
        };
    }

    private static NemotronHAttentionWeights LoadAttentionLayer(
        string prefix, nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config, int numKvHeads)
    {
        var q = tensors[$"{prefix}.attn_q.weight"];
        var k = tensors[$"{prefix}.attn_k.weight"];
        var v = tensors[$"{prefix}.attn_v.weight"];
        var o = tensors[$"{prefix}.attn_output.weight"];

        return new NemotronHAttentionWeights
        {
            QWeight = dataBase + (nint)q.DataOffset,
            QQuantType = q.QuantizationType,
            QInputDim = q.Shape[0],
            QOutputDim = q.Shape[1],

            KWeight = dataBase + (nint)k.DataOffset,
            KQuantType = k.QuantizationType,
            KInputDim = k.Shape[0],
            KOutputDim = k.Shape[1],

            VWeight = dataBase + (nint)v.DataOffset,
            VQuantType = v.QuantizationType,
            VInputDim = v.Shape[0],
            VOutputDim = v.Shape[1],

            OWeight = dataBase + (nint)o.DataOffset,
            OQuantType = o.QuantizationType,
            OInputDim = o.Shape[0],
            OOutputDim = o.Shape[1],

            NumKvHeads = numKvHeads,
        };
    }

    private static NemotronHFfnWeights LoadFfnLayer(
        string prefix, nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        int intermediateSize)
    {
        if (tensors.ContainsKey($"{prefix}.ffn_gate.weight"))
            throw new InvalidDataException(
                $"{prefix}.ffn_gate.weight present — Nemotron-H FFN is non-gated (squared-ReLU). " +
                "This GGUF may be mislabelled or a MoE variant (nemotron_h_moe), which is unsupported.");

        var up = tensors[$"{prefix}.ffn_up.weight"];
        var down = tensors[$"{prefix}.ffn_down.weight"];

        return new NemotronHFfnWeights
        {
            UpWeight = dataBase + (nint)up.DataOffset,
            UpQuantType = up.QuantizationType,
            UpInputDim = up.Shape[0],
            UpOutputDim = up.Shape[1],

            DownWeight = dataBase + (nint)down.DataOffset,
            DownQuantType = down.QuantizationType,
            DownInputDim = down.Shape[0],
            DownOutputDim = down.Shape[1],

            IntermediateSize = intermediateSize,
        };
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    [SkipLocalsInit]
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        _ = _ssm;       // consumed by stage 7
        _ = kvCache;    // consumed by stage 6

        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;

        int maxSeq = Config.MaxSequenceLength;
        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        _state.EnsureCapacity(seqLen);

        float* hidden = (float*)_state.HiddenState;
        float* residual = (float*)_state.Residual;
        float* normOut = (float*)_state.NormOutput;
        float* ffnMid = (float*)_state.FfnIntermediate;
        float* logits = (float*)_state.Logits;
        byte* inputQ8Scratch = (byte*)_state.InputQ8Scratch;

        EmbedTokens(tokenIds, hidden, hiddenSize);

        var kinds = _layout.LayerKind;
        for (int layer = 0; layer < _layers.Length; layer++)
        {
            var lw = _layers[layer];
            switch (kinds[layer])
            {
                case HybridLayerKind.Ffn:
                    ForwardFfnLayer(lw, seqLen, hiddenSize, eps,
                                    hidden, residual, normOut, ffnMid, inputQ8Scratch);
                    break;
                case HybridLayerKind.Ssm:
                case HybridLayerKind.Attention:
                    throw new NotImplementedException(
                        $"Nemotron-H {kinds[layer]} layer (index {layer}) — " +
                        "stage 6/7 of feature/mamba-3");
                default:
                    throw new InvalidOperationException(
                        $"Unknown HybridLayerKind {kinds[layer]} at layer {layer}.");
            }
        }

        for (int t = 0; t < seqLen; t++)
        {
            RmsNorm.Execute(
                new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                _outputNormWeight, eps,
                new Span<float>(hidden + t * hiddenSize, hiddenSize));
        }

        Gemm(_outputWeight, _outputQuantType, hidden, logits,
             _outputOutputDim, _outputInputDim, seqLen, preQuantizedInput: null);

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        new Span<float>(logits, seqLen * vocabSize).CopyTo(
            new Span<float>((void*)result.DataPointer, seqLen * vocabSize));

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ForwardFfnLayer(NemotronHLayerWeights lw, int seqLen, int hiddenSize, float eps,
                                 float* hidden, float* residual, float* normOut, float* ffnMid,
                                 byte* inputQ8Scratch)
    {
        var ffn = lw.Ffn!;
        int intermediateSize = ffn.UpOutputDim;

        new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));

        for (int t = 0; t < seqLen; t++)
        {
            RmsNorm.Execute(
                new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                lw.AttnNormWeight, eps,
                new Span<float>(normOut + t * hiddenSize, hiddenSize));
        }

        // Pre-quantize normed input once because Q-typed matmuls expect a packed input representation.
        byte* preQuantUp = QuantizeInput(normOut, inputQ8Scratch, hiddenSize, seqLen, ffn.UpQuantType);
        Gemm(ffn.UpWeight, ffn.UpQuantType, normOut, ffnMid,
             ffn.UpOutputDim, ffn.UpInputDim, seqLen, preQuantUp);

        for (int t = 0; t < seqLen; t++)
        {
            ReluSquared.Execute(
                new ReadOnlySpan<float>(ffnMid + t * intermediateSize, intermediateSize),
                new Span<float>(ffnMid + t * intermediateSize, intermediateSize));
        }

        byte* preQuantDown = QuantizeInput(ffnMid, inputQ8Scratch, intermediateSize, seqLen, ffn.DownQuantType);
        Gemm(ffn.DownWeight, ffn.DownQuantType, ffnMid, normOut,
             ffn.DownOutputDim, ffn.DownInputDim, seqLen, preQuantDown);

        for (int t = 0; t < seqLen; t++)
        {
            Add.Execute(
                new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                new Span<float>(hidden + t * hiddenSize, hiddenSize));
        }
    }

    private void EmbedTokens(ReadOnlySpan<int> tokenIds, float* hidden, int hiddenSize)
    {
        nint embPtr = _tokenEmbedWeight;
        var qt = _tokenEmbedQuantType;

        for (int t = 0; t < tokenIds.Length; t++)
        {
            int tokenId = tokenIds[t];
            if ((uint)tokenId >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {tokenId} at position {t} is out of range [0, {Config.VocabSize}).");

            float* dest = hidden + t * hiddenSize;
            var destSpan = new Span<float>(dest, hiddenSize);

            if (qt == QuantizationType.F32)
            {
                float* src = (float*)embPtr + (long)tokenId * hiddenSize;
                new ReadOnlySpan<float>(src, hiddenSize).CopyTo(destSpan);
            }
            else if (qt == QuantizationType.F16)
            {
                Half* src = (Half*)embPtr + (long)tokenId * hiddenSize;
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(src, hiddenSize), destSpan);
            }
            else
            {
                long rowBytes = Dequantize.RowByteSize(hiddenSize, qt);
                nint rowPtr = embPtr + (nint)((long)tokenId * rowBytes);
                Dequantize.ToFloat32(rowPtr, hiddenSize, qt, destSpan);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Gemm(nint weights, QuantizationType qt, float* b, float* c,
                              int m, int k, int n, byte* preQuantizedInput)
    {
        if (qt == QuantizationType.Q8_0)
            MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, preQuantizedInput);
        else if (qt == QuantizationType.Q5_0)
            MatMul.GemmQ5_0((byte*)weights, b, c, m, k, n, preQuantizedInput);
        else if (qt == QuantizationType.Q4_K)
            MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, preQuantizedInput);
        else if (qt == QuantizationType.Q5_K)
            MatMul.GemmQ5_K((byte*)weights, b, c, m, k, n, preQuantizedInput);
        else if (qt == QuantizationType.Q6_K)
            MatMul.GemmQ6_K((byte*)weights, b, c, m, k, n, preQuantizedInput);
        else if (qt == QuantizationType.F32)
            MatMul.GemmF32((float*)weights, b, c, m, k, n);
        else if (qt == QuantizationType.F16)
            MatMul.GemmF16(weights, b, c, m, k, n);
        else
            GemmDequantFallback(weights, qt, b, c, m, k, n);
    }

    private static void GemmDequantFallback(nint weights, QuantizationType qt, float* b, float* c,
                                            int m, int k, int n)
    {
        long rowBytes = Dequantize.RowByteSize(k, qt);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int t = 0; t < n; t++)
            {
                var xSpan = new ReadOnlySpan<float>(b + t * k, k);
                for (int i = 0; i < m; i++)
                {
                    Dequantize.ToFloat32(weights + i * (nint)rowBytes, k, qt, rowSpan);
                    c[t * m + i] = TensorPrimitives.Dot(new ReadOnlySpan<float>(rowBuf, 0, k), xSpan);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
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

    /// <inheritdoc/>
    public void Dispose()
    {
        _state.Dispose();
        GC.SuppressFinalize(this);
    }

    private static float[] DequantizeNorm(nint dataBase, GgufTensorDescriptor desc, int expectedSize)
        => DequantizeF32(dataBase, desc, expectedSize);

    private static float[] DequantizeF32(nint dataBase, GgufTensorDescriptor desc, int expectedSize)
    {
        nint ptr = dataBase + (nint)desc.DataOffset;
        float[] result = new float[expectedSize];
        Dequantize.ToFloat32(ptr, expectedSize, desc.QuantizationType, result);
        return result;
    }
}
