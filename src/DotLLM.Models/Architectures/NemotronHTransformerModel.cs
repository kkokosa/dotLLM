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
public sealed class NemotronHTransformerModel : IModel
{
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

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => 0; // scratch buffers not yet allocated (Forward unimplemented)

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
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        // Touch fields so the compiler and readers see they are live and the
        // loader side is fully exercised even before the forward pass lands.
        _ = _tokenEmbedWeight;
        _ = _tokenEmbedQuantType;
        _ = _outputWeight;
        _ = _outputQuantType;
        _ = _outputOutputDim;
        _ = _outputInputDim;
        _ = _outputNormWeight;
        _ = _layers;
        _ = _layout;
        _ = _ssm;
        _ = tokenIds;
        _ = positions;
        _ = deviceId;
        _ = kvCache;

        // TODO(feature/mamba-3 stage 5): implement FFN-only forward.
        // TODO(feature/mamba-3 stage 6): implement GQA attention sub-layer.
        // TODO(feature/mamba-3 stage 7): implement Mamba2 selective scan + conv1d + group RMSNorm
        //     and wire the full hybrid dispatch. See DESIGN.md section 5 and
        //     llama.cpp src/models/mamba-base.cpp :: build_mamba2_layer for the reference.
        throw new NotImplementedException(
            "NemotronHTransformerModel.Forward is not yet implemented. " +
            "Weight loading and dispatch scaffolding are in place; see DESIGN.md and " +
            "the TODOs in this method for the remaining stages.");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        // No owned unmanaged resources yet — _gguf is owned by the caller and
        // weights live in its mmap region. Scratch buffers will be added when
        // Forward is implemented.
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
