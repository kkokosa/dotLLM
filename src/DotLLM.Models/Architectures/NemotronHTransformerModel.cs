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
/// </summary>
public sealed unsafe class NemotronHTransformerModel : IModel
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int Q8_1GroupSize = 32;

    private readonly GgufFile _gguf; // keep alive
    private readonly NemotronHLayerWeights[] _layers;
    private readonly float[] _outputNormWeight;

    private readonly nint _tokenEmbedWeight;
    private readonly QuantizationType _tokenEmbedQuantType;
    private readonly nint _outputWeight;
    private readonly QuantizationType _outputQuantType;
    private readonly int _outputOutputDim;
    private readonly int _outputInputDim;

    private readonly HybridLayerLayout _layout;
    private readonly MambaSsmConfig _ssm;

    // Sparse KV-cache mapping: physical layer index -> slot in a cache sized to
    // the attention-layer count. -1 for non-attention layers.
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;

    // Ordinal mapping: physical layer index -> index into _ssmCache for SSM layers; -1 otherwise.
    private readonly int[] _ssmLayerOrdinal;
    private readonly int _numSsmLayers;

    private readonly float[] _ropeCosTable;
    private readonly float[] _ropeSinTable;
    private readonly int _ropeDim;

    private readonly NemotronHForwardState _state;
    private readonly SsmStateCache _ssmCache;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _ssmCache.AllocatedBytes;

    /// <summary>Number of attention layers — the matching sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    private NemotronHTransformerModel(
        ModelConfig config,
        GgufFile gguf,
        NemotronHLayerWeights[] layers,
        float[] outputNormWeight,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim,
        int[] kvSlotForLayer, int attentionLayerCount,
        float[] ropeCosTable, float[] ropeSinTable, int ropeDim)
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
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;
        _ropeCosTable = ropeCosTable;
        _ropeSinTable = ropeSinTable;
        _ropeDim = ropeDim;

        _ssmLayerOrdinal = new int[config.NumLayers];
        int ssmOrdinal = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            _ssmLayerOrdinal[i] = _layout.LayerKind[i] == HybridLayerKind.Ssm
                ? ssmOrdinal++
                : -1;
        }
        _numSsmLayers = ssmOrdinal;

        int maxIntermediate = 0;
        for (int i = 0; i < _layers.Length; i++)
        {
            var ffn = _layers[i].Ffn;
            if (ffn is not null && ffn.UpOutputDim > maxIntermediate)
                maxIntermediate = ffn.UpOutputDim;
        }
        if (maxIntermediate == 0) maxIntermediate = config.HiddenSize;

        _state = new NemotronHForwardState(
            hiddenSize: config.HiddenSize,
            maxIntermediateSize: maxIntermediate,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            inputProjectionDim: _ssm.InputProjectionDim,
            convDim: _ssm.ConvDim,
            dConv: _ssm.DConv,
            dInner: _ssm.DInner,
            nHead: _ssm.NHead,
            nGroup: _ssm.NGroup,
            dState: _ssm.DState);

        _ssmCache = new SsmStateCache(_ssm, _numSsmLayers);
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

        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase + (nint)embDesc.DataOffset;

        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormWeight = DequantizeF32(dataBase, outNormDesc, config.HiddenSize);

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
            outputPtr = embPtr;
            outputQt = embDesc.QuantizationType;
            outputK = embDesc.Shape[0];
            outputM = embDesc.Shape[1];
        }

        int ropeDim = config.RoPEConfig?.DimensionCount ?? 0;
        if (ropeDim <= 0)
            throw new InvalidDataException(
                "NemotronH requires rope.dimension_count in GGUF metadata (expected 78 for Nemotron-3).");

        var layers = new NemotronHLayerWeights[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayer(i, dataBase, tensors, config, layout, ssm);

            if (layout.LayerKind[i] == HybridLayerKind.Attention)
            {
                // RoPE preconditions: even, ≤ head_dim. A strict rope_dim == 78 check was the
                // Nemotron-3 4B Q4_K_M case from DESIGN.md §2, but other nemotron_h variants
                // (e.g., the Ollama nemotron-3-nano 4B) use full-head RoPE with rope_dim == head_dim.
                if ((ropeDim & 1) != 0)
                    throw new InvalidDataException(
                        $"NemotronH rope_dim={ropeDim} must be even for pair-wise rotation.");
                if (ropeDim > config.HeadDim)
                    throw new InvalidDataException(
                        $"NemotronH attention layer {i}: rope_dim={ropeDim} exceeds head_dim={config.HeadDim}.");

                kvSlotForLayer[i] = attentionLayerCount++;
            }
            else
            {
                kvSlotForLayer[i] = -1;
            }
        }

        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        int halfRope = ropeDim / 2;
        var ropeCos = new float[config.MaxSequenceLength * halfRope];
        var ropeSin = new float[config.MaxSequenceLength * halfRope];
        RoPE.PrecomputeFrequencyTable(config.MaxSequenceLength, ropeDim, ropeTheta, ropeCos, ropeSin);

        return new NemotronHTransformerModel(
            config, gguf, layers, outputNormWeight,
            embPtr, embDesc.QuantizationType,
            outputPtr, outputQt, outputM, outputK,
            kvSlotForLayer, attentionLayerCount,
            ropeCos, ropeSin, ropeDim);
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
        float[] attnNormWeight = DequantizeF32(dataBase, attnNormDesc, hiddenSize);

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
        var conv1dWDesc = tensors[$"{prefix}.ssm_conv1d.weight"];
        var conv1dBDesc = tensors[$"{prefix}.ssm_conv1d.bias"];
        var aDesc = tensors[$"{prefix}.ssm_a"];
        var dDesc = tensors[$"{prefix}.ssm_d"];
        var dtBDesc = tensors[$"{prefix}.ssm_dt.bias"];
        var normDesc = tensors[$"{prefix}.ssm_norm.weight"];
        var outDesc = tensors[$"{prefix}.ssm_out.weight"];

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
        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
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
        float* q = (float*)_state.QScratch;
        float* k = (float*)_state.KScratch;
        float* v = (float*)_state.VScratch;
        float* attnOut = (float*)_state.AttnOutput;

        EmbedTokens(tokenIds, hidden, hiddenSize);

        if (NemotronHDiagnostics.TraceLayers)
            NemotronHDiagnostics.DumpStats("embed[last]", hidden + (seqLen - 1) * hiddenSize, hiddenSize);

        var kinds = _layout.LayerKind;
        for (int layer = 0; layer < _layers.Length; layer++)
        {
            var lw = _layers[layer];

            // Save residual snapshot and pre-norm into normOut (shared by all three sub-layers).
            new Span<float>(hidden, seqLen * hiddenSize).CopyTo(new Span<float>(residual, seqLen * hiddenSize));
            for (int t = 0; t < seqLen; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden + t * hiddenSize, hiddenSize),
                    lw.AttnNormWeight, eps,
                    new Span<float>(normOut + t * hiddenSize, hiddenSize));
            }

            char kindTag;
            switch (kinds[layer])
            {
                case HybridLayerKind.Ffn:
                    ForwardFfnBody(lw.Ffn!, seqLen, hiddenSize, normOut, ffnMid, inputQ8Scratch);
                    kindTag = 'F';
                    break;
                case HybridLayerKind.Attention:
                    ForwardAttentionBody(lw.Attention!, layer, seqLen, positions,
                        normOut, q, k, v, attnOut,
                        numHeads, numKvHeads, headDim, kvCache);
                    kindTag = 'A';
                    break;
                case HybridLayerKind.Ssm:
                    ForwardSsmBody(lw.Ssm!, layer, seqLen, hiddenSize, normOut, eps);
                    kindTag = 'S';
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Unknown HybridLayerKind {kinds[layer]} at layer {layer}.");
            }

            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual + t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(normOut + t * hiddenSize, hiddenSize),
                    new Span<float>(hidden + t * hiddenSize, hiddenSize));
            }

            if (NemotronHDiagnostics.TraceLayers)
                NemotronHDiagnostics.DumpStats($"L{layer:D2}{kindTag}[last]", hidden + (seqLen - 1) * hiddenSize, hiddenSize);
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
    private static void ForwardFfnBody(NemotronHFfnWeights ffn, int seqLen, int hiddenSize,
                                       float* normOut, float* ffnMid, byte* inputQ8Scratch)
    {
        int intermediateSize = ffn.UpOutputDim;

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
    }

    private void ForwardAttentionBody(
        NemotronHAttentionWeights attn, int layer, int seqLen, ReadOnlySpan<int> positions,
        float* normOut, float* q, float* k, float* v, float* attnOut,
        int numHeads, int numKvHeads, int headDim, IKvCache? kvCache)
    {
        int kvStride = numKvHeads * headDim;

        Gemm(attn.QWeight, attn.QQuantType, normOut, q, attn.QOutputDim, attn.QInputDim, seqLen, preQuantizedInput: null);
        Gemm(attn.KWeight, attn.KQuantType, normOut, k, attn.KOutputDim, attn.KInputDim, seqLen, preQuantizedInput: null);
        Gemm(attn.VWeight, attn.VQuantType, normOut, v, attn.VOutputDim, attn.VInputDim, seqLen, preQuantizedInput: null);

        // Partial RoPE: rotates the first _ropeDim=78 dims of each head, leaves the remainder untouched.
        RoPE.Execute(
            new Span<float>(q, seqLen * numHeads * headDim),
            new Span<float>(k, seqLen * kvStride),
            positions,
            numHeads, numKvHeads, headDim, _ropeDim,
            _ropeCosTable, _ropeSinTable, RoPEType.Norm);

        if (kvCache is not null)
        {
            int kvSlot = _kvSlotForLayer[layer];
            if (kvSlot < 0)
                throw new InvalidOperationException(
                    $"Layer {layer} has no KV-cache slot (not an attention layer).");

            var kRef = new TensorRef(seqLen, kvStride, DType.Float32, -1, (nint)k);
            var vRef = new TensorRef(seqLen, kvStride, DType.Float32, -1, (nint)v);
            kvCache.Update(kRef, vRef, positions, kvSlot);

            int seqKv = kvCache.CurrentLength;
            var cachedK = kvCache.GetKeysRef(kvSlot);
            var cachedV = kvCache.GetValuesRef(kvSlot);

            Attention.Execute(q, (float*)cachedK.DataPointer, (float*)cachedV.DataPointer, attnOut,
                seqLen, seqKv, numHeads, numKvHeads, headDim, positions[0], pool: null,
                slidingWindowSize: null);
        }
        else
        {
            Attention.Execute(q, k, v, attnOut,
                seqLen, seqLen, numHeads, numKvHeads, headDim, 0, pool: null,
                slidingWindowSize: null);
        }

        Gemm(attn.OWeight, attn.OQuantType, attnOut, normOut, attn.OOutputDim, attn.OInputDim, seqLen, preQuantizedInput: null);
    }

    /// <summary>
    /// Mamba2 SSM sub-layer forward (DESIGN.md §4). Reads pre-normed activations from
    /// <paramref name="normOut"/> and writes the ssm_out projection back to the same buffer.
    /// Advances the per-layer conv/SSM state in place.
    /// </summary>
    [SkipLocalsInit]
    private void ForwardSsmBody(NemotronHSsmWeights ssmW, int absoluteLayerIndex, int seqLen,
                                int hiddenSize, float* normOut, float eps)
    {
        bool traceSsm = NemotronHDiagnostics.IsSsmTraced(absoluteLayerIndex);
        int ssmOrdinal = _ssmLayerOrdinal[absoluteLayerIndex];

        int dInner = _ssm.DInner;
        int dConv = _ssm.DConv;
        int nHead = _ssm.NHead;
        int headDim = _ssm.HeadDim;
        int dState = _ssm.DState;
        int nGroup = _ssm.NGroup;
        int convDim = _ssm.ConvDim;
        int groupDim = dInner / nGroup;
        int inProjDim = _ssm.InputProjectionDim;

        float* zxbcdt = (float*)_state.Zxbcdt;
        float* convInput = (float*)_state.ConvInput;
        float* xbc = (float*)_state.XBC;
        float* dtBuf = (float*)_state.DtBuffer;
        float* xBuf = (float*)_state.SsmX;
        float* bBuf = (float*)_state.SsmB;
        float* cBuf = (float*)_state.SsmC;
        float* yBuf = (float*)_state.SsmY;
        int bcDim = nGroup * dState;

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.normOut[last]",
                normOut + (seqLen - 1) * hiddenSize, hiddenSize);

        // 1. ssm_in GEMM
        Gemm(ssmW.InWeight, ssmW.InQuantType, normOut, zxbcdt,
             inProjDim, hiddenSize, seqLen, preQuantizedInput: null);

        if (traceSsm)
        {
            int off = (seqLen - 1) * inProjDim;
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.z[last]",
                zxbcdt + off, dInner);
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.xBC[last]",
                zxbcdt + off + dInner, convDim);
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.dt[last]",
                zxbcdt + off + 2 * dInner + 2 * nGroup * dState, nHead);
        }

        // 2. conv_input = concat(conv_state, xBC rows from zxbcdt)
        var convState = _ssmCache.GetConvState(ssmOrdinal);
        convState.CopyTo(new Span<float>(convInput, (dConv - 1) * convDim));
        for (int t = 0; t < seqLen; t++)
        {
            var src = new ReadOnlySpan<float>(zxbcdt + t * inProjDim + dInner, convDim);
            var dst = new Span<float>(convInput + (dConv - 1 + t) * convDim, convDim);
            src.CopyTo(dst);
        }

        // 3. Conv1d + SiLU
        int convInputElems = (dConv - 1 + seqLen) * convDim;
        int xbcElems = seqLen * convDim;
        Conv1dCausal.Execute(
            input: new ReadOnlySpan<float>(convInput, convInputElems),
            weight: ssmW.Conv1dWeight,
            bias: ssmW.Conv1dBias,
            output: new Span<float>(xbc, xbcElems),
            dConv: dConv,
            channels: convDim,
            seqLen: seqLen);

        SiLu.Execute(
            new ReadOnlySpan<float>(xbc, xbcElems),
            new Span<float>(xbc, xbcElems));

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.xbc_post_silu[last]",
                xbc + (seqLen - 1) * convDim, convDim);

        // 4. Save last (d_conv-1) rows of conv_input (pre-SiLU) back into conv_state.
        for (int r = 0; r < dConv - 1; r++)
        {
            var src = new ReadOnlySpan<float>(
                convInput + (seqLen + r) * convDim, convDim);
            var dst = convState.Slice(r * convDim, convDim);
            src.CopyTo(dst);
        }

        // 5. dt = zxbcdt[:, dtOffset..] + ssm_dt.bias
        int dtOffset = 2 * dInner + 2 * nGroup * dState;
        for (int t = 0; t < seqLen; t++)
        {
            var src = new ReadOnlySpan<float>(zxbcdt + t * inProjDim + dtOffset, nHead);
            var dst = new Span<float>(dtBuf + t * nHead, nHead);
            TensorPrimitives.Add(src, ssmW.DtBias, dst);
        }

        // 6. Selective scan — pack contiguous x/B/C into long-lived state buffers.
        // xbc row layout is [x (dInner) | B (bcDim) | C (bcDim)]; the scan kernel wants each
        // of those as its own contiguous [T, ...] buffer, so split them per token.
        int xElems = seqLen * dInner;
        int bcElems = seqLen * bcDim;

        for (int t = 0; t < seqLen; t++)
        {
            ReadOnlySpan<float> row = new(xbc + t * convDim, convDim);
            row.Slice(0, dInner).CopyTo(new Span<float>(xBuf + t * dInner, dInner));
            row.Slice(dInner, bcDim).CopyTo(new Span<float>(bBuf + t * bcDim, bcDim));
            row.Slice(dInner + bcDim, bcDim).CopyTo(new Span<float>(cBuf + t * bcDim, bcDim));
        }

        var ssmState = _ssmCache.GetSsmState(ssmOrdinal);

        Mamba2SelectiveScan.Execute(
            state: ssmState,
            x: new ReadOnlySpan<float>(xBuf, xElems),
            dt: new ReadOnlySpan<float>(dtBuf, seqLen * nHead),
            a: ssmW.A,
            b: new ReadOnlySpan<float>(bBuf, bcElems),
            c: new ReadOnlySpan<float>(cBuf, bcElems),
            y: new Span<float>(yBuf, xElems),
            nHead: nHead,
            headDim: headDim,
            dState: dState,
            nGroup: nGroup,
            seqLen: seqLen);

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.y_post_scan[last]",
                yBuf + (seqLen - 1) * dInner, dInner);

        // 7. y += x * D[h] (broadcast D per head across head_dim)
        for (int t = 0; t < seqLen; t++)
        {
            int tBase = t * dInner;
            for (int h = 0; h < nHead; h++)
            {
                float dh = ssmW.D[h];
                int rowBase = tBase + h * headDim;
                for (int iHead = 0; iHead < headDim; iHead++)
                {
                    yBuf[rowBase + iHead] += xBuf[rowBase + iHead] * dh;
                }
            }
        }

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.y_post_D[last]",
                yBuf + (seqLen - 1) * dInner, dInner);

        // 8. SwiGLU gating: y = SiLU(z) * y, z = zxbcdt[:, 0..dInner)
        for (int t = 0; t < seqLen; t++)
        {
            var z = new ReadOnlySpan<float>(zxbcdt + t * inProjDim, dInner);
            var yRow = new Span<float>(yBuf + t * dInner, dInner);
            FusedOps.SwiGLU(z, yRow, yRow);
        }

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.y_post_swiglu[last]",
                yBuf + (seqLen - 1) * dInner, dInner);

        // 9. Group RMSNorm
        for (int t = 0; t < seqLen; t++)
        {
            for (int g = 0; g < nGroup; g++)
            {
                int off = t * dInner + g * groupDim;
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(yBuf + off, groupDim),
                    ssmW.NormWeight.AsSpan(g * groupDim, groupDim),
                    eps,
                    new Span<float>(yBuf + off, groupDim));
            }
        }

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.y_post_groupnorm[last]",
                yBuf + (seqLen - 1) * dInner, dInner);

        // 10. ssm_out projection into normOut
        Gemm(ssmW.OutWeight, ssmW.OutQuantType, yBuf, normOut,
             hiddenSize, dInner, seqLen, preQuantizedInput: null);

        if (traceSsm)
            NemotronHDiagnostics.DumpStats($"L{absoluteLayerIndex:D2}S ssm.out[last]",
                normOut + (seqLen - 1) * hiddenSize, hiddenSize);
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
        switch (qt)
        {
            case QuantizationType.Q8_0:
                MatMul.GemmQ8_0((byte*)weights, b, c, m, k, n, preQuantizedInput);
                return;
            case QuantizationType.Q5_0:
                MatMul.GemmQ5_0((byte*)weights, b, c, m, k, n, preQuantizedInput);
                return;
            case QuantizationType.Q4_K:
                MatMul.GemmQ4_K((byte*)weights, b, c, m, k, n, preQuantizedInput);
                return;
            case QuantizationType.Q5_K:
                MatMul.GemmQ5_K((byte*)weights, b, c, m, k, n, preQuantizedInput);
                return;
            case QuantizationType.Q6_K:
                MatMul.GemmQ6_K((byte*)weights, b, c, m, k, n, preQuantizedInput);
                return;
            case QuantizationType.F32:
                MatMul.GemmF32((float*)weights, b, c, m, k, n);
                return;
            case QuantizationType.F16:
                MatMul.GemmF16(weights, b, c, m, k, n);
                return;
            default:
                GemmDequantFallback(weights, qt, b, c, m, k, n);
                return;
        }
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
        _ssmCache.Dispose();
        GC.SuppressFinalize(this);
    }

    private static float[] DequantizeF32(nint dataBase, GgufTensorDescriptor desc, int expectedSize)
    {
        nint ptr = dataBase + (nint)desc.DataOffset;
        float[] result = new float[expectedSize];
        Dequantize.ToFloat32(ptr, expectedSize, desc.QuantizationType, result);
        return result;
    }
}
