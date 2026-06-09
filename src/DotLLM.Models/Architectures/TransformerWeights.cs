using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-layer dense-routing MoE weight bundle. Present on a
/// <see cref="TransformerLayerWeights"/> when the layer replaces its FFN
/// with a Mixtral-convention or Qwen-MoE-convention MoE block. All pointers
/// are F32 row-major — bf16 and F16 tensors are upcast at load time so the
/// MoE kernel can feed <see cref="DotLLM.Cpu.Kernels.MoeSwiGluMlp"/>
/// directly without per-call dequant.
/// </summary>
/// <remarks>
/// <para>
/// Qwen-MoE and DeepSeek-V2/V3 add optional shared-expert pointers — each
/// carried as parallel arrays (<see cref="SharedGateProj"/>, <see cref="SharedUpProj"/>,
/// <see cref="SharedDownProj"/>) of length <see cref="NumSharedExperts"/>.
/// Qwen1.5-MoE ships a single shared expert optionally gated by a
/// <see cref="SharedExpertGate"/> sigmoid; DeepSeek-V2/V3 ships
/// <c>n_shared_experts</c> shared experts (often 1 or 2) and does not gate.
/// When <see cref="HasSharedExpert"/> is true, the forward pass runs each
/// shared expert as a dense SwiGLU over the token, sums their outputs, and
/// adds the (optionally gated) sum to the routed top-k sum. The
/// <see cref="NormTopKProb"/> flag controls whether the selected top-k
/// probabilities are renormalised to sum to 1.0 (Mixtral + Qwen3-MoE) or
/// left as raw softmax values (Qwen1.5-MoE-A2.7B).
/// </para>
/// </remarks>
internal sealed class MoeLayerWeights
{
    /// <summary>Router gate.weight as F32 [numExperts, hiddenSize] row-major.</summary>
    public readonly float[] Gate;

    /// <summary>Per-expert <c>w1</c> (gate_proj) F32 pointers [intermediateSize, hiddenSize] row-major.</summary>
    public readonly nint[] W1;

    /// <summary>Per-expert <c>w2</c> (down_proj) F32 pointers [hiddenSize, intermediateSize] row-major.</summary>
    public readonly nint[] W2;

    /// <summary>Per-expert <c>w3</c> (up_proj) F32 pointers [intermediateSize, hiddenSize] row-major.</summary>
    public readonly nint[] W3;

    public readonly int NumExperts;
    public readonly int NumExpertsPerTok;
    public readonly int HiddenSize;
    public readonly int IntermediateSize;

    /// <summary>
    /// When <c>true</c>, the kernel renormalises the selected top-k
    /// probabilities to sum to 1.0 (Mixtral + Qwen3-MoE). When <c>false</c>,
    /// the raw softmax probabilities are used as gating weights (Qwen1.5-MoE).
    /// </summary>
    public readonly bool NormTopKProb;

    /// <summary>
    /// Per-shared-expert <c>gate_proj</c> pointers — F32
    /// [sharedIntermediateSize, hiddenSize] row-major, one per shared expert.
    /// Length equals <see cref="NumSharedExperts"/>; empty when no shared
    /// experts are present.
    /// </summary>
    public readonly nint[] SharedGateProj;
    /// <summary>
    /// Per-shared-expert <c>up_proj</c> pointers — F32
    /// [sharedIntermediateSize, hiddenSize] row-major, one per shared expert.
    /// </summary>
    public readonly nint[] SharedUpProj;
    /// <summary>
    /// Per-shared-expert <c>down_proj</c> pointers — F32
    /// [hiddenSize, sharedIntermediateSize] row-major, one per shared expert.
    /// </summary>
    public readonly nint[] SharedDownProj;
    /// <summary>
    /// Per-shared-expert intermediate width (0 when no shared expert).
    /// Applies uniformly across all shared experts (they share width).
    /// </summary>
    public readonly int SharedIntermediateSize;
    /// <summary>
    /// Number of parallel shared experts whose outputs are summed. 1 for
    /// Qwen1.5-MoE, &gt;=1 for DeepSeek-V2/V3 (<c>n_shared_experts</c>).
    /// Zero only when there is no shared-expert branch.
    /// </summary>
    public readonly int NumSharedExperts;
    /// <summary>
    /// Optional shared-expert sigmoid gate weight — F32 [hiddenSize]. When
    /// present, per-token <c>sigmoid(hidden . SharedExpertGate)</c> scales
    /// the summed shared-expert output before it's added to the routed sum
    /// (Qwen1.5-MoE convention; ALWAYS paired with a single shared expert).
    /// Null = no gate, summed shared-expert output added unscaled
    /// (DeepSeek-V2/V3 convention).
    /// </summary>
    public readonly float[]? SharedExpertGate;

    /// <summary>True iff a shared-expert branch is present on this layer.</summary>
    public bool HasSharedExpert => SharedIntermediateSize > 0 && NumSharedExperts > 0;

    /// <summary>Mixtral-convention ctor (no shared expert, always renormalise top-k).</summary>
    public MoeLayerWeights(
        float[] gate,
        nint[] w1, nint[] w2, nint[] w3,
        int numExperts, int numExpertsPerTok, int hiddenSize, int intermediateSize)
        : this(gate, w1, w2, w3, numExperts, numExpertsPerTok, hiddenSize, intermediateSize,
               normTopKProb: true,
               sharedGateProj: Array.Empty<nint>(),
               sharedUpProj: Array.Empty<nint>(),
               sharedDownProj: Array.Empty<nint>(),
               sharedIntermediateSize: 0,
               sharedExpertGate: null)
    {
    }

    /// <summary>
    /// Full ctor covering Qwen-MoE and DeepSeek extensions: per-shared-expert
    /// pointer arrays, <c>norm_topk_prob</c> flag, optional sigmoid gate.
    /// Length of the three shared arrays must agree; a zero-length array set
    /// disables the shared-expert branch.
    /// </summary>
    public MoeLayerWeights(
        float[] gate,
        nint[] w1, nint[] w2, nint[] w3,
        int numExperts, int numExpertsPerTok, int hiddenSize, int intermediateSize,
        bool normTopKProb,
        nint[] sharedGateProj, nint[] sharedUpProj, nint[] sharedDownProj,
        int sharedIntermediateSize, float[]? sharedExpertGate)
    {
        if (sharedGateProj.Length != sharedUpProj.Length || sharedGateProj.Length != sharedDownProj.Length)
            throw new ArgumentException(
                "Shared-expert pointer arrays must all have the same length (number of shared experts).");

        Gate = gate;
        W1 = w1; W2 = w2; W3 = w3;
        NumExperts = numExperts;
        NumExpertsPerTok = numExpertsPerTok;
        HiddenSize = hiddenSize;
        IntermediateSize = intermediateSize;
        NormTopKProb = normTopKProb;
        SharedGateProj = sharedGateProj;
        SharedUpProj = sharedUpProj;
        SharedDownProj = sharedDownProj;
        SharedIntermediateSize = sharedIntermediateSize;
        NumSharedExperts = sharedGateProj.Length;
        SharedExpertGate = sharedExpertGate;
    }
}

/// <summary>
/// Holds per-layer weight references for a single transformer layer.
/// Norm weights are dequantized to <c>float[]</c> at load time (small).
/// Linear projection weights remain as mmap pointers with their quantization type.
/// Bias arrays are nullable — null when the model has no biases (e.g. standard Llama/Mistral).
/// </summary>
internal readonly struct TransformerLayerWeights
{
    /// <summary>Pre-attention RMSNorm weight [hiddenSize].</summary>
    public readonly float[] AttnNormWeight;

    /// <summary>Optional QK-norm weight [headDim]. Applied per-head to Q after projection, before RoPE. Null when absent (e.g. Qwen2, Llama).</summary>
    public readonly float[]? QNormWeight;
    /// <summary>Optional QK-norm weight [headDim]. Applied per-head to K after projection, before RoPE. Null when absent.</summary>
    public readonly float[]? KNormWeight;

    /// <summary>Q projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint QWeight;
    public readonly QuantizationType QQuantType;
    public readonly int QOutputDim;
    public readonly int QInputDim;
    /// <summary>Optional Q projection bias [QOutputDim]. Null when absent.</summary>
    public readonly float[]? QBias;

    /// <summary>K projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint KWeight;
    public readonly QuantizationType KQuantType;
    public readonly int KOutputDim;
    public readonly int KInputDim;
    /// <summary>Optional K projection bias [KOutputDim]. Null when absent.</summary>
    public readonly float[]? KBias;

    /// <summary>V projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint VWeight;
    public readonly QuantizationType VQuantType;
    public readonly int VOutputDim;
    public readonly int VInputDim;
    /// <summary>Optional V projection bias [VOutputDim]. Null when absent.</summary>
    public readonly float[]? VBias;

    /// <summary>Output projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint OWeight;
    public readonly QuantizationType OQuantType;
    public readonly int OOutputDim;
    public readonly int OInputDim;
    /// <summary>Optional output projection bias [OOutputDim]. Null when absent.</summary>
    public readonly float[]? OBias;

    /// <summary>Pre-FFN RMSNorm weight [hiddenSize].</summary>
    public readonly float[] FfnNormWeight;

    /// <summary>SwiGLU gate projection.</summary>
    public readonly nint GateWeight;
    public readonly QuantizationType GateQuantType;
    public readonly int GateOutputDim;
    public readonly int GateInputDim;
    /// <summary>Optional gate projection bias [GateOutputDim]. Null when absent.</summary>
    public readonly float[]? GateBias;

    /// <summary>SwiGLU up projection.</summary>
    public readonly nint UpWeight;
    public readonly QuantizationType UpQuantType;
    public readonly int UpOutputDim;
    public readonly int UpInputDim;
    /// <summary>Optional up projection bias [UpOutputDim]. Null when absent.</summary>
    public readonly float[]? UpBias;

    /// <summary>Down projection.</summary>
    public readonly nint DownWeight;
    public readonly QuantizationType DownQuantType;
    public readonly int DownOutputDim;
    public readonly int DownInputDim;
    /// <summary>Optional down projection bias [DownOutputDim]. Null when absent.</summary>
    public readonly float[]? DownBias;

    // ──────────────────────────── MLA attention ────────────────────────────
    // DeepSeek-V2/V3 replaces the monolithic Q/K/V/O projections with a
    // low-rank-factorised set. When <see cref="Mla"/> is non-null, the
    // forward pass routes through MlaAttention and ignores the legacy
    // Q/K/V slots above (O is still used as the output projection).

    /// <summary>
    /// Non-null on DeepSeek-V2/V3 MLA layers. Carries all MLA-specific
    /// projection pointers + hyperparameters (qk nope/rope dims, v_head_dim,
    /// q/kv LoRA ranks). When present, <see cref="QWeight"/>/<see cref="KWeight"/>/
    /// <see cref="VWeight"/> are zeroed and the forward pass takes the MLA branch.
    /// </summary>
    public readonly MlaLayerWeights? Mla;

    /// <summary>
    /// MoE FFN bundle for Mixtral-convention layers. When non-null the dense
    /// <see cref="GateWeight"/>/<see cref="UpWeight"/>/<see cref="DownWeight"/>
    /// slots are ignored by the forward pass and MoE routing runs instead.
    /// </summary>
    public readonly MoeLayerWeights? Moe;

    public TransformerLayerWeights(
        float[] attnNormWeight,
        nint qWeight, QuantizationType qQuantType, int qOutputDim, int qInputDim,
        nint kWeight, QuantizationType kQuantType, int kOutputDim, int kInputDim,
        nint vWeight, QuantizationType vQuantType, int vOutputDim, int vInputDim,
        nint oWeight, QuantizationType oQuantType, int oOutputDim, int oInputDim,
        float[] ffnNormWeight,
        nint gateWeight, QuantizationType gateQuantType, int gateOutputDim, int gateInputDim,
        nint upWeight, QuantizationType upQuantType, int upOutputDim, int upInputDim,
        nint downWeight, QuantizationType downQuantType, int downOutputDim, int downInputDim,
        float[]? qBias = null, float[]? kBias = null, float[]? vBias = null, float[]? oBias = null,
        float[]? gateBias = null, float[]? upBias = null, float[]? downBias = null,
        float[]? qNormWeight = null, float[]? kNormWeight = null,
        MlaLayerWeights? mla = null,
        MoeLayerWeights? moe = null)
    {
        AttnNormWeight = attnNormWeight;
        QNormWeight = qNormWeight;
        KNormWeight = kNormWeight;
        QWeight = qWeight; QQuantType = qQuantType; QOutputDim = qOutputDim; QInputDim = qInputDim; QBias = qBias;
        KWeight = kWeight; KQuantType = kQuantType; KOutputDim = kOutputDim; KInputDim = kInputDim; KBias = kBias;
        VWeight = vWeight; VQuantType = vQuantType; VOutputDim = vOutputDim; VInputDim = vInputDim; VBias = vBias;
        OWeight = oWeight; OQuantType = oQuantType; OOutputDim = oOutputDim; OInputDim = oInputDim; OBias = oBias;
        FfnNormWeight = ffnNormWeight;
        GateWeight = gateWeight; GateQuantType = gateQuantType; GateOutputDim = gateOutputDim; GateInputDim = gateInputDim; GateBias = gateBias;
        UpWeight = upWeight; UpQuantType = upQuantType; UpOutputDim = upOutputDim; UpInputDim = upInputDim; UpBias = upBias;
        DownWeight = downWeight; DownQuantType = downQuantType; DownOutputDim = downOutputDim; DownInputDim = downInputDim; DownBias = downBias;
        Mla = mla;
        Moe = moe;
    }
}

/// <summary>
/// Per-layer MLA (Multi-head Latent Attention) weight bundle for DeepSeek-V2/V3.
/// All projection pointers are F32 row-major — F16 / BF16 tensors are upcast at
/// load time (via <c>ResolveLinearAsF32</c>) so the kernel can consume a uniform
/// F32 layout matching <see cref="DotLLM.Cpu.Kernels.MlaAttention.Execute"/>.
/// </summary>
/// <remarks>
/// <para>
/// Exactly one of the Q paths is populated:
/// <list type="bullet">
///   <item>LoRA-factored Q (<see cref="QLoraRank"/> &gt; 0): <see cref="QAProj"/>,
///     <see cref="QALayernormWeight"/>, <see cref="QBProj"/> are all non-zero;
///     <see cref="QProj"/> is zero.</item>
///   <item>Monolithic Q (<see cref="QLoraRank"/> == 0): <see cref="QProj"/> is
///     non-zero; <see cref="QAProj"/>, <see cref="QBProj"/> are zero and
///     <see cref="QALayernormWeight"/> is null.</item>
/// </list>
/// The KV path is always LoRA-factored (<see cref="KvAProjWithMqa"/>,
/// <see cref="KvALayernormWeight"/>, <see cref="KvBProj"/>).
/// </para>
/// </remarks>
internal sealed class MlaLayerWeights
{
    /// <summary>Q down-projection [qLoraRank, hidden]. Zero when <see cref="QLoraRank"/>==0.</summary>
    public readonly nint QAProj;
    /// <summary>Q LoRA RMSNorm weight [qLoraRank]. Null when <see cref="QLoraRank"/>==0.</summary>
    public readonly float[]? QALayernormWeight;
    /// <summary>Q up-projection [numHeads * qkHeadDim, qLoraRank]. Zero when <see cref="QLoraRank"/>==0.</summary>
    public readonly nint QBProj;
    /// <summary>Monolithic Q projection [numHeads * qkHeadDim, hidden]. Zero when <see cref="QLoraRank"/>&gt;0.</summary>
    public readonly nint QProj;

    /// <summary>KV down-projection with shared-rope-K [kvLoraRank + qkRopeHeadDim, hidden].</summary>
    public readonly nint KvAProjWithMqa;
    /// <summary>KV LoRA RMSNorm weight [kvLoraRank].</summary>
    public readonly float[] KvALayernormWeight;
    /// <summary>KV up-projection [numHeads * (qkNopeHeadDim + vHeadDim), kvLoraRank].</summary>
    public readonly nint KvBProj;

    // Hyperparameters (mirrors MlaConfig, carried on the layer for forward-path convenience).
    public readonly int NumHeads;
    public readonly int QkNopeHeadDim;
    public readonly int QkRopeHeadDim;
    public readonly int VHeadDim;
    public readonly int QLoraRank;
    public readonly int KvLoraRank;

    public MlaLayerWeights(
        nint qAProj, float[]? qALayernormWeight, nint qBProj, nint qProj,
        nint kvAProjWithMqa, float[] kvALayernormWeight, nint kvBProj,
        int numHeads, int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int qLoraRank, int kvLoraRank)
    {
        QAProj = qAProj;
        QALayernormWeight = qALayernormWeight;
        QBProj = qBProj;
        QProj = qProj;
        KvAProjWithMqa = kvAProjWithMqa;
        KvALayernormWeight = kvALayernormWeight;
        KvBProj = kvBProj;
        NumHeads = numHeads;
        QkNopeHeadDim = qkNopeHeadDim;
        QkRopeHeadDim = qkRopeHeadDim;
        VHeadDim = vHeadDim;
        QLoraRank = qLoraRank;
        KvLoraRank = kvLoraRank;
    }
}

/// <summary>
/// Holds R4-interleaved weight buffers for all projections in a single transformer layer.
/// Disposed when the parent <see cref="TransformerWeights"/> is disposed.
/// </summary>
internal sealed class RepackedLayerWeights : IDisposable
{
    public WeightRepacking.RepackedWeight Q, K, V, O, Gate, Up, Down;

    public void Dispose()
    {
        Q.Dispose(); K.Dispose(); V.Dispose(); O.Dispose();
        Gate.Dispose(); Up.Dispose(); Down.Dispose();
    }
}

/// <summary>
/// Organizes all weight tensor references from a loaded GGUF file for a transformer-family model.
/// Norm weights are dequantized to managed <c>float[]</c> at load time.
/// Linear projections remain as raw mmap pointers for zero-copy inference.
/// Optionally holds R4-interleaved weight buffers for improved cache locality in 4-row SIMD kernels.
/// </summary>
internal sealed class TransformerWeights : IDisposable
{
    /// <summary>Token embedding pointer and metadata.</summary>
    public nint TokenEmbedWeight { get; }
    public QuantizationType TokenEmbedQuantType { get; }
    public int VocabSize { get; }
    public int HiddenSize { get; }

    /// <summary>Per-layer weights.</summary>
    public TransformerLayerWeights[] Layers { get; }

    /// <summary>Final RMSNorm weight [hiddenSize].</summary>
    public float[] OutputNormWeight { get; }

    /// <summary>LM head (output projection) pointer and metadata.</summary>
    public nint OutputWeight { get; }
    public QuantizationType OutputQuantType { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }

    /// <summary>Per-layer R4-interleaved weights. Null until <see cref="RepackWeights"/> is called.</summary>
    public RepackedLayerWeights[]? RepackedLayers { get; private set; }

    /// <summary>R4-interleaved LM head weights. Null until <see cref="RepackWeights"/> is called or if type is not repackable.</summary>
    public WeightRepacking.RepackedWeight? RepackedOutput { get; private set; }

    /// <summary>
    /// Loader-owned 64-byte-aligned allocations created at load time (e.g.
    /// bf16 → F32 upcasts for the safetensors path). Freed by
    /// <see cref="Dispose"/>. Empty for pure-mmap GGUF loads.
    /// </summary>
    private readonly List<nint>? _ownedAllocations;

    private TransformerWeights(
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType, int vocabSize, int hiddenSize,
        TransformerLayerWeights[] layers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim,
        List<nint>? ownedAllocations = null)
    {
        TokenEmbedWeight = tokenEmbedWeight;
        TokenEmbedQuantType = tokenEmbedQuantType;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        Layers = layers;
        OutputNormWeight = outputNormWeight;
        OutputWeight = outputWeight;
        OutputQuantType = outputQuantType;
        OutputOutputDim = outputOutputDim;
        OutputInputDim = outputInputDim;
        _ownedAllocations = ownedAllocations;
    }

    /// <summary>
    /// Factory used by the safetensors loader. Wraps the private constructor
    /// and accepts the list of owned allocations (bf16→F32 upcast buffers)
    /// that must be freed when the weights are disposed.
    /// </summary>
    internal static TransformerWeights CreateFromSafetensors(
        nint tokenEmbedWeight, QuantizationType tokenEmbedQt, int vocabSize, int hiddenSize,
        TransformerLayerWeights[] layers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQt, int outputM, int outputK,
        List<nint> ownedAllocations)
    {
        return new TransformerWeights(
            tokenEmbedWeight, tokenEmbedQt, vocabSize, hiddenSize,
            layers,
            outputNormWeight,
            outputWeight, outputQt, outputM, outputK,
            ownedAllocations);
    }

    /// <summary>
    /// Loads all weight references from an opened GGUF file.
    /// Norm weights are dequantized to <c>float[]</c>. Linear projections stay as mmap pointers.
    /// </summary>
    public static TransformerWeights LoadFromGguf(GgufFile gguf, ModelConfig config)
    {
        nint dataBase = gguf.DataBasePointer;
        var tensors = gguf.TensorsByName;

        // Token embeddings
        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase + (nint)embDesc.DataOffset;

        // Per-layer weights
        var layers = new TransformerLayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayer(i, dataBase, tensors, config);
        }

        // Output norm
        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormWeight = DequantizeNorm(dataBase, outNormDesc, config.HiddenSize);

        // LM head — may be tied to token embeddings
        nint outputPtr;
        QuantizationType outputQt;
        int outputM, outputK;

        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            outputPtr = dataBase + (nint)outDesc.DataOffset;
            outputQt = outDesc.QuantizationType;
            // GGUF: Dimensions[0] = input dim (K), Dimensions[1] = output dim (M)
            outputK = outDesc.Shape[0];
            outputM = outDesc.Shape[1];
        }
        else
        {
            // Tied embeddings: alias token_embd.weight
            outputPtr = embPtr;
            outputQt = embDesc.QuantizationType;
            outputK = embDesc.Shape[0];
            outputM = embDesc.Shape[1];
        }

        return new TransformerWeights(
            embPtr, embDesc.QuantizationType, config.VocabSize, config.HiddenSize,
            layers,
            outputNormWeight,
            outputPtr, outputQt, outputM, outputK);
    }

    /// <summary>
    /// Repacks all linear projection weights into R4 interleaved layout for improved
    /// cache locality in 4-row SIMD kernels. Skips token embeddings (random row access)
    /// and non-block-structured types (F32, F16).
    /// </summary>
    public void RepackWeights()
    {
        var repacked = new RepackedLayerWeights[Layers.Length];
        for (int i = 0; i < Layers.Length; i++)
        {
            ref readonly var lw = ref Layers[i];
            // MLA layers don't populate the legacy Q/K/V slots — the MLA forward
            // takes its weights from lw.Mla and calls the scalar MlaAttention
            // kernel which does not consume R4 repacks.
            bool isMla = lw.Mla is not null;
            // MoE layers don't populate the dense gate/up/down slots —
            // repack only the attention projections. The MoE FFN path runs
            // without R4 interleaving (the per-expert GEMMs are tiny and
            // the win would be microscopic).
            bool isMoe = lw.Moe is not null;
            repacked[i] = new RepackedLayerWeights
            {
                Q = isMla ? default : TryRepack(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim),
                K = isMla ? default : TryRepack(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim),
                V = isMla ? default : TryRepack(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim),
                O = isMla ? default : TryRepack(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim),
                Gate = isMoe ? default : TryRepack(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim),
                Up = isMoe ? default : TryRepack(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim),
                Down = isMoe ? default : TryRepack(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim),
            };
        }
        RepackedLayers = repacked;

        if (WeightRepacking.IsRepackable(OutputQuantType))
            RepackedOutput = WeightRepacking.RepackR4(OutputWeight, OutputQuantType, OutputOutputDim, OutputInputDim);
    }

    private static WeightRepacking.RepackedWeight TryRepack(nint ptr, QuantizationType qt, int m, int k)
    {
        if (!WeightRepacking.IsRepackable(qt))
            return default;
        return WeightRepacking.RepackR4(ptr, qt, m, k);
    }

    /// <summary>Frees all R4-interleaved weight buffers and any owned aligned allocations.</summary>
    public unsafe void Dispose()
    {
        if (RepackedLayers is not null)
        {
            foreach (var rl in RepackedLayers)
                rl.Dispose();
            RepackedLayers = null;
        }
        RepackedOutput?.Dispose();
        RepackedOutput = null;

        if (_ownedAllocations is not null)
        {
            foreach (var ptr in _ownedAllocations)
            {
                if (ptr != nint.Zero)
                    NativeMemory.AlignedFree((void*)ptr);
            }
            _ownedAllocations.Clear();
        }
    }

    private static TransformerLayerWeights LoadLayer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;

        // Attention norm — dequantize to float[]
        var attnNormDesc = tensors[$"{prefix}.attn_norm.weight"];
        float[] attnNorm = DequantizeNorm(dataBase, attnNormDesc, hiddenSize);

        // Q/K/V projections — check for fused attn_qkv.weight (Phi-3 style)
        nint qPtr, kPtr, vPtr;
        QuantizationType qQt, kQt, vQt;
        int qM, qK, kM, kK, vM, vK;

        if (tensors.TryGetValue($"{prefix}.attn_qkv.weight", out var qkvDesc))
        {
            // Fused QKV — split by row offset
            nint qkvPtr = dataBase + (nint)qkvDesc.DataOffset;
            int inputDim = qkvDesc.Shape[0]; // hidden_size
            long rowBytes = Dequantize.RowByteSize(inputDim, qkvDesc.QuantizationType);

            int qDim = config.NumAttentionHeads * config.HeadDim;
            int kvDim = config.NumKvHeads * config.HeadDim;

            qPtr = qkvPtr; qQt = qkvDesc.QuantizationType; qM = qDim; qK = inputDim;
            kPtr = qkvPtr + (nint)(qDim * rowBytes); kQt = qkvDesc.QuantizationType; kM = kvDim; kK = inputDim;
            vPtr = qkvPtr + (nint)((qDim + kvDim) * rowBytes); vQt = qkvDesc.QuantizationType; vM = kvDim; vK = inputDim;
        }
        else
        {
            // Separate Q/K/V (standard path)
            (qPtr, qQt, qM, qK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_q.weight"]);
            (kPtr, kQt, kM, kK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_k.weight"]);
            (vPtr, vQt, vM, vK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_v.weight"]);
        }

        var (oPtr, oQt, oM, oK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_output.weight"]);

        // Optional biases — check for fused attn_qkv.bias (Phi-3 style)
        float[]? qBias, kBias, vBias;
        if (tensors.TryGetValue($"{prefix}.attn_qkv.bias", out var qkvBiasDesc))
        {
            // Fused QKV bias — split by element offset
            nint biasPtr = dataBase + (nint)qkvBiasDesc.DataOffset;
            int qDim = config.NumAttentionHeads * config.HeadDim;
            int kvDim = config.NumKvHeads * config.HeadDim;

            qBias = new float[qDim];
            kBias = new float[kvDim];
            vBias = new float[kvDim];

            Dequantize.ToFloat32(biasPtr, qDim, qkvBiasDesc.QuantizationType, qBias);
            Dequantize.ToFloat32(biasPtr + qDim * sizeof(float), kvDim, qkvBiasDesc.QuantizationType, kBias);
            Dequantize.ToFloat32(biasPtr + (qDim + kvDim) * sizeof(float), kvDim, qkvBiasDesc.QuantizationType, vBias);
        }
        else
        {
            qBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_q.bias");
            kBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_k.bias");
            vBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_v.bias");
        }
        float[]? oBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_output.bias");

        // Optional QK-norms (Qwen3-style): per-head RMSNorm applied to Q/K after projection, before RoPE
        float[]? qNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_q_norm.weight", config.HeadDim);
        float[]? kNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_k_norm.weight", config.HeadDim);

        // FFN norm
        var ffnNormDesc = tensors[$"{prefix}.ffn_norm.weight"];
        float[] ffnNorm = DequantizeNorm(dataBase, ffnNormDesc, hiddenSize);

        // FFN projections — check for fused gate+up (Phi-3 style: ffn_up.weight has 2x intermediate rows)
        nint gatePtr, upPtr, downPtr;
        QuantizationType gateQt, upQt, downQt;
        int gateM, gateK, upM, upK, downM, downK;
        float[]? gateBias, upBias, downBias;

        (downPtr, downQt, downM, downK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_down.weight"]);
        downBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_down.bias");

        if (tensors.TryGetValue($"{prefix}.ffn_gate.weight", out var gateDesc))
        {
            // Standard separate gate/up (Llama, Mistral, Qwen)
            (gatePtr, gateQt, gateM, gateK) = LoadLinear(dataBase, gateDesc);
            (upPtr, upQt, upM, upK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_up.weight"]);
            gateBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_gate.bias");
            upBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_up.bias");
        }
        else
        {
            // Fused gate+up in ffn_up.weight (Phi-3 style): output dim = 2 * intermediate_size
            // Split: first intermediate_size rows = gate, next intermediate_size rows = up
            var fusedDesc = tensors[$"{prefix}.ffn_up.weight"];
            nint fusedPtr = dataBase + (nint)fusedDesc.DataOffset;
            int inputDim = fusedDesc.Shape[0]; // hidden_size
            int fusedOutputDim = fusedDesc.Shape[1]; // 2 * intermediate_size
            int halfDim = fusedOutputDim / 2;
            long rowBytes = Dequantize.RowByteSize(inputDim, fusedDesc.QuantizationType);

            gatePtr = fusedPtr; gateQt = fusedDesc.QuantizationType; gateM = halfDim; gateK = inputDim;
            upPtr = fusedPtr + (nint)(halfDim * rowBytes); upQt = fusedDesc.QuantizationType; upM = halfDim; upK = inputDim;

            // Fused bias split (if present)
            if (tensors.TryGetValue($"{prefix}.ffn_up.bias", out var fusedBiasDesc))
            {
                nint biasPtr = dataBase + (nint)fusedBiasDesc.DataOffset;
                gateBias = new float[halfDim];
                upBias = new float[halfDim];
                Dequantize.ToFloat32(biasPtr, halfDim, fusedBiasDesc.QuantizationType, gateBias);
                Dequantize.ToFloat32(biasPtr + halfDim * sizeof(float), halfDim, fusedBiasDesc.QuantizationType, upBias);
            }
            else
            {
                gateBias = null;
                upBias = null;
            }
        }

        return new TransformerLayerWeights(
            attnNorm,
            qPtr, qQt, qM, qK,
            kPtr, kQt, kM, kK,
            vPtr, vQt, vM, vK,
            oPtr, oQt, oM, oK,
            ffnNorm,
            gatePtr, gateQt, gateM, gateK,
            upPtr, upQt, upM, upK,
            downPtr, downQt, downM, downK,
            qBias, kBias, vBias, oBias,
            gateBias, upBias, downBias,
            qNormWeight, kNormWeight);
    }

    private static (nint ptr, QuantizationType qt, int outputDim, int inputDim) LoadLinear(
        nint dataBase, GgufTensorDescriptor desc)
    {
        nint ptr = dataBase + (nint)desc.DataOffset;
        // GGUF: Dimensions[0] = input dim (K), Dimensions[1] = output dim (M)
        int k = desc.Shape[0];
        int m = desc.Shape[1];
        return (ptr, desc.QuantizationType, m, k);
    }

    private static float[] DequantizeNorm(nint dataBase, GgufTensorDescriptor desc, int expectedSize)
    {
        nint ptr = dataBase + (nint)desc.DataOffset;
        float[] result = new float[expectedSize];
        Dequantize.ToFloat32(ptr, expectedSize, desc.QuantizationType, result);
        return result;
    }

    /// <summary>
    /// Loads an optional norm weight tensor. Returns null when the tensor is absent.
    /// </summary>
    private static float[]? LoadOptionalNorm(nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, string name, int expectedSize)
    {
        if (!tensors.TryGetValue(name, out var desc)) return null;
        return DequantizeNorm(dataBase, desc, expectedSize);
    }

    /// <summary>
    /// Loads an optional bias tensor (F32 in GGUF). Returns null when the tensor is absent.
    /// </summary>
    private static float[]? LoadOptionalBias(nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, string name)
    {
        if (!tensors.TryGetValue(name, out var desc)) return null;
        int size = (int)desc.Shape.ElementCount;
        float[] result = new float[size];
        Dequantize.ToFloat32(dataBase + (nint)desc.DataOffset, size, desc.QuantizationType, result);
        return result;
    }
}
