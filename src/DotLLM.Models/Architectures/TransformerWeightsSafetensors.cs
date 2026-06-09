using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Loads <see cref="TransformerWeights"/> from a HuggingFace-convention
/// safetensors source. Mirrors <see cref="TransformerWeights.LoadFromGguf"/>
/// but reads the HF tensor naming scheme
/// (<c>model.layers.{i}.self_attn.q_proj.weight</c>,
/// <c>model.embed_tokens.weight</c>, <c>lm_head.weight</c>, …).
/// </summary>
/// <remarks>
/// <para>
/// Stored F32 tensors are wired as zero-copy <c>nint</c> handles into the
/// mmap view. BF16 tensors are upcast into 64-byte-aligned
/// <see cref="NativeMemory.AlignedAlloc"/> scratch (a copy-at-load cost, but
/// the only way to feed existing F32 SIMD kernels). Any owned scratch
/// allocations are tracked by <see cref="TransformerWeights"/> and released
/// by its <see cref="TransformerWeights.Dispose"/>.
/// </para>
/// <para>
/// <b>tie_word_embeddings.</b> When the HF config declares tied embeddings
/// and <c>lm_head.weight</c> is physically absent from the safetensors file,
/// the LM-head pointer aliases <c>model.embed_tokens.weight</c>. The
/// resulting <c>TransformerWeights</c> treats that alias as a plain pointer
/// with no extra ownership — the mmap anchor keeps it alive.
/// </para>
/// <para>
/// The loader consumes an <see cref="ISafetensorsTensorSource"/> so that
/// both the single-file (<see cref="SafetensorsFile"/>) and future
/// multi-shard implementations can be plugged in without branching on the
/// concrete type. Tensor-pointer access goes through
/// <see cref="ISafetensorsTensorSource.GetTensorPointer(string)"/> — for
/// <see cref="SafetensorsFile"/> that is byte-identical to the previous
/// <c>DataBasePointer + DataBeginOffset</c> arithmetic.
/// </para>
/// </remarks>
internal static class TransformerWeightsSafetensorsLoader
{
    /// <summary>
    /// Resolves every transformer weight tensor from <paramref name="file"/>
    /// against the HF naming scheme for the architectures in
    /// <paramref name="config"/>. Throws on missing required tensors.
    /// </summary>
    public static TransformerWeights Load(ISafetensorsTensorSource file, ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        var owned = new List<nint>();
        try
        {
            // Token embedding
            var (embPtr, embQt, embM, embK) = ResolveLinear(file, "model.embed_tokens.weight", owned);
            if (embM != config.VocabSize || embK != config.HiddenSize)
                throw new InvalidDataException(
                    $"model.embed_tokens.weight shape [{embM},{embK}] does not match config [vocab={config.VocabSize}, hidden={config.HiddenSize}].");

            var layers = new TransformerLayerWeights[config.NumLayers];
            bool isDeepSeekMla = config.Architecture
                                   is DotLLM.Core.Configuration.Architecture.DeepSeekV2
                                   or DotLLM.Core.Configuration.Architecture.DeepSeekV3
                                 && config.MlaConfig is not null;
            for (int i = 0; i < config.NumLayers; i++)
            {
                layers[i] = isDeepSeekMla
                    ? LoadDeepSeekMlaLayer(i, file, config, owned)
                    : LoadLayer(i, file, config, owned);
            }

            // Final RMSNorm
            float[] outputNorm = ResolveNorm(file, "model.norm.weight", config.HiddenSize);

            // LM head — may be tied to embeddings
            nint outPtr;
            QuantizationType outQt;
            int outM, outK;
            if (file.TensorsByName.ContainsKey("lm_head.weight"))
            {
                (outPtr, outQt, outM, outK) = ResolveLinear(file, "lm_head.weight", owned);
            }
            else
            {
                // Tied: alias the embedding matrix. lm_head is logically [vocab, hidden]
                // and so is the embedding, so the shape/pointer line up directly.
                outPtr = embPtr;
                outQt = embQt;
                outM = embM;
                outK = embK;
            }
            if (outM != config.VocabSize || outK != config.HiddenSize)
                throw new InvalidDataException(
                    $"lm_head.weight shape [{outM},{outK}] does not match config [vocab={config.VocabSize}, hidden={config.HiddenSize}].");

            return TransformerWeights.CreateFromSafetensors(
                tokenEmbedWeight: embPtr, tokenEmbedQt: embQt,
                vocabSize: config.VocabSize, hiddenSize: config.HiddenSize,
                layers: layers,
                outputNormWeight: outputNorm,
                outputWeight: outPtr, outputQt: outQt, outputM: outM, outputK: outK,
                ownedAllocations: owned);
        }
        catch
        {
            // Roll back any allocations we made before rethrowing.
            foreach (var p in owned)
                unsafe { NativeMemory.AlignedFree((void*)p); }
            throw;
        }
    }

    private static TransformerLayerWeights LoadLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        string prefix = $"model.layers.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        int headDim = config.HeadDim;
        int qOut = config.NumAttentionHeads * headDim;
        int kvOut = config.NumKvHeads * headDim;

        // Input (pre-attention) RMSNorm
        float[] attnNorm = ResolveNorm(file, $"{prefix}.input_layernorm.weight", hiddenSize);

        // Q / K / V / O projections
        var (qPtr, qQt, qM, qK) = ResolveLinear(file, $"{prefix}.self_attn.q_proj.weight", owned);
        var (kPtr, kQt, kM, kK) = ResolveLinear(file, $"{prefix}.self_attn.k_proj.weight", owned);
        var (vPtr, vQt, vM, vK) = ResolveLinear(file, $"{prefix}.self_attn.v_proj.weight", owned);
        var (oPtr, oQt, oM, oK) = ResolveLinear(file, $"{prefix}.self_attn.o_proj.weight", owned);

        ValidateProjectionShape(qM, qK, qOut, hiddenSize, $"{prefix}.self_attn.q_proj.weight");
        ValidateProjectionShape(kM, kK, kvOut, hiddenSize, $"{prefix}.self_attn.k_proj.weight");
        ValidateProjectionShape(vM, vK, kvOut, hiddenSize, $"{prefix}.self_attn.v_proj.weight");
        ValidateProjectionShape(oM, oK, hiddenSize, qOut, $"{prefix}.self_attn.o_proj.weight");

        // Optional projection biases (Qwen2 has q/k/v biases; Llama does not)
        float[]? qBias = ResolveOptionalBias(file, $"{prefix}.self_attn.q_proj.bias", qOut);
        float[]? kBias = ResolveOptionalBias(file, $"{prefix}.self_attn.k_proj.bias", kvOut);
        float[]? vBias = ResolveOptionalBias(file, $"{prefix}.self_attn.v_proj.bias", kvOut);
        float[]? oBias = ResolveOptionalBias(file, $"{prefix}.self_attn.o_proj.bias", hiddenSize);

        // Optional QK-norms (Qwen3 per-head RMSNorm). Not emitted by vanilla HF
        // Llama/Mistral/Qwen2. Qwen3 names them {q_norm,k_norm}.weight.
        float[]? qNorm = ResolveOptionalNorm(file, $"{prefix}.self_attn.q_norm.weight", headDim);
        float[]? kNorm = ResolveOptionalNorm(file, $"{prefix}.self_attn.k_norm.weight", headDim);

        // Post-attention (pre-FFN) RMSNorm
        float[] ffnNorm = ResolveNorm(file, $"{prefix}.post_attention_layernorm.weight", hiddenSize);

        // FFN — dense (Llama/Mistral/Qwen), Mixtral-convention MoE, or
        // Qwen-MoE-convention MoE (possibly interleaved with dense layers via
        // decoder_sparse_step / mlp_only_layers).
        if (config.Moe is not null)
        {
            MoeLayerWeights? moe = null;
            bool useRoutedMoE = config.Architecture switch
            {
                // Mixtral: every layer is MoE.
                DotLLM.Core.Configuration.Architecture.Mixtral => true,
                // Qwen-MoE: per-layer decision based on decoder_sparse_step
                // and mlp_only_layers. A "dense" Qwen-MoE layer uses the
                // standard Llama-style mlp.{gate,up,down}_proj names — fall
                // through to the dense path below.
                DotLLM.Core.Configuration.Architecture.QwenMoe => config.Moe.IsMoeLayer(layerIdx),
                _ => true,
            };

            if (useRoutedMoE)
            {
                moe = config.Architecture switch
                {
                    DotLLM.Core.Configuration.Architecture.QwenMoe => LoadQwenMoeLayer(layerIdx, file, config, owned),
                    _ => LoadMixtralMoeLayer(layerIdx, file, config, owned),
                };
                return new TransformerLayerWeights(
                    attnNorm,
                    qPtr, qQt, qM, qK,
                    kPtr, kQt, kM, kK,
                    vPtr, vQt, vM, vK,
                    oPtr, oQt, oM, oK,
                    ffnNorm,
                    gateWeight: 0, gateQuantType: QuantizationType.F32, gateOutputDim: 0, gateInputDim: 0,
                    upWeight: 0, upQuantType: QuantizationType.F32, upOutputDim: 0, upInputDim: 0,
                    downWeight: 0, downQuantType: QuantizationType.F32, downOutputDim: 0, downInputDim: 0,
                    qBias, kBias, vBias, oBias,
                    gateBias: null, upBias: null, downBias: null,
                    qNormWeight: qNorm, kNormWeight: kNorm,
                    moe: moe);
            }
            // Otherwise: Qwen-MoE interleaved DENSE layer — fall through to
            // the Llama-style dense SwiGLU resolution below.
        }

        // Dense FFN — HF SwiGLU names: gate_proj, up_proj, down_proj.
        var (gatePtr, gateQt, gateM, gateK) = ResolveLinear(file, $"{prefix}.mlp.gate_proj.weight", owned);
        var (upPtr, upQt, upM, upK) = ResolveLinear(file, $"{prefix}.mlp.up_proj.weight", owned);
        var (downPtr, downQt, downM, downK) = ResolveLinear(file, $"{prefix}.mlp.down_proj.weight", owned);

        ValidateProjectionShape(gateM, gateK, config.IntermediateSize, hiddenSize, $"{prefix}.mlp.gate_proj.weight");
        ValidateProjectionShape(upM, upK, config.IntermediateSize, hiddenSize, $"{prefix}.mlp.up_proj.weight");
        ValidateProjectionShape(downM, downK, hiddenSize, config.IntermediateSize, $"{prefix}.mlp.down_proj.weight");

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
            gateBias: null, upBias: null, downBias: null,
            qNormWeight: qNorm, kNormWeight: kNorm);
    }

    /// <summary>
    /// Loads one transformer layer for a DeepSeek-V2 / DeepSeek-V3 checkpoint.
    /// Routes the attention projections through the MLA-specific tensor
    /// naming (<c>q_a_proj</c> / <c>q_b_proj</c> or monolithic <c>q_proj</c>,
    /// <c>kv_a_proj_with_mqa</c>, <c>kv_b_proj</c>, their layernorms, and
    /// <c>o_proj</c>). The FFN side currently loads a Llama-style dense
    /// SwiGLU — the DeepSeek MoE branch lands with the MoE foundation PR.
    /// All MLA tensors are coerced to F32 via
    /// <see cref="ResolveLinearAsF32"/>; the scalar MLA kernel consumes F32
    /// row-major throughout.
    /// </summary>
    private static TransformerLayerWeights LoadDeepSeekMlaLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        var mlaCfg = config.MlaConfig
                     ?? throw new InvalidOperationException(
                         "LoadDeepSeekMlaLayer called but ModelConfig.MlaConfig is null.");

        string prefix = $"model.layers.{layerIdx}";
        int hiddenSize = config.HiddenSize;
        int numHeads = config.NumAttentionHeads;
        int qkNope = mlaCfg.QkNopeHeadDim;
        int qkRope = mlaCfg.QkRopeHeadDim;
        int qkHead = qkNope + qkRope;
        int vHead = mlaCfg.VHeadDim;
        int qLoraRank = mlaCfg.QLoraRank;
        int kvLoraRank = mlaCfg.KvLoraRank;
        int qTotalOut = numHeads * qkHead;
        int kvBOut = numHeads * (qkNope + vHead);
        int oInputDim = numHeads * vHead;

        // Pre-attention RMSNorm (standard Llama-style input_layernorm).
        float[] attnNorm = ResolveNorm(file, $"{prefix}.input_layernorm.weight", hiddenSize);

        // Q path: LoRA-factored (V2 full, V3) or monolithic (V2-Lite). The
        // kernel decides which path to take based on qLoraRank; we pass zero
        // pointers for the unused set.
        nint qAProj = 0, qBProj = 0, qProj = 0;
        float[]? qALayernorm = null;
        if (qLoraRank > 0)
        {
            (qAProj, _, int qAm, int qAk) = ResolveLinearAsF32(
                file, $"{prefix}.self_attn.q_a_proj.weight", owned);
            ValidateProjectionShape(qAm, qAk, qLoraRank, hiddenSize,
                $"{prefix}.self_attn.q_a_proj.weight");
            qALayernorm = ResolveNorm(file, $"{prefix}.self_attn.q_a_layernorm.weight", qLoraRank);
            (qBProj, _, int qBm, int qBk) = ResolveLinearAsF32(
                file, $"{prefix}.self_attn.q_b_proj.weight", owned);
            ValidateProjectionShape(qBm, qBk, qTotalOut, qLoraRank,
                $"{prefix}.self_attn.q_b_proj.weight");
        }
        else
        {
            (qProj, _, int qM, int qK) = ResolveLinearAsF32(
                file, $"{prefix}.self_attn.q_proj.weight", owned);
            ValidateProjectionShape(qM, qK, qTotalOut, hiddenSize,
                $"{prefix}.self_attn.q_proj.weight");
        }

        // KV path: always LoRA-factored. kv_a_proj_with_mqa emits
        // [kvLoraRank + qkRopeHeadDim] per token — the first kvLoraRank rows
        // feed kv_a_layernorm then kv_b_proj, the last qkRopeHeadDim rows are
        // the MQA-shared rope-K. No separate LayerNorm on the rope-K side.
        int kvADim = kvLoraRank + qkRope;
        (nint kvAProj, _, int kvaM, int kvaK) = ResolveLinearAsF32(
            file, $"{prefix}.self_attn.kv_a_proj_with_mqa.weight", owned);
        ValidateProjectionShape(kvaM, kvaK, kvADim, hiddenSize,
            $"{prefix}.self_attn.kv_a_proj_with_mqa.weight");
        float[] kvALayernorm = ResolveNorm(
            file, $"{prefix}.self_attn.kv_a_layernorm.weight", kvLoraRank);
        (nint kvBProj, _, int kvbM, int kvbK) = ResolveLinearAsF32(
            file, $"{prefix}.self_attn.kv_b_proj.weight", owned);
        ValidateProjectionShape(kvbM, kvbK, kvBOut, kvLoraRank,
            $"{prefix}.self_attn.kv_b_proj.weight");

        // Output projection: hidden ← n_heads * v_head_dim. Kept in the
        // existing O slot (not MLA-specific) because the forward path still
        // applies bias (if any) through the same AddBias logic.
        var (oPtr, oQt, oM, oK) = ResolveLinearAsF32(
            file, $"{prefix}.self_attn.o_proj.weight", owned);
        ValidateProjectionShape(oM, oK, hiddenSize, oInputDim,
            $"{prefix}.self_attn.o_proj.weight");
        float[]? oBias = ResolveOptionalBias(file, $"{prefix}.self_attn.o_proj.bias", hiddenSize);

        var mla = new MlaLayerWeights(
            qAProj: qAProj, qALayernormWeight: qALayernorm, qBProj: qBProj, qProj: qProj,
            kvAProjWithMqa: kvAProj, kvALayernormWeight: kvALayernorm, kvBProj: kvBProj,
            numHeads: numHeads,
            qkNopeHeadDim: qkNope, qkRopeHeadDim: qkRope, vHeadDim: vHead,
            qLoraRank: qLoraRank, kvLoraRank: kvLoraRank);

        // Post-attention RMSNorm (shared with Llama convention).
        float[] ffnNorm = ResolveNorm(file, $"{prefix}.post_attention_layernorm.weight", hiddenSize);

        // Dense FFN (Llama SwiGLU convention). DeepSeek-V2/V3 interleaves
        // dense MLP (first_k_dense_replace layers) with MoE (rest) — only
        // the dense path is wired in this foundation PR. The MoE FFN branch
        // and its layer-level routing land with the MoE foundation PR.
        var (gatePtr, gateQt, gateM, gateK) = ResolveLinear(
            file, $"{prefix}.mlp.gate_proj.weight", owned);
        var (upPtr, upQt, upM, upK) = ResolveLinear(
            file, $"{prefix}.mlp.up_proj.weight", owned);
        var (downPtr, downQt, downM, downK) = ResolveLinear(
            file, $"{prefix}.mlp.down_proj.weight", owned);
        ValidateProjectionShape(gateM, gateK, config.IntermediateSize, hiddenSize,
            $"{prefix}.mlp.gate_proj.weight");
        ValidateProjectionShape(upM, upK, config.IntermediateSize, hiddenSize,
            $"{prefix}.mlp.up_proj.weight");
        ValidateProjectionShape(downM, downK, hiddenSize, config.IntermediateSize,
            $"{prefix}.mlp.down_proj.weight");

        return new TransformerLayerWeights(
            attnNorm,
            qWeight: 0, qQuantType: QuantizationType.F32, qOutputDim: 0, qInputDim: 0,
            kWeight: 0, kQuantType: QuantizationType.F32, kOutputDim: 0, kInputDim: 0,
            vWeight: 0, vQuantType: QuantizationType.F32, vOutputDim: 0, vInputDim: 0,
            oPtr, oQt, oM, oK,
            ffnNorm,
            gatePtr, gateQt, gateM, gateK,
            upPtr, upQt, upM, upK,
            downPtr, downQt, downM, downK,
            qBias: null, kBias: null, vBias: null, oBias: oBias,
            gateBias: null, upBias: null, downBias: null,
            qNormWeight: null, kNormWeight: null,
            mla: mla);
    }

    /// <summary>
    /// Resolves a rank-2 projection weight as an F32 pointer. F32 tensors are
    /// returned zero-copy; F16 and BF16 tensors are upcast into 64-byte-aligned
    /// owned scratch and registered in <paramref name="owned"/>. Similar to
    /// <see cref="ResolveLinear"/> but always hands back F32 — the scalar MLA
    /// kernel expects F32 throughout (quantised MLA loaders land in a
    /// follow-up).
    /// </summary>
    private static unsafe (nint ptr, QuantizationType qt, int m, int k) ResolveLinearAsF32(
        ISafetensorsTensorSource file, string name, List<nint> owned)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc))
            throw new InvalidDataException($"Safetensors file is missing required tensor '{name}'.");
        if (desc.Shape.Length != 2)
            throw new InvalidDataException($"Tensor '{name}' expected to be rank-2, got rank {desc.Shape.Length}.");

        int m = desc.Shape[0], k = desc.Shape[1];
        long count = (long)m * k;
        nint srcPtr = file.GetTensorPointer(name);

        switch (desc.DType)
        {
            case SafetensorsDType.F32:
                return (srcPtr, QuantizationType.F32, m, k);

            case SafetensorsDType.BF16:
            {
                nint dst = AllocBf16ToF32(srcPtr, count);
                owned.Add(dst);
                return (dst, QuantizationType.F32, m, k);
            }

            case SafetensorsDType.F16:
            {
                nuint byteCount = checked((nuint)count * sizeof(float));
                nint dst = (nint)NativeMemory.AlignedAlloc(byteCount, 64);
                owned.Add(dst);
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>((void*)srcPtr, (int)count),
                    new Span<float>((void*)dst, (int)count));
                return (dst, QuantizationType.F32, m, k);
            }

            default:
                throw new NotSupportedException(
                    $"Tensor '{name}' has dtype {desc.DType} — MLA loader supports F32/F16/BF16 only.");
        }
    }

    /// <summary>
    /// Loads Qwen-MoE-convention MoE weights for one transformer layer:
    /// <c>model.layers.{i}.mlp.gate.weight</c> and
    /// <c>model.layers.{i}.mlp.experts.{j}.{gate_proj,up_proj,down_proj}.weight</c>
    /// — math-identical to Mixtral but with HF Llama-style tensor names.
    /// When <see cref="MoeConfig.SharedExpertIntermediateSize"/> is set the
    /// parallel shared-expert branch (<c>mlp.shared_expert.*</c>) and
    /// optionally the <c>mlp.shared_expert_gate.weight</c> sigmoid gate are
    /// resolved too. Everything lands in F32 via
    /// <see cref="ResolveLinearAsF32"/> so the kernel is uniform in dtype.
    /// </summary>
    private static MoeLayerWeights LoadQwenMoeLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        var moe = config.Moe
                  ?? throw new InvalidOperationException("LoadQwenMoeLayer called with null Moe config.");

        string prefix = $"model.layers.{layerIdx}.mlp";
        int hiddenSize = config.HiddenSize;
        int intermediateSize = moe.MoeIntermediateSize;
        int numExperts = moe.NumExperts;

        // Router gate — F32 [E, H].
        float[] gate = ResolveDense2D(file, $"{prefix}.gate.weight", numExperts, hiddenSize);

        var w1 = new nint[numExperts];
        var w2 = new nint[numExperts];
        var w3 = new nint[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            // w1 ≡ gate_proj: [intermediate, hidden]
            (w1[e], _, int w1M, int w1K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.gate_proj.weight", owned);
            ValidateProjectionShape(w1M, w1K, intermediateSize, hiddenSize,
                $"{prefix}.experts.{e}.gate_proj.weight");
            // w3 ≡ up_proj: [intermediate, hidden]
            (w3[e], _, int w3M, int w3K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.up_proj.weight", owned);
            ValidateProjectionShape(w3M, w3K, intermediateSize, hiddenSize,
                $"{prefix}.experts.{e}.up_proj.weight");
            // w2 ≡ down_proj: [hidden, intermediate]
            (w2[e], _, int w2M, int w2K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.down_proj.weight", owned);
            ValidateProjectionShape(w2M, w2K, hiddenSize, intermediateSize,
                $"{prefix}.experts.{e}.down_proj.weight");
        }

        // Shared expert(s). Two naming conventions:
        //   - Qwen1.5-MoE-A2.7B: singular mlp.shared_expert.{gate,up,down}_proj
        //     (always exactly one shared expert; optionally gated by
        //     mlp.shared_expert_gate.weight).
        //   - DeepSeek-V2/V3: plural mlp.shared_experts.{k}.{gate,up,down}_proj
        //     (n_shared_experts >= 1, summed, no gate).
        // We resolve whichever set of tensors the file actually contains; the
        // kernel sees a uniform pointer-array API. If the config flags a shared
        // expert but the tensors are absent, we silently fall back to routed-only.
        nint[] sharedGate = Array.Empty<nint>();
        nint[] sharedUp = Array.Empty<nint>();
        nint[] sharedDown = Array.Empty<nint>();
        int sharedIntermediate = 0;
        float[]? sharedExpertGate = null;
        if (moe.SharedExpertIntermediateSize is int sharedI)
        {
            int numShared = moe.NumSharedExperts;
            // Detect the tensor-name convention. Prefer plural (DeepSeek) when
            // present — this is the forward-compatible format. Fall back to
            // singular (Qwen1.5-MoE) when only that exists.
            bool hasPlural = numShared >= 1
                && file.TensorsByName.ContainsKey($"{prefix}.shared_experts.0.gate_proj.weight");
            bool hasSingular = numShared == 1
                && file.TensorsByName.ContainsKey($"{prefix}.shared_expert.gate_proj.weight");

            if (hasPlural)
            {
                sharedIntermediate = sharedI;
                sharedGate = new nint[numShared];
                sharedUp = new nint[numShared];
                sharedDown = new nint[numShared];
                for (int k = 0; k < numShared; k++)
                {
                    (sharedGate[k], _, int sgM, int sgK) = ResolveLinearAsF32(file,
                        $"{prefix}.shared_experts.{k}.gate_proj.weight", owned);
                    ValidateProjectionShape(sgM, sgK, sharedI, hiddenSize,
                        $"{prefix}.shared_experts.{k}.gate_proj.weight");
                    (sharedUp[k], _, int suM, int suK) = ResolveLinearAsF32(file,
                        $"{prefix}.shared_experts.{k}.up_proj.weight", owned);
                    ValidateProjectionShape(suM, suK, sharedI, hiddenSize,
                        $"{prefix}.shared_experts.{k}.up_proj.weight");
                    (sharedDown[k], _, int sdM, int sdK) = ResolveLinearAsF32(file,
                        $"{prefix}.shared_experts.{k}.down_proj.weight", owned);
                    ValidateProjectionShape(sdM, sdK, hiddenSize, sharedI,
                        $"{prefix}.shared_experts.{k}.down_proj.weight");
                }
            }
            else if (hasSingular)
            {
                sharedIntermediate = sharedI;
                sharedGate = new nint[1];
                sharedUp = new nint[1];
                sharedDown = new nint[1];
                (sharedGate[0], _, int sgM, int sgK) = ResolveLinearAsF32(file,
                    $"{prefix}.shared_expert.gate_proj.weight", owned);
                ValidateProjectionShape(sgM, sgK, sharedI, hiddenSize,
                    $"{prefix}.shared_expert.gate_proj.weight");
                (sharedUp[0], _, int suM, int suK) = ResolveLinearAsF32(file,
                    $"{prefix}.shared_expert.up_proj.weight", owned);
                ValidateProjectionShape(suM, suK, sharedI, hiddenSize,
                    $"{prefix}.shared_expert.up_proj.weight");
                (sharedDown[0], _, int sdM, int sdK) = ResolveLinearAsF32(file,
                    $"{prefix}.shared_expert.down_proj.weight", owned);
                ValidateProjectionShape(sdM, sdK, hiddenSize, sharedI,
                    $"{prefix}.shared_expert.down_proj.weight");

                // Optional sigmoid gate — HF stores it as [1, hiddenSize] (a plain
                // Linear(hidden -> 1, bias=False)). ElementCount == hiddenSize, so
                // ResolveNorm slots in cleanly.
                string gateName = $"{prefix}.shared_expert_gate.weight";
                if (moe.HasSharedExpertGate && file.TensorsByName.ContainsKey(gateName))
                {
                    sharedExpertGate = ResolveNorm(file, gateName, hiddenSize);
                }
            }
            // else: config declared a shared branch but the file has neither
            // plural nor singular tensors — silently fall back to routed-only
            // (sharedIntermediate stays 0, arrays stay empty).
        }

        return new MoeLayerWeights(
            gate: gate,
            w1: w1, w2: w2, w3: w3,
            numExperts: numExperts,
            numExpertsPerTok: moe.NumExpertsPerTok,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            normTopKProb: moe.NormTopKProb,
            sharedGateProj: sharedGate,
            sharedUpProj: sharedUp,
            sharedDownProj: sharedDown,
            sharedIntermediateSize: sharedIntermediate,
            sharedExpertGate: sharedExpertGate);
    }

    /// <summary>
    /// Loads Mixtral-convention MoE weights for one transformer layer:
    /// <c>model.layers.{i}.block_sparse_moe.gate.weight</c> and
    /// <c>model.layers.{i}.block_sparse_moe.experts.{j}.(w1|w2|w3).weight</c>.
    /// Router gate is resolved into a managed <c>float[]</c> (tiny —
    /// numExperts × hiddenSize). Per-expert weights are F32 pointers; bf16/
    /// F16 tensors are upcast at load time into 64-byte-aligned scratch and
    /// registered in <paramref name="owned"/>.
    /// </summary>
    private static MoeLayerWeights LoadMixtralMoeLayer(
        int layerIdx, ISafetensorsTensorSource file, ModelConfig config, List<nint> owned)
    {
        var moe = config.Moe
                  ?? throw new InvalidOperationException("LoadMixtralMoeLayer called with null Moe config.");

        string prefix = $"model.layers.{layerIdx}.block_sparse_moe";
        int hiddenSize = config.HiddenSize;
        int intermediateSize = moe.MoeIntermediateSize;
        int numExperts = moe.NumExperts;

        // Router gate — F32 [E, H].
        float[] gate = ResolveDense2D(file, $"{prefix}.gate.weight", numExperts, hiddenSize);

        var w1 = new nint[numExperts];
        var w2 = new nint[numExperts];
        var w3 = new nint[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            // w1 (gate_proj): [intermediate, hidden]
            (w1[e], _, int w1M, int w1K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.w1.weight", owned);
            ValidateProjectionShape(w1M, w1K, intermediateSize, hiddenSize,
                $"{prefix}.experts.{e}.w1.weight");
            // w3 (up_proj): [intermediate, hidden]
            (w3[e], _, int w3M, int w3K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.w3.weight", owned);
            ValidateProjectionShape(w3M, w3K, intermediateSize, hiddenSize,
                $"{prefix}.experts.{e}.w3.weight");
            // w2 (down_proj): [hidden, intermediate]
            (w2[e], _, int w2M, int w2K) = ResolveLinearAsF32(file, $"{prefix}.experts.{e}.w2.weight", owned);
            ValidateProjectionShape(w2M, w2K, hiddenSize, intermediateSize,
                $"{prefix}.experts.{e}.w2.weight");
        }

        return new MoeLayerWeights(
            gate: gate,
            w1: w1, w2: w2, w3: w3,
            numExperts: numExperts,
            numExpertsPerTok: moe.NumExpertsPerTok,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize);
    }

    /// <summary>
    /// Resolves a rank-2 tensor as a managed <c>float[]</c>, up-casting F16 /
    /// BF16 on the way in. Used for small weights (router gate) where a copy
    /// costs nothing and is simpler than tracking owned allocations.
    /// </summary>
    private static unsafe float[] ResolveDense2D(
        ISafetensorsTensorSource file, string name, int expectedM, int expectedK)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc))
            throw new InvalidDataException($"Safetensors file is missing required tensor '{name}'.");
        if (desc.Shape.Length != 2)
            throw new InvalidDataException($"Tensor '{name}' expected to be rank-2, got rank {desc.Shape.Length}.");
        int m = desc.Shape[0], k = desc.Shape[1];
        if (m != expectedM || k != expectedK)
            throw new InvalidDataException(
                $"Tensor '{name}' shape [{m},{k}] does not match expected [{expectedM},{expectedK}].");

        int count = m * k;
        var result = new float[count];
        nint src = file.GetTensorPointer(name);
        DecodeFloatTensor(src, desc.DType, count, result, name);
        return result;
    }

    private static void ValidateProjectionShape(int actualM, int actualK, int expectedM, int expectedK, string name)
    {
        if (actualM != expectedM || actualK != expectedK)
            throw new InvalidDataException(
                $"{name} shape [M={actualM}, K={actualK}] does not match expected [M={expectedM}, K={expectedK}].");
    }

    /// <summary>
    /// Resolves a safetensors tensor as a linear projection weight:
    /// HF shape <c>[out_features, in_features]</c> → (ptr, dtype, M, K).
    /// F32 tensors are zero-copy; BF16 tensors are upcast into an owned
    /// 64-byte-aligned scratch buffer and registered in
    /// <paramref name="owned"/>.
    /// </summary>
    private static unsafe (nint ptr, QuantizationType qt, int m, int k) ResolveLinear(
        ISafetensorsTensorSource file, string name, List<nint> owned)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc))
            throw new InvalidDataException(
                $"Safetensors file is missing required tensor '{name}'.");

        if (desc.Shape.Length != 2)
            throw new InvalidDataException(
                $"Tensor '{name}' expected to be rank-2, got rank {desc.Shape.Length}.");

        int m = desc.Shape[0];
        int k = desc.Shape[1];

        nint srcPtr = file.GetTensorPointer(name);

        switch (desc.DType)
        {
            case SafetensorsDType.F32:
                return (srcPtr, QuantizationType.F32, m, k);

            case SafetensorsDType.BF16:
            {
                long elementCount = (long)m * k;
                nint dst = AllocBf16ToF32(srcPtr, elementCount);
                owned.Add(dst);
                return (dst, QuantizationType.F32, m, k);
            }

            case SafetensorsDType.F16:
            {
                // Keep as F16 (kernels support it directly). No copy.
                return (srcPtr, QuantizationType.F16, m, k);
            }

            default:
                throw new NotSupportedException(
                    $"Tensor '{name}' has dtype {desc.DType} which is not yet supported by the safetensors transformer loader (F32/F16/BF16 only).");
        }
    }

    /// <summary>
    /// Resolves a norm weight tensor into a managed <c>float[]</c>. Norms
    /// are small and read once per forward call, so the load-time copy has
    /// no measurable inference cost.
    /// </summary>
    private static float[] ResolveNorm(ISafetensorsTensorSource file, string name, int expectedSize)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc))
            throw new InvalidDataException(
                $"Safetensors file is missing required tensor '{name}'.");

        long elementCount = desc.ElementCount;
        if (elementCount != expectedSize)
            throw new InvalidDataException(
                $"Tensor '{name}' has {elementCount} elements, expected {expectedSize}.");

        var result = new float[expectedSize];
        nint src = file.GetTensorPointer(name);
        DecodeFloatTensor(src, desc.DType, expectedSize, result, name);
        return result;
    }

    private static float[]? ResolveOptionalNorm(ISafetensorsTensorSource file, string name, int expectedSize)
    {
        if (!file.TensorsByName.ContainsKey(name)) return null;
        return ResolveNorm(file, name, expectedSize);
    }

    private static float[]? ResolveOptionalBias(ISafetensorsTensorSource file, string name, int expectedSize)
    {
        if (!file.TensorsByName.TryGetValue(name, out var desc)) return null;

        long elementCount = desc.ElementCount;
        if (elementCount != expectedSize)
            throw new InvalidDataException(
                $"Bias tensor '{name}' has {elementCount} elements, expected {expectedSize}.");

        var result = new float[expectedSize];
        nint src = file.GetTensorPointer(name);
        DecodeFloatTensor(src, desc.DType, expectedSize, result, name);
        return result;
    }

    private static unsafe void DecodeFloatTensor(
        nint src, SafetensorsDType dtype, int elementCount, float[] dest, string name)
    {
        switch (dtype)
        {
            case SafetensorsDType.F32:
                new ReadOnlySpan<float>((void*)src, elementCount).CopyTo(dest);
                break;
            case SafetensorsDType.F16:
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>((void*)src, elementCount), dest);
                break;
            case SafetensorsDType.BF16:
                DecodeBf16((ushort*)src, elementCount, dest);
                break;
            default:
                throw new NotSupportedException(
                    $"Tensor '{name}' has dtype {dtype} which is not supported for norm/bias load (F32/F16/BF16 only).");
        }
    }

    /// <summary>
    /// Upcasts a bf16 tensor to a 64-byte-aligned F32 buffer owned by the
    /// caller. bf16 is "the high 16 bits of an IEEE-754 binary32", so the
    /// upcast is a shift-left-by-16-bits reinterpret — identical to what
    /// llama.cpp does when it normalises HF checkpoints to F32.
    /// </summary>
    private static unsafe nint AllocBf16ToF32(nint srcBf16, long elementCount)
    {
        nuint byteCount = checked((nuint)elementCount * sizeof(float));
        nint dst = (nint)NativeMemory.AlignedAlloc(byteCount, 64);
        DecodeBf16((ushort*)srcBf16, (int)elementCount, new Span<float>((void*)dst, (int)elementCount));
        return dst;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void DecodeBf16(ushort* src, int count, Span<float> dest)
    {
        // bf16 → f32: shift the 16 bits into the high half of a 32-bit word,
        // then reinterpret as float. NaN/Inf bit patterns transfer cleanly.
        fixed (float* dstPtr = dest)
        {
            uint* dw = (uint*)dstPtr;
            for (int i = 0; i < count; i++)
                dw[i] = (uint)src[i] << 16;
        }
    }
}
