using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Parses a HuggingFace <c>config.json</c> for a dense-transformer checkpoint
/// (Llama, Mistral, Phi, Qwen) into a populated <see cref="ModelConfig"/>.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors <see cref="DotLLM.Models.Gguf.GgufModelConfigExtractor"/> but reads
/// JSON rather than GGUF metadata KV pairs. The source of truth for the field
/// names is the <c>transformers</c> per-architecture <c>configuration_*.py</c>
/// file — e.g. <c>LlamaConfig</c> (<c>hidden_size</c>, <c>num_hidden_layers</c>,
/// <c>num_attention_heads</c>, <c>num_key_value_heads</c>, <c>intermediate_size</c>,
/// <c>vocab_size</c>, <c>max_position_embeddings</c>, <c>rope_theta</c>,
/// <c>rms_norm_eps</c>, <c>tie_word_embeddings</c>, <c>architectures[0]</c>).
/// </para>
/// <para>
/// Defensive about common HF quirks: <c>num_key_value_heads</c> may be absent
/// (implies MHA: equal to <c>num_attention_heads</c>), <c>head_dim</c> may be
/// stored explicitly (Qwen3/some Llamas) or implied by <c>hidden_size /
/// num_attention_heads</c>, and the top-level <c>architectures</c> array
/// carries the class name (e.g. <c>LlamaForCausalLM</c>) which disambiguates
/// Llama vs Mistral vs Phi3 vs Qwen2 when <c>model_type</c> alone is ambiguous.
/// </para>
/// </remarks>
public static class HfConfigExtractor
{
    /// <summary>
    /// Parses a HF <c>config.json</c> payload (raw string) into a
    /// <see cref="ModelConfig"/>.
    /// </summary>
    public static ModelConfig Extract(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using var doc = JsonDocument.Parse(json);
        return Extract(doc.RootElement);
    }

    /// <summary>
    /// Parses a HF <c>config.json</c> already deserialised into a
    /// <see cref="JsonElement"/> into a <see cref="ModelConfig"/>.
    /// </summary>
    /// <exception cref="InvalidDataException">
    /// Required fields missing / illegal values / unsupported architecture.
    /// </exception>
    public static ModelConfig Extract(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("HF config.json root must be a JSON object.");

        Architecture architecture = ResolveArchitecture(root);

        int hiddenSize = GetInt32(root, "hidden_size");
        int numLayers = GetInt32(root, "num_hidden_layers");
        int numAttentionHeads = GetInt32(root, "num_attention_heads");
        int numKvHeads = GetInt32OrDefault(root, "num_key_value_heads", numAttentionHeads);
        int intermediateSize = GetInt32(root, "intermediate_size");
        int vocabSize = GetInt32(root, "vocab_size");
        int maxSeqLen = GetInt32OrDefault(root, "max_position_embeddings", 2048);

        bool isMla = architecture is Architecture.DeepSeekV2 or Architecture.DeepSeekV3;

        // MLA surfaces head_dim via a non-standard split: the Q/K "head_dim"
        // is qk_nope_head_dim + qk_rope_head_dim, while V has its own
        // v_head_dim. The ModelConfig.HeadDim field is reused to carry
        // qk_head_dim so downstream KV-cache / shape logic sees a single
        // number per head; attention callers gate on MlaConfig != null for
        // the MLA-specific per-head splits.
        int headDim;
        MlaConfig? mla;
        if (isMla)
        {
            mla = ExtractMlaConfig(root);
            headDim = mla!.QkHeadDim;
        }
        else
        {
            mla = null;
            headDim = GetInt32OrDefault(root, "head_dim", hiddenSize / numAttentionHeads);
        }

        float normEps = GetFloatOrDefault(root, "rms_norm_eps",
            GetFloatOrDefault(root, "layer_norm_eps", 1e-5f));
        float ropeTheta = GetFloatOrDefault(root, "rope_theta", 10000.0f);
        bool tieEmbeddings = GetBoolOrDefault(root, "tie_word_embeddings", DefaultTieForArch(architecture));

        int? slidingWindow = GetInt32NullableIfPositive(root, "sliding_window");

        // RoPE element-pairing convention — identical to GgufModelConfigExtractor.
        // Llama/Mistral/Mixtral/DeepSeek-V2 use interleaved (Norm); Qwen/Qwen-MoE/Phi use non-interleaved (NeoX).
        RoPEType ropeType = architecture switch
        {
            Architecture.Qwen or Architecture.QwenMoe or Architecture.Phi => RoPEType.NeoX,
            _ => RoPEType.Norm,
        };

        // MoE — Mixtral, Qwen*-MoE, Phi-3.5-MoE all expose num_local_experts +
        // num_experts_per_tok. Shared experts (DeepSeek-V3, old Qwen1.5-MoE)
        // add more fields and are explicitly out of scope here.
        MoeConfig? moe = ExtractMoeConfig(root, intermediateSize);

        var ropeConfig = new RoPEConfig(
            Theta: ropeTheta,
            DimensionCount: headDim,
            Type: ropeType);

        return new ModelConfig
        {
            Architecture = architecture,
            VocabSize = vocabSize,
            HiddenSize = hiddenSize,
            IntermediateSize = intermediateSize,
            NumLayers = numLayers,
            NumAttentionHeads = numAttentionHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = maxSeqLen,
            AttentionType = isMla ? AttentionType.MLA : AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = ropeConfig,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = normEps,
            TiedEmbeddings = tieEmbeddings,
            SlidingWindowSize = slidingWindow,
            MlaConfig = mla,
            Moe = moe,
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Extracts <see cref="MlaConfig"/> from a DeepSeek-V2/V3 HF config.json.
    /// Required fields: <c>kv_lora_rank</c>, <c>qk_nope_head_dim</c>,
    /// <c>qk_rope_head_dim</c>, <c>v_head_dim</c>. <c>q_lora_rank</c> is
    /// optional (0 / null means a monolithic <c>q_proj</c> is used instead).
    /// YaRN rope scaling fields are captured but not yet consumed by the
    /// attention kernel — see <see cref="MlaConfig.RopeScalingFactor"/>.
    /// </summary>
    private static MlaConfig ExtractMlaConfig(JsonElement root)
    {
        int kvLoraRank = GetInt32(root, "kv_lora_rank");
        int qkNope = GetInt32(root, "qk_nope_head_dim");
        int qkRope = GetInt32(root, "qk_rope_head_dim");
        int vHead = GetInt32(root, "v_head_dim");

        // q_lora_rank may be absent (V3 variants skip Q factorisation) or null.
        int qLora = 0;
        if (root.TryGetProperty("q_lora_rank", out var qLoraProp)
            && qLoraProp.ValueKind == JsonValueKind.Number
            && qLoraProp.TryGetInt32(out int qLoraVal)
            && qLoraVal > 0)
        {
            qLora = qLoraVal;
        }

        float ropeTheta = GetFloatOrDefault(root, "rope_theta", 10000.0f);

        // Optional rope_scaling (YaRN) — surface but do not yet apply.
        float? scalingFactor = null;
        float? scalingMscale = null;
        float? scalingMscaleAllDim = null;
        int? scalingOriginalMax = null;
        if (root.TryGetProperty("rope_scaling", out var rs) && rs.ValueKind == JsonValueKind.Object)
        {
            if (rs.TryGetProperty("factor", out var f)
                && f.ValueKind == JsonValueKind.Number
                && f.TryGetSingle(out float fv))
                scalingFactor = fv;
            if (rs.TryGetProperty("mscale", out var m)
                && m.ValueKind == JsonValueKind.Number
                && m.TryGetSingle(out float mv))
                scalingMscale = mv;
            if (rs.TryGetProperty("mscale_all_dim", out var mad)
                && mad.ValueKind == JsonValueKind.Number
                && mad.TryGetSingle(out float madv))
                scalingMscaleAllDim = madv;
            if (rs.TryGetProperty("original_max_position_embeddings", out var om)
                && om.ValueKind == JsonValueKind.Number
                && om.TryGetInt32(out int omv))
                scalingOriginalMax = omv;
        }

        return new MlaConfig
        {
            KvLoraRank = kvLoraRank,
            QLoraRank = qLora,
            QkNopeHeadDim = qkNope,
            QkRopeHeadDim = qkRope,
            VHeadDim = vHead,
            RopeTheta = ropeTheta,
            RopeScalingFactor = scalingFactor,
            RopeScalingMscale = scalingMscale,
            RopeScalingMscaleAllDim = scalingMscaleAllDim,
            RopeScalingOriginalMaxPositionEmbeddings = scalingOriginalMax,
        };
    }

    /// <summary>
    /// Detects MoE from a HF <c>config.json</c> and returns a
    /// <see cref="MoeConfig"/> when present, else null. Recognises:
    /// <list type="bullet">
    ///   <item><c>num_local_experts</c> (Mixtral) or <c>num_experts</c> (Qwen-MoE, DBRX) &gt; 0</item>
    ///   <item><c>num_experts_per_tok</c> (top-k)</item>
    ///   <item><c>moe_intermediate_size</c> override (Phi-3.5-MoE, Qwen-MoE per-expert width);
    ///     falls back to <paramref name="defaultIntermediateSize"/></item>
    ///   <item><c>norm_topk_prob</c> (Qwen-MoE top-k renormalisation flag; defaults to true — Mixtral behaviour)</item>
    ///   <item><c>shared_expert_intermediate_size</c> (Qwen1.5-MoE shared-expert width); absent → no shared expert</item>
    ///   <item><c>decoder_sparse_step</c> and <c>mlp_only_layers</c> (Qwen3-MoE layer-level sparsity)</item>
    /// </list>
    /// Returns null if neither expert-count key is present — the model is
    /// treated as dense.
    /// </summary>
    private static MoeConfig? ExtractMoeConfig(JsonElement root, int defaultIntermediateSize)
    {
        int numExperts = GetInt32OrDefault(root, "num_local_experts", 0);
        if (numExperts <= 0)
            numExperts = GetInt32OrDefault(root, "num_experts", 0);
        if (numExperts <= 0)
            numExperts = GetInt32OrDefault(root, "n_routed_experts", 0); // DeepSeek convention
        if (numExperts <= 0)
            return null;

        int numExpertsPerTok = GetInt32OrDefault(root, "num_experts_per_tok", 0);
        if (numExpertsPerTok <= 0)
            throw new InvalidDataException(
                $"HF config.json declares {numExperts} MoE experts but is missing or has invalid 'num_experts_per_tok'.");
        if (numExpertsPerTok > numExperts)
            throw new InvalidDataException(
                $"HF config.json has num_experts_per_tok={numExpertsPerTok} > num_experts={numExperts}.");

        // Phi-3.5-MoE + Qwen-MoE + DeepSeek-V2/V3 expose moe_intermediate_size.
        // Mixtral reuses intermediate_size for the expert width.
        int moeIntermediateSize = GetInt32OrDefault(root, "moe_intermediate_size", defaultIntermediateSize);

        // Qwen-MoE / DeepSeek: norm_topk_prob governs whether top-k probs are
        // renormalised to sum to 1. Mixtral always does this so its config
        // never ships the key — default to true to preserve Mixtral behaviour.
        bool normTopKProb = GetBoolOrDefault(root, "norm_topk_prob", true);

        // Shared-expert intermediate width and count.
        //   Qwen1.5-MoE-A2.7B: ships `shared_expert_intermediate_size` directly
        //     with a single shared expert (singular `mlp.shared_expert.*`),
        //     optionally sigmoid-gated by `mlp.shared_expert_gate.weight`.
        //   DeepSeek-V2/V3: ships `moe_intermediate_size` per shared expert
        //     with `n_shared_experts` plural shared experts (tensor naming
        //     `mlp.shared_experts.{k}.*`). Each shared expert is
        //     moe_intermediate_size wide; outputs are summed (equally
        //     weighted, no sigmoid gate). The MoE kernel iterates over
        //     individual experts and sums their dense SwiGLU outputs into
        //     the routed sum.
        //
        // DeepSeek is detected by the presence of `n_shared_experts` (which
        // neither Qwen nor any other MoE family ships). Architecture enum
        // dispatch (Architecture.DeepSeekV2 / V3) lands separately with the
        // MLA chain; this PR does not depend on it.
        int? sharedExpertIntermediate;
        int numSharedExperts = 1;
        bool hasSharedGate;
        bool isDeepSeek = root.TryGetProperty("n_shared_experts", out _);
        if (isDeepSeek)
        {
            int nShared = GetInt32OrDefault(root, "n_shared_experts", 0);
            if (nShared > 0)
            {
                sharedExpertIntermediate = moeIntermediateSize;
                numSharedExperts = nShared;
            }
            else
            {
                sharedExpertIntermediate = null;
            }
            hasSharedGate = false; // DeepSeek does NOT gate the shared expert.
        }
        else
        {
            // Qwen1.5-MoE-A2.7B ships shared_expert_intermediate_size; absent
            // on Mixtral, Phi-3.5-MoE, and Qwen3-MoE.
            sharedExpertIntermediate = GetInt32NullableIfPositive(root, "shared_expert_intermediate_size");
            // shared_expert_gate is a tensor (not a config key), so we default
            // to "present iff the model declares a shared expert" — the
            // safetensors loader turns this back off if the tensor is missing.
            // Qwen1.5-MoE always ships it when shared_expert_intermediate_size
            // is set.
            hasSharedGate = sharedExpertIntermediate is not null;
            // Qwen1.5-MoE ships a single shared expert; keep the default of 1.
        }

        // Qwen3-MoE layer-level sparsity: decoder_sparse_step (default 1 —
        // every layer is MoE) and mlp_only_layers (force-dense overrides).
        int decoderSparseStep = GetInt32OrDefault(root, "decoder_sparse_step", 1);
        if (decoderSparseStep <= 0) decoderSparseStep = 1;
        IReadOnlyList<int>? mlpOnlyLayers = GetInt32ArrayOrDefault(root, "mlp_only_layers");

        return new MoeConfig
        {
            NumExperts = numExperts,
            NumExpertsPerTok = numExpertsPerTok,
            MoeIntermediateSize = moeIntermediateSize,
            NormTopKProb = normTopKProb,
            SharedExpertIntermediateSize = sharedExpertIntermediate,
            NumSharedExperts = numSharedExperts,
            HasSharedExpertGate = hasSharedGate,
            DecoderSparseStep = decoderSparseStep,
            MlpOnlyLayers = mlpOnlyLayers,
        };
    }

    private static IReadOnlyList<int>? GetInt32ArrayOrDefault(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Array)
            return null;
        int len = prop.GetArrayLength();
        if (len == 0) return null;
        var result = new int[len];
        int i = 0;
        foreach (var el in prop.EnumerateArray())
        {
            if (el.ValueKind != JsonValueKind.Number || !el.TryGetInt32(out int v))
                return null;
            result[i++] = v;
        }
        return result;
    }

    /// <summary>
    /// Peeks at <c>model_type</c> / <c>architectures[0]</c> so the caller
    /// (e.g. <c>ModelLoader.LoadFromSafetensors</c>) can pre-dispatch before
    /// running the full extractor.
    /// </summary>
    public static Architecture ResolveArchitecture(JsonElement root)
    {
        string? archName = null;
        if (root.TryGetProperty("architectures", out var archArr)
            && archArr.ValueKind == JsonValueKind.Array
            && archArr.GetArrayLength() > 0)
        {
            var first = archArr[0];
            if (first.ValueKind == JsonValueKind.String)
                archName = first.GetString();
        }

        string? modelType = GetStringOrDefault(root, "model_type", null);

        return (archName?.ToLowerInvariant(), modelType?.ToLowerInvariant()) switch
        {
            // DeepSeek-V3 must be checked before V2 and before any Llama/Mistral
            // fallback — architectures[0] = 'DeepseekV3ForCausalLM'.
            (var a, _) when a is not null && a.Contains("deepseekv3") => Architecture.DeepSeekV3,
            (_, "deepseek_v3") => Architecture.DeepSeekV3,
            (var a, _) when a is not null && a.Contains("deepseekv2") => Architecture.DeepSeekV2,
            (_, "deepseek_v2") => Architecture.DeepSeekV2,
            // Mixtral must be checked before generic "mistral" — the architecture
            // class name is 'MixtralForCausalLM' but the organization namespace
            // is mistralai, so a substring match for "mistral" would otherwise
            // shadow it.
            (var a, _) when a is not null && a.Contains("mixtral") => Architecture.Mixtral,
            (_, "mixtral") => Architecture.Mixtral,
            // Qwen-MoE variants must be checked before generic "qwen" — the
            // architecture class name is Qwen{2,3}MoeForCausalLM.
            (var a, _) when a is not null && (a.Contains("qwen2moe") || a.Contains("qwen3moe")
                || a.Contains("qwen2_moe") || a.Contains("qwen3_moe")
                || a.Contains("qwenmoe") || a.Contains("qwen_moe")) => Architecture.QwenMoe,
            (_, "qwen2_moe" or "qwen3_moe" or "qwen_moe") => Architecture.QwenMoe,
            (var a, _) when a is not null && a.Contains("llama") => Architecture.Llama,
            (var a, _) when a is not null && a.Contains("mistral") => Architecture.Mistral,
            (var a, _) when a is not null && a.StartsWith("phi") => Architecture.Phi,
            (var a, _) when a is not null && a.Contains("qwen") => Architecture.Qwen,
            (_, "llama") => Architecture.Llama,
            (_, "mistral") => Architecture.Mistral,
            (_, "phi" or "phi3" or "phi2") => Architecture.Phi,
            (_, "qwen" or "qwen2" or "qwen3") => Architecture.Qwen,
            _ => throw new InvalidDataException(
                $"Unsupported HF architecture: architectures[0]='{archName}', model_type='{modelType}'.")
        };
    }

    /// <summary>
    /// Default tie-embeddings behaviour for architectures where HF typically
    /// omits the key. Gemma/Phi3 tie by default; Llama/Mistral/Qwen don't.
    /// Safest behaviour is "don't tie unless declared", which matches the
    /// spec for Llama/Mistral/Qwen. Phi's config almost always states it
    /// explicitly so this fallback rarely fires.
    /// </summary>
    private static bool DefaultTieForArch(Architecture arch) => arch switch
    {
        Architecture.Phi => true,
        _ => false,
    };

    private static int GetInt32(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            throw new InvalidDataException($"HF config.json missing required integer key '{key}'.");
        if (!prop.TryGetInt32(out int value))
            throw new InvalidDataException($"HF config.json key '{key}' is not a 32-bit integer.");
        return value;
    }

    private static int GetInt32OrDefault(JsonElement root, string key, int fallback)
    {
        if (!root.TryGetProperty(key, out var prop)) return fallback;
        // HF sometimes stores None as JSON null (e.g. num_key_value_heads) —
        // defensively coerce that to the fallback.
        if (prop.ValueKind != JsonValueKind.Number) return fallback;
        return prop.TryGetInt32(out int value) ? value : fallback;
    }

    private static int? GetInt32NullableIfPositive(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out var prop)) return null;
        if (prop.ValueKind != JsonValueKind.Number) return null;
        if (!prop.TryGetInt32(out int v)) return null;
        return v > 0 ? v : null;
    }

    private static float GetFloatOrDefault(JsonElement root, string key, float fallback)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.Number)
            return fallback;
        return prop.TryGetSingle(out float value) ? value : fallback;
    }

    private static bool GetBoolOrDefault(JsonElement root, string key, bool fallback)
    {
        if (!root.TryGetProperty(key, out var prop)) return fallback;
        return prop.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => fallback,
        };
    }

    private static string? GetStringOrDefault(JsonElement root, string key, string? fallback)
    {
        if (!root.TryGetProperty(key, out var prop) || prop.ValueKind != JsonValueKind.String)
            return fallback;
        return prop.GetString() ?? fallback;
    }
}
