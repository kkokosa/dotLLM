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
        // Llama/Mistral/DeepSeek-V2 use interleaved (Norm); Qwen/Phi use non-interleaved (NeoX).
        RoPEType ropeType = architecture switch
        {
            Architecture.Qwen or Architecture.Phi => RoPEType.NeoX,
            _ => RoPEType.Norm,
        };

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
