using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Extracts a <see cref="ModelConfig"/> from GGUF metadata following standard GGUF key conventions.
/// </summary>
public static class GgufModelConfigExtractor
{
    /// <summary>
    /// Builds a <see cref="ModelConfig"/> from the given GGUF metadata.
    /// </summary>
    /// <param name="metadata">Parsed GGUF metadata.</param>
    /// <returns>A fully populated <see cref="ModelConfig"/>.</returns>
    /// <exception cref="InvalidDataException">Required metadata keys are missing or have invalid values.</exception>
    public static ModelConfig Extract(GgufMetadata metadata)
    {
        string archString = metadata.GetString("general.architecture");
        Architecture architecture = ParseArchitecture(archString);
        string arch = archString.ToLowerInvariant();

        int hiddenSize = (int)metadata.GetUInt32($"{arch}.embedding_length");
        int numLayers = (int)metadata.GetUInt32($"{arch}.block_count");
        int intermediateSize = (int)metadata.GetUInt32($"{arch}.feed_forward_length");
        int numAttentionHeads = (int)metadata.GetUInt32($"{arch}.attention.head_count");
        int numKvHeads = (int)metadata.GetUInt32OrDefault($"{arch}.attention.head_count_kv", (uint)numAttentionHeads);

        // Head dimension: prefer explicit GGUF key (needed for models like Qwen3 where
        // head_dim != hidden_size / num_heads), fall back to derived value.
        // For DeepSeek-V2/V3 MLA, key_length is the qk_nope_head_dim only — total
        // qk_head_dim is qk_nope + qk_rope; HeadDim is fixed up after MLA config
        // extraction below.
        int headDim = (int)metadata.GetUInt32OrDefault($"{arch}.attention.key_length",
                                                        (uint)(hiddenSize / numAttentionHeads));
        int maxSeqLen = (int)metadata.GetUInt32OrDefault($"{arch}.context_length", 2048);

        float normEps = metadata.GetFloat32OrDefault($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);

        int? slidingWindowSize = null;
        uint swValue = metadata.GetUInt32OrDefault($"{arch}.attention.sliding_window", 0);
        if (swValue > 0)
            slidingWindowSize = (int)swValue;

        int vocabSize = ResolveVocabSize(metadata, arch);

        string? chatTemplate = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        if (string.IsNullOrEmpty(chatTemplate))
            chatTemplate = null;

        RoPEConfig? ropeConfig = ExtractRoPEConfig(metadata, arch, headDim, architecture);

        // DeepSeek-V2/V3: extract MLA config and patch HeadDim to the full
        // qk_head_dim (key_length stores qk_nope only; total = qk_nope + qk_rope).
        // MoE config detection ships in the parallel DeepSeek-GGUF A-2 PR
        // (3D-stacked-expert tensor loader).
        MlaConfig? mlaConfig = null;
        AttentionType attentionType = AttentionType.GQA;
        if (architecture is Architecture.DeepSeekV2 or Architecture.DeepSeekV3)
        {
            mlaConfig = ExtractMlaConfig(metadata, arch, ropeConfig);
            attentionType = AttentionType.MLA;
            // GGUF's attention.key_length is qk_nope only. Total per-head dim
            // for MLA attention is qk_nope + qk_rope — patch HeadDim so the
            // GQA-shaped pieces of the model (cache stride etc.) see the full
            // value.
            headDim = mlaConfig.QkNopeHeadDim + mlaConfig.QkRopeHeadDim;
        }

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
            NormEpsilon = normEps,
            AttentionType = attentionType,
            RoPEConfig = ropeConfig,
            PositionEncodingType = ropeConfig.HasValue ? PositionEncodingType.RoPE : PositionEncodingType.None,
            SlidingWindowSize = slidingWindowSize,
            MlaConfig = mlaConfig,
            ChatTemplate = chatTemplate,
        };
    }

    /// <summary>
    /// Extracts an <see cref="MlaConfig"/> from DeepSeek-V2/V3 GGUF metadata.
    /// Required keys (per llama.cpp's gguf_writer):
    /// <list type="bullet">
    ///   <item><c>{arch}.attention.q_lora_rank</c> — Q LoRA bottleneck (0 = monolithic, V2-Lite default)</item>
    ///   <item><c>{arch}.attention.kv_lora_rank</c> — KV LoRA bottleneck (typically 512)</item>
    ///   <item><c>{arch}.attention.key_length</c> — TOTAL per-head qk dim (qk_nope + qk_rope; 192 on V2-Lite)</item>
    ///   <item><c>{arch}.attention.value_length</c> — v_head_dim (may differ from qk_nope_head_dim)</item>
    ///   <item><c>{arch}.rope.dimension_count</c> — qk_rope_head_dim (must be even; 64 on V2-Lite)</item>
    /// </list>
    /// <para>
    /// <b>qk_nope_head_dim is derived</b> as <c>key_length - rope.dimension_count</c>.
    /// Confirmed against the bartowski DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf
    /// (key_length=192, rope.dimension_count=64 ⇒ qk_nope=128).
    /// </para>
    /// </summary>
    private static MlaConfig ExtractMlaConfig(GgufMetadata metadata, string arch, RoPEConfig? ropeConfig)
    {
        // q_lora_rank may be absent or zero on V2-Lite (monolithic-Q variant).
        int qLoraRank = (int)metadata.GetUInt32OrDefault($"{arch}.attention.q_lora_rank", 0);
        int kvLoraRank = (int)metadata.GetUInt32($"{arch}.attention.kv_lora_rank");
        int qkTotal = (int)metadata.GetUInt32($"{arch}.attention.key_length");
        int vHead = (int)metadata.GetUInt32($"{arch}.attention.value_length");
        int qkRope = (int)metadata.GetUInt32($"{arch}.rope.dimension_count");
        int qkNope = qkTotal - qkRope;

        if (kvLoraRank <= 0)
            throw new InvalidDataException(
                $"DeepSeek-V2 MLA requires '{arch}.attention.kv_lora_rank' > 0; got {kvLoraRank}.");
        if (qkRope <= 0 || (qkRope & 1) != 0)
            throw new InvalidDataException(
                $"DeepSeek-V2 MLA requires '{arch}.rope.dimension_count' (qk_rope) to be a positive even number; got {qkRope}.");
        if (qkNope <= 0)
            throw new InvalidDataException(
                $"DeepSeek-V2 MLA requires '{arch}.attention.key_length' > '{arch}.rope.dimension_count' " +
                $"(qk_nope = key_length - rope.dimension_count); got {qkTotal} and {qkRope}.");

        float ropeTheta = ropeConfig?.Theta ?? 10000.0f;

        // YaRN params (when rope.scaling.type=yarn). Already extracted into
        // ropeConfig but MLA carries its own copy for the standalone MLA
        // softmax-scale correction (see MlaConfig.ComputeYarnSoftmaxScaleMultiplier).
        float? ropeScalingFactor = null;
        float? ropeScalingMscale = null;
        float? ropeScalingMscaleAllDim = null;
        int? ropeScalingOrigCtx = null;
        if (ropeConfig is { ScalingType: RoPEScalingType.YaRN } yarn)
        {
            ropeScalingFactor = yarn.ScalingFactor;
            ropeScalingMscale = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.yarn_log_multiplier", 0.0f);
            ropeScalingMscaleAllDim = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.attn_factor", 1.0f);
            ropeScalingOrigCtx = yarn.OrigMaxSeqLen > 0 ? yarn.OrigMaxSeqLen : null;
        }

        return new MlaConfig
        {
            KvLoraRank = kvLoraRank,
            QLoraRank = qLoraRank,
            QkNopeHeadDim = qkNope,
            QkRopeHeadDim = qkRope,
            VHeadDim = vHead,
            RopeTheta = ropeTheta,
            RopeScalingFactor = ropeScalingFactor,
            RopeScalingMscale = ropeScalingMscale,
            RopeScalingMscaleAllDim = ropeScalingMscaleAllDim,
            RopeScalingOriginalMaxPositionEmbeddings = ropeScalingOrigCtx,
        };
    }

    private static Architecture ParseArchitecture(string archString)
    {
        return archString.ToLowerInvariant() switch
        {
            "llama" => Architecture.Llama,
            "mistral" or "mistral3" => Architecture.Mistral,
            "phi" or "phi2" or "phi3" => Architecture.Phi,
            "qwen" or "qwen2" or "qwen3" => Architecture.Qwen,
            // Pre-V2 DeepSeek (legacy placeholder — never actually loaded by us).
            "deepseek" => Architecture.DeepSeek,
            // V2 / V2-Lite — MLA + MoE per <c>convert_hf_to_gguf.py</c>'s
            // <c>DeepseekV2Model</c>. Distinct from V3 only in routing details.
            "deepseek2" => Architecture.DeepSeekV2,
            // V3 / V3-MoE — MLA + sigmoid-gated routing + group-norm experts.
            "deepseek3" => Architecture.DeepSeekV3,
            _ => throw new InvalidDataException($"Unsupported GGUF architecture: '{archString}'.")
        };
    }

    private static int ResolveVocabSize(GgufMetadata metadata, string arch)
    {
        uint vocabSize = metadata.GetUInt32OrDefault($"{arch}.vocab_size", 0);
        if (vocabSize > 0)
            return (int)vocabSize;

        // Fallback: count entries in the tokenizer vocabulary array.
        if (metadata.ContainsKey("tokenizer.ggml.tokens"))
        {
            string[] tokens = metadata.GetStringArray("tokenizer.ggml.tokens");
            return tokens.Length;
        }

        throw new InvalidDataException(
            "Cannot determine vocabulary size: neither '{arch}.vocab_size' nor 'tokenizer.ggml.tokens' found.");
    }

    private static RoPEConfig? ExtractRoPEConfig(GgufMetadata metadata, string arch, int headDim,
        Architecture architecture)
    {
        // If no rope keys exist at all, this model may not use RoPE.
        string freqBaseKey = $"{arch}.rope.freq_base";
        string dimCountKey = $"{arch}.rope.dimension_count";
        if (!metadata.ContainsKey(freqBaseKey) && !metadata.ContainsKey(dimCountKey))
            return null;

        float theta = metadata.GetFloat32OrDefault(freqBaseKey, 10000.0f);
        int dimCount = (int)metadata.GetUInt32OrDefault(dimCountKey, (uint)headDim);

        // Determine RoPE element-pairing convention. Must match the GGUF Q/K weight layout:
        // - Llama/Mistral: converter permutes Q/K weights → interleaved (Norm)
        // - Qwen/Phi: weights kept in HuggingFace order → non-interleaved (NeoX)
        RoPEType ropeType = architecture switch
        {
            Architecture.Qwen or Architecture.Phi => RoPEType.NeoX,
            _ => RoPEType.Norm,
        };

        RoPEScalingType scalingType = RoPEScalingType.None;
        float scalingFactor = 1.0f;
        int origMaxSeqLen = 0;
        float attnFactor = 1.0f;
        float betaFast = 32.0f;
        float betaSlow = 1.0f;

        string scalingTypeKey = $"{arch}.rope.scaling.type";
        if (metadata.ContainsKey(scalingTypeKey))
        {
            string scalingTypeStr = metadata.GetString(scalingTypeKey);
            scalingType = scalingTypeStr.ToLowerInvariant() switch
            {
                "linear" => RoPEScalingType.Linear,
                "yarn" => RoPEScalingType.YaRN,
                "ntk" => RoPEScalingType.NTK,
                "dynamic" or "dynamic_ntk" => RoPEScalingType.DynamicNTK,
                "su" or "longrope" => RoPEScalingType.Su,
                _ => RoPEScalingType.None
            };

            scalingFactor = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.factor", 1.0f);
            origMaxSeqLen = (int)metadata.GetUInt32OrDefault($"{arch}.rope.scaling.original_context_length", 0);
            attnFactor = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.attn_factor", 1.0f);
            betaFast = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.beta_fast", 32.0f);
            betaSlow = metadata.GetFloat32OrDefault($"{arch}.rope.scaling.beta_slow", 1.0f);
        }

        return new RoPEConfig(
            Theta: theta,
            DimensionCount: dimCount,
            Type: ropeType,
            ScalingType: scalingType,
            ScalingFactor: scalingFactor,
            OrigMaxSeqLen: origMaxSeqLen,
            AttnFactor: attnFactor,
            BetaFast: betaFast,
            BetaSlow: betaSlow);
    }
}
