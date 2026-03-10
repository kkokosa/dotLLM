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
            RoPEConfig = ropeConfig,
            PositionEncodingType = ropeConfig.HasValue ? PositionEncodingType.RoPE : PositionEncodingType.None,
            SlidingWindowSize = slidingWindowSize,
            ChatTemplate = chatTemplate,
        };
    }

    private static Architecture ParseArchitecture(string archString)
    {
        return archString.ToLowerInvariant() switch
        {
            "llama" => Architecture.Llama,
            "mistral" => Architecture.Mistral,
            "phi" or "phi2" or "phi3" => Architecture.Phi,
            "qwen" or "qwen2" or "qwen3" => Architecture.Qwen,
            "deepseek" or "deepseek2" => Architecture.DeepSeek,
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
