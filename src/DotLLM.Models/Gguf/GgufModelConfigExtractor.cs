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
        int numAttentionHeads = (int)metadata.GetUInt32($"{arch}.attention.head_count");

        // Hybrid models (Nemotron-H) store head_count_kv and feed_forward_length as
        // per-layer Int32 arrays whose entries are zero for layers of the wrong kind.
        // Build a HybridLayerLayout in that case; for pure-Transformer architectures
        // both keys are scalar UInt32.
        HybridLayerLayout? hybridLayout = TryExtractHybridLayout(metadata, arch, numLayers);

        int intermediateSize;
        int numKvHeads;
        if (hybridLayout is not null)
        {
            // Use the *attention-layer* values as the canonical scalar config so existing
            // attention/KV-cache code paths see meaningful sizes. Fall back to zeros only
            // when the model has no attention layers at all (unsupported here).
            numKvHeads = MaxNonZero(hybridLayout.HeadCountKv, numAttentionHeads);
            intermediateSize = MaxNonZero(hybridLayout.FeedForwardLength, 0);
        }
        else
        {
            intermediateSize = (int)metadata.GetUInt32($"{arch}.feed_forward_length");
            numKvHeads = (int)metadata.GetUInt32OrDefault($"{arch}.attention.head_count_kv", (uint)numAttentionHeads);
        }

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
        MambaSsmConfig? ssmConfig = TryExtractSsmConfig(metadata, arch);

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
            ActivationFunction = architecture == Architecture.NemotronH
                ? ActivationFunction.ReluSquared
                : ActivationFunction.SiLU,
            RoPEConfig = ropeConfig,
            PositionEncodingType = ropeConfig.HasValue ? PositionEncodingType.RoPE : PositionEncodingType.None,
            SlidingWindowSize = slidingWindowSize,
            HybridLayout = hybridLayout,
            SsmConfig = ssmConfig,
            ChatTemplate = chatTemplate,
        };
    }

    private static HybridLayerLayout? TryExtractHybridLayout(GgufMetadata metadata, string arch, int numLayers)
    {
        string kvKey = $"{arch}.attention.head_count_kv";
        string ffKey = $"{arch}.feed_forward_length";

        if (!metadata.TryGetValue(kvKey, out var kvEntry) || kvEntry.Type != GgufValueType.Array) return null;
        if (!metadata.TryGetValue(ffKey, out var ffEntry) || ffEntry.Type != GgufValueType.Array) return null;

        // Both keys are per-layer Int32 arrays in hybrid models (Nemotron-H).
        int[] headCountKv = metadata.GetInt32Array(kvKey);
        int[] feedForwardLength = metadata.GetInt32Array(ffKey);

        if (headCountKv.Length != numLayers)
            throw new InvalidDataException(
                $"'{kvKey}' array length {headCountKv.Length} does not match block_count {numLayers}.");
        if (feedForwardLength.Length != numLayers)
            throw new InvalidDataException(
                $"'{ffKey}' array length {feedForwardLength.Length} does not match block_count {numLayers}.");

        var kinds = new HybridLayerKind[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            bool hasAttn = headCountKv[i] > 0;
            bool hasFfn = feedForwardLength[i] > 0;
            kinds[i] = (hasAttn, hasFfn) switch
            {
                (true, false) => HybridLayerKind.Attention,
                (false, true) => HybridLayerKind.Ffn,
                (false, false) => HybridLayerKind.Ssm,
                (true, true) => throw new InvalidDataException(
                    $"Layer {i} has both non-zero head_count_kv and feed_forward_length; hybrid block kinds must be exclusive.")
            };
        }

        return new HybridLayerLayout
        {
            LayerKind = kinds,
            HeadCountKv = headCountKv,
            FeedForwardLength = feedForwardLength,
        };
    }

    private static MambaSsmConfig? TryExtractSsmConfig(GgufMetadata metadata, string arch)
    {
        string innerKey = $"{arch}.ssm.inner_size";
        if (!metadata.ContainsKey(innerKey)) return null;

        int dConv = (int)metadata.GetUInt32($"{arch}.ssm.conv_kernel");
        int dInner = (int)metadata.GetUInt32(innerKey);
        int dState = (int)metadata.GetUInt32($"{arch}.ssm.state_size");
        int nGroup = (int)metadata.GetUInt32OrDefault($"{arch}.ssm.group_count", 1);
        int nHead = (int)metadata.GetUInt32($"{arch}.ssm.time_step_rank");

        if (dInner % nHead != 0)
            throw new InvalidDataException(
                $"SSM inner_size {dInner} not divisible by time_step_rank {nHead}.");
        if (dInner % nGroup != 0)
            throw new InvalidDataException(
                $"SSM inner_size {dInner} not divisible by group_count {nGroup}.");
        if (nHead % nGroup != 0)
            throw new InvalidDataException(
                $"SSM time_step_rank {nHead} not divisible by group_count {nGroup}.");

        return new MambaSsmConfig(dConv, dInner, dState, nGroup, nHead);
    }

    private static int MaxNonZero(int[] values, int fallback)
    {
        int max = 0;
        foreach (int v in values) if (v > max) max = v;
        return max > 0 ? max : fallback;
    }

    private static Architecture ParseArchitecture(string archString)
    {
        return archString.ToLowerInvariant() switch
        {
            "llama" => Architecture.Llama,
            "mistral" or "mistral3" => Architecture.Mistral,
            "phi" or "phi2" or "phi3" => Architecture.Phi,
            "qwen" or "qwen2" or "qwen3" => Architecture.Qwen,
            "deepseek" or "deepseek2" => Architecture.DeepSeek,
            "nemotron_h" => Architecture.NemotronH,
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
