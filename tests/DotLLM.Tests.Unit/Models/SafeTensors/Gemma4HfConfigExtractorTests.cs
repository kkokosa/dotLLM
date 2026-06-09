using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// HF <c>config.json</c> extraction coverage for the Gemma 4 family
/// (<see cref="Architecture.Gemma4"/>). Mirrors the Gemma 3 coverage in
/// <see cref="HfConfigExtractorTests"/> but exercises the Gemma 4 model_type
/// + architectures variants released by Google in 2026-04 — text-only
/// (<c>Gemma4ForCausalLM</c> / <c>gemma4_text</c>), multimodal
/// (<c>Gemma4ForConditionalGeneration</c> / <c>gemma4</c>), and the newer
/// "unified" multimodal release (<c>Gemma4UnifiedForConditionalGeneration</c>
/// / <c>gemma4_unified</c>).
/// </summary>
public sealed class Gemma4HfConfigExtractorTests
{
    /// <summary>
    /// Gemma 4 text-only checkpoint (<c>google/gemma-4-12B-*</c>'s
    /// <c>text_config</c> reshaped as a standalone causal-LM config). Verifies
    /// architecture detection, the Gemma-family soft-cap + sliding-window
    /// plumbing, GeluTanh activation, NeoX RoPE, and tied embeddings.
    /// </summary>
    [Fact]
    public void Gemma4_TextOnly_PopulatesGemmaFields()
    {
        // Compact subset of the real google/gemma-4-12B `text_config` shape,
        // truncated to 6 layers so the synthetic `layer_types` block stays
        // legible. 5 sliding + 1 full preserves the actual Gemma 4 pattern.
        const string json = """
        {
            "architectures": ["Gemma4ForCausalLM"],
            "model_type": "gemma4_text",
            "hidden_size": 32,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 128,
            "vocab_size": 256,
            "max_position_embeddings": 4096,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "sliding_window": 1024,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ],
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": true
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);

        Assert.Equal(Architecture.Gemma4, cfg.Architecture);
        Assert.Equal(32, cfg.HiddenSize);
        Assert.Equal(6, cfg.NumLayers);
        Assert.Equal(4, cfg.NumAttentionHeads);
        Assert.Equal(2, cfg.NumKvHeads);
        Assert.Equal(8, cfg.HeadDim);
        Assert.Equal(128, cfg.IntermediateSize);
        Assert.Equal(256, cfg.VocabSize);
        Assert.Equal(4096, cfg.MaxSequenceLength);

        // Sliding-window per-layer mask: 5 sliding + 1 full (the canonical
        // Gemma 4 12B/31B pattern). Indices 0..4 carry the window; index 5
        // is null (full attention).
        Assert.NotNull(cfg.PerLayerSlidingWindow);
        Assert.Equal(6, cfg.PerLayerSlidingWindow!.Count);
        for (int i = 0; i < 5; i++)
            Assert.Equal(1024, cfg.PerLayerSlidingWindow[i]);
        Assert.Null(cfg.PerLayerSlidingWindow[5]);

        Assert.Equal(1024, cfg.SlidingWindowSize);
        Assert.Null(cfg.AttnLogitSoftcap);
        Assert.Equal(30.0f, cfg.FinalLogitSoftcap);
        Assert.Null(cfg.QueryPreAttnScalar);
        Assert.Equal(ActivationFunction.GELUTanh, cfg.ActivationFunction);
        Assert.Equal(NormType.RMSNorm, cfg.NormType);
        Assert.Equal(1_000_000.0f, cfg.RoPEConfig!.Value.Theta);
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig.Value.Type);
        Assert.True(cfg.TiedEmbeddings);
    }

    /// <summary>
    /// Gemma 4 multimodal checkpoint (<c>Gemma4ForConditionalGeneration</c> /
    /// <c>model_type=gemma4</c>) houses the text-tower config under a
    /// <c>text_config</c> sub-object. The extractor must hoist that sub-object
    /// so every field is read from the text tower, exactly as for Gemma 3
    /// multimodal.
    /// </summary>
    [Fact]
    public void Gemma4_Multimodal_HoistsTextConfig()
    {
        const string json = """
        {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "vision_config": {
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "patch_size": 16
            },
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "intermediate_size": 64,
                "vocab_size": 256,
                "max_position_embeddings": 2048,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-6,
                "hidden_activation": "gelu_pytorch_tanh",
                "sliding_window": 512,
                "layer_types": [
                    "sliding_attention", "sliding_attention",
                    "sliding_attention", "full_attention"
                ],
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true
            }
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);

        Assert.Equal(Architecture.Gemma4, cfg.Architecture);
        // Values come from text_config, NOT the top-level.
        Assert.Equal(32, cfg.HiddenSize);
        Assert.Equal(4, cfg.NumLayers);
        Assert.Equal(512, cfg.SlidingWindowSize);
        Assert.Equal(30.0f, cfg.FinalLogitSoftcap);
        Assert.NotNull(cfg.PerLayerSlidingWindow);
        Assert.Equal(4, cfg.PerLayerSlidingWindow!.Count);
        Assert.Equal(512, cfg.PerLayerSlidingWindow[0]);
        Assert.Null(cfg.PerLayerSlidingWindow[3]);
    }

    /// <summary>
    /// Newer "unified" multimodal Gemma 4 release
    /// (<c>Gemma4UnifiedForConditionalGeneration</c> /
    /// <c>model_type=gemma4_unified</c>) carries an audio_config tower in
    /// addition to vision. The text-tower hoist must still find the
    /// <c>text_config</c> sub-object — its <c>model_type</c> is the variant
    /// <c>gemma4_unified_text</c>, which our resolver maps onto Gemma 4 too.
    /// </summary>
    [Fact]
    public void Gemma4_UnifiedMultimodal_DetectsArchAndHoistsTextConfig()
    {
        const string json = """
        {
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
            "audio_config": { "hidden_size": 640 },
            "vision_config": { "hidden_size": 1152 },
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 32,
                "num_hidden_layers": 6,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "intermediate_size": 128,
                "vocab_size": 256,
                "max_position_embeddings": 2048,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-6,
                "hidden_activation": "gelu_pytorch_tanh",
                "sliding_window": 1024,
                "layer_types": [
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention"
                ],
                "final_logit_softcapping": 30.0,
                "tie_word_embeddings": true
            }
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Gemma4, cfg.Architecture);
        Assert.Equal(32, cfg.HiddenSize);
        Assert.Equal(6, cfg.NumLayers);
        Assert.True(cfg.TiedEmbeddings);
    }

    /// <summary>
    /// Defensive coverage for the <c>tie_word_embeddings</c> default. Gemma 4
    /// ships this true in every public SKU but if a config omits it the
    /// fallback must still tie (<see cref="HfConfigExtractor"/>'s
    /// DefaultTieForArch returns true for Gemma 4).
    /// </summary>
    [Fact]
    public void Gemma4_OmittedTieFlag_DefaultsToTied()
    {
        // Same shape as the text-only test, just without `tie_word_embeddings`.
        const string json = """
        {
            "architectures": ["Gemma4ForCausalLM"],
            "model_type": "gemma4_text",
            "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 2, "intermediate_size": 32, "vocab_size": 64,
            "max_position_embeddings": 256, "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6, "sliding_window": 64,
            "layer_types": ["sliding_attention", "full_attention"]
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Gemma4, cfg.Architecture);
        Assert.True(cfg.TiedEmbeddings);
    }

    /// <summary>
    /// Architecture priority — when an HF config carries both a Gemma 4
    /// architecture class name and any other Gemma substring, the version
    /// match must win. Equivalent to the existing Gemma 3 priority check;
    /// guards against future <c>"gemma"</c>-substring rules silently
    /// shadowing the version-specific row.
    /// </summary>
    [Fact]
    public void Gemma4_TakesPriority_OverGenericGemmaPattern()
    {
        const string json = """
        {
            "architectures": ["Gemma4ForCausalLM"],
            "model_type": "gemma4_text",
            "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 2, "intermediate_size": 32, "vocab_size": 64,
            "max_position_embeddings": 256, "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6, "sliding_window": 64,
            "layer_types": ["sliding_attention", "full_attention"]
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Gemma4, cfg.Architecture);
        Assert.NotEqual(Architecture.Gemma3, cfg.Architecture);
    }
}
