using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Unit tests for <see cref="HfConfigExtractor"/> — the HuggingFace
/// <c>config.json</c> → <see cref="Core.Models.ModelConfig"/> parser.
/// </summary>
public sealed class HfConfigExtractorTests
{
    [Fact]
    public void Llama_MinimalConfig_PopulatesCoreFields()
    {
        const string json = """
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 256,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-5
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);

        Assert.Equal(Architecture.Llama, cfg.Architecture);
        Assert.Equal(128, cfg.HiddenSize);
        Assert.Equal(2, cfg.NumLayers);
        Assert.Equal(4, cfg.NumAttentionHeads);
        Assert.Equal(2, cfg.NumKvHeads);
        Assert.Equal(256, cfg.IntermediateSize);
        Assert.Equal(1000, cfg.VocabSize);
        Assert.Equal(512, cfg.MaxSequenceLength);
        Assert.Equal(32, cfg.HeadDim); // 128 / 4
        Assert.Equal(1e-5f, cfg.NormEpsilon);
        Assert.Equal(PositionEncodingType.RoPE, cfg.PositionEncodingType);
        Assert.NotNull(cfg.RoPEConfig);
        Assert.Equal(500000.0f, cfg.RoPEConfig!.Value.Theta);
        Assert.Equal(RoPEType.Norm, cfg.RoPEConfig.Value.Type);
        Assert.False(cfg.TiedEmbeddings);
    }

    [Fact]
    public void Mistral_UsesNormRoPE()
    {
        const string json = """
        {
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "intermediate_size": 128, "vocab_size": 500, "max_position_embeddings": 256,
            "sliding_window": 64
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Mistral, cfg.Architecture);
        Assert.Equal(4, cfg.NumKvHeads); // defaults to num_attention_heads
        Assert.Equal(64, cfg.SlidingWindowSize);
        Assert.Equal(RoPEType.Norm, cfg.RoPEConfig!.Value.Type);
    }

    [Fact]
    public void Phi_UsesNeoXRoPE_AndTiesByDefault()
    {
        const string json = """
        {
            "architectures": ["Phi3ForCausalLM"],
            "model_type": "phi3",
            "hidden_size": 96, "num_hidden_layers": 2, "num_attention_heads": 4,
            "intermediate_size": 192, "vocab_size": 500, "max_position_embeddings": 256
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Phi, cfg.Architecture);
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig!.Value.Type);
        Assert.True(cfg.TiedEmbeddings);
    }

    [Fact]
    public void Qwen_UsesNeoXRoPE_AndExplicitHeadDim()
    {
        const string json = """
        {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 256, "vocab_size": 500, "max_position_embeddings": 256,
            "head_dim": 48,
            "tie_word_embeddings": false
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Qwen, cfg.Architecture);
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig!.Value.Type);
        Assert.Equal(48, cfg.HeadDim);
        Assert.False(cfg.TiedEmbeddings);
    }

    [Fact]
    public void NullNumKvHeads_FallsBackToAttentionHeads()
    {
        // HF checkpoints sometimes emit `"num_key_value_heads": null` to mean
        // "use num_attention_heads". JSON null must not crash the parser.
        const string json = """
        {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": null,
            "intermediate_size": 128, "vocab_size": 100, "max_position_embeddings": 128
        }
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(4, cfg.NumKvHeads);
    }

    [Fact]
    public void UnsupportedArchitecture_Throws()
    {
        const string json = """
        {"architectures": ["BertForMaskedLM"], "model_type": "bert",
         "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
         "intermediate_size": 128, "vocab_size": 100, "max_position_embeddings": 128}
        """;
        var ex = Assert.Throws<InvalidDataException>(() => HfConfigExtractor.Extract(json));
        Assert.Contains("Unsupported HF architecture", ex.Message);
    }

    /// <summary>
    /// DeepSeek-V2-Lite (deepseek-ai/DeepSeek-V2-Lite) — verifies MLA detection,
    /// <see cref="Core.Models.MlaConfig"/> population, and that
    /// <c>head_dim</c> reuses <c>qk_head_dim = qk_nope + qk_rope</c>.
    /// MoE assertions are deferred to the MoE extraction PR — this PR only
    /// covers the MLA attention foundation.
    /// </summary>
    [Fact]
    public void DeepSeekV2Lite_PopulatesMla()
    {
        const string json = """
        {
            "architectures": ["DeepseekV2ForCausalLM"],
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "intermediate_size": 10944,
            "vocab_size": 102400,
            "max_position_embeddings": 163840,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "kv_lora_rank": 512,
            "q_lora_rank": 0,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);

        Assert.Equal(Architecture.DeepSeekV2, cfg.Architecture);
        Assert.Equal(AttentionType.MLA, cfg.AttentionType);

        Assert.NotNull(cfg.MlaConfig);
        Assert.Equal(512, cfg.MlaConfig!.KvLoraRank);
        Assert.Equal(0, cfg.MlaConfig.QLoraRank);
        Assert.Equal(128, cfg.MlaConfig.QkNopeHeadDim);
        Assert.Equal(64, cfg.MlaConfig.QkRopeHeadDim);
        Assert.Equal(128, cfg.MlaConfig.VHeadDim);
        Assert.Equal(192, cfg.MlaConfig.QkHeadDim);  // 128 + 64
        Assert.Equal(192, cfg.HeadDim);              // HeadDim reuses qk_head_dim
    }

    /// <summary>
    /// DeepSeek-V2 full (non-Lite) uses <c>q_lora_rank = 1536</c>. Verifies
    /// the optional Q-factorisation rank is captured into
    /// <see cref="Core.Models.MlaConfig.QLoraRank"/>.
    /// </summary>
    [Fact]
    public void DeepSeekV2_WithQLoraRank_PopulatesQFactorisationRank()
    {
        const string json = """
        {
            "architectures": ["DeepseekV2ForCausalLM"],
            "model_type": "deepseek_v2",
            "hidden_size": 5120,
            "num_hidden_layers": 60,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "intermediate_size": 12288,
            "vocab_size": 102400,
            "max_position_embeddings": 163840,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.DeepSeekV2, cfg.Architecture);
        Assert.NotNull(cfg.MlaConfig);
        Assert.Equal(1536, cfg.MlaConfig!.QLoraRank);
        Assert.Equal(192, cfg.MlaConfig.QkHeadDim);
    }

    /// <summary>
    /// DeepSeek-V3 detected by <c>architectures[0] = "DeepseekV3ForCausalLM"</c>
    /// and <c>model_type = "deepseek_v3"</c>. Verifies the V3 detection branch
    /// and MlaConfig population — MoE-specific assertions land with the MoE PR.
    /// </summary>
    [Fact]
    public void DeepSeekV3_DetectedByArchitectureName()
    {
        const string json = """
        {
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
            "hidden_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 4,
            "intermediate_size": 256, "vocab_size": 100,
            "max_position_embeddings": 128,
            "kv_lora_rank": 32, "q_lora_rank": 24,
            "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.DeepSeekV3, cfg.Architecture);
        Assert.Equal(AttentionType.MLA, cfg.AttentionType);
        Assert.NotNull(cfg.MlaConfig);
        Assert.Equal(32, cfg.MlaConfig!.KvLoraRank);
        Assert.Equal(24, cfg.MlaConfig.QLoraRank);
    }
}
