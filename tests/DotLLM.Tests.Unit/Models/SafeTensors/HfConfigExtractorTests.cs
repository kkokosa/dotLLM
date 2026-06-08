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

    /// <summary>
    /// Mixtral config is detected from <c>architectures[0] = "MixtralForCausalLM"</c>
    /// AND populates <see cref="Core.Models.MoeConfig"/> from
    /// <c>num_local_experts</c> / <c>num_experts_per_tok</c>. Copy of the
    /// <c>yujiepan/mixtral-tiny-random</c> config (2026-04).
    /// </summary>
    [Fact]
    public void Mixtral_TinyRandom_PopulatesMoeConfig()
    {
        const string json = """
        {
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "hidden_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-5,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "tie_word_embeddings": false,
            "sliding_window": null
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.Mixtral, cfg.Architecture);
        Assert.NotNull(cfg.Moe);
        Assert.Equal(8, cfg.Moe!.NumExperts);
        Assert.Equal(2, cfg.Moe.NumExpertsPerTok);
        Assert.Equal(8, cfg.Moe.MoeIntermediateSize); // defaults to intermediate_size
        // Attention path stays GQA/RoPE — nothing Mixtral-specific there.
        Assert.Equal(4, cfg.NumAttentionHeads);
        Assert.Equal(2, cfg.NumKvHeads);
        Assert.Equal(1, cfg.HeadDim); // 4 / 4
        Assert.Equal(RoPEType.Norm, cfg.RoPEConfig!.Value.Type);
    }

    /// <summary>
    /// When <c>moe_intermediate_size</c> is declared explicitly (Phi-3.5-MoE
    /// convention), <see cref="Core.Models.MoeConfig.MoeIntermediateSize"/>
    /// should reflect that value, not the top-level <c>intermediate_size</c>.
    /// </summary>
    [Fact]
    public void Mixtral_OverrideMoeIntermediateSize_UsedOverTopLevel()
    {
        const string json = """
        {
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 4,
            "num_key_value_heads": 4, "intermediate_size": 64, "moe_intermediate_size": 32,
            "vocab_size": 100, "max_position_embeddings": 128,
            "num_local_experts": 4, "num_experts_per_tok": 2
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.NotNull(cfg.Moe);
        Assert.Equal(32, cfg.Moe!.MoeIntermediateSize);
        Assert.Equal(64, cfg.IntermediateSize);
    }

    /// <summary>
    /// Non-MoE configs must leave <c>ModelConfig.Moe</c> null — the dense
    /// FFN path keys off that.
    /// </summary>
    [Fact]
    public void DenseLlama_NoMoeConfig()
    {
        const string json = """
        {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
         "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
         "intermediate_size": 128, "vocab_size": 100, "max_position_embeddings": 128}
        """;
        var cfg = HfConfigExtractor.Extract(json);
        Assert.Null(cfg.Moe);
    }

    /// <summary>
    /// Declaring experts without a top-k should throw — misconfigured MoE is
    /// never silently ignored.
    /// </summary>
    [Fact]
    public void Mixtral_MissingNumExpertsPerTok_Throws()
    {
        const string json = """
        {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
         "hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 4,
         "intermediate_size": 8, "vocab_size": 100, "max_position_embeddings": 128,
         "num_local_experts": 8}
        """;
        var ex = Assert.Throws<InvalidDataException>(() => HfConfigExtractor.Extract(json));
        Assert.Contains("num_experts_per_tok", ex.Message);
    }

    /// <summary>
    /// Qwen3-MoE detection path — copy of the real
    /// <c>yujiepan/qwen3-moe-tiny-random</c> config (2026-04). Must resolve
    /// to <see cref="Architecture.QwenMoe"/>, populate MoE fields, use NeoX
    /// RoPE (Qwen family), leave shared-expert fields null, and carry the
    /// <c>decoder_sparse_step=2</c> layer-level sparsity across.
    /// </summary>
    [Fact]
    public void Qwen3Moe_TinyRandom_PopulatesMoeConfig_NoSharedExpert()
    {
        const string json = """
        {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "intermediate_size": 128,
            "moe_intermediate_size": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 40960,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "decoder_sparse_step": 2,
            "mlp_only_layers": [],
            "tie_word_embeddings": true
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.QwenMoe, cfg.Architecture);
        Assert.Equal(RoPEType.NeoX, cfg.RoPEConfig!.Value.Type);
        Assert.NotNull(cfg.Moe);
        Assert.Equal(8, cfg.Moe!.NumExperts);
        Assert.Equal(2, cfg.Moe.NumExpertsPerTok);
        Assert.Equal(128, cfg.Moe.MoeIntermediateSize);
        Assert.True(cfg.Moe.NormTopKProb);
        Assert.Null(cfg.Moe.SharedExpertIntermediateSize);
        Assert.False(cfg.Moe.HasSharedExpertGate);
        Assert.Equal(2, cfg.Moe.DecoderSparseStep);
        // decoder_sparse_step=2 ⇒ layer 0 is dense, layer 1 is MoE.
        Assert.False(cfg.Moe.IsMoeLayer(0));
        Assert.True(cfg.Moe.IsMoeLayer(1));
    }

    /// <summary>
    /// Qwen1.5-MoE-A2.7B config (2026-04) — has a shared expert with sigmoid
    /// gate and <c>norm_topk_prob=false</c>. Must surface all three via the
    /// extracted <see cref="Core.Models.MoeConfig"/>.
    /// </summary>
    [Fact]
    public void Qwen15Moe_A27B_PopulatesSharedExpertAndRawTopKProb()
    {
        const string json = """
        {
            "architectures": ["Qwen2MoeForCausalLM"],
            "model_type": "qwen2_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "intermediate_size": 5632,
            "moe_intermediate_size": 1408,
            "shared_expert_intermediate_size": 5632,
            "vocab_size": 151936,
            "max_position_embeddings": 8192,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "num_experts": 60,
            "num_experts_per_tok": 4,
            "norm_topk_prob": false,
            "decoder_sparse_step": 1,
            "tie_word_embeddings": false
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.Equal(Architecture.QwenMoe, cfg.Architecture);
        Assert.NotNull(cfg.Moe);
        Assert.Equal(60, cfg.Moe!.NumExperts);
        Assert.Equal(4, cfg.Moe.NumExpertsPerTok);
        Assert.Equal(1408, cfg.Moe.MoeIntermediateSize);
        Assert.False(cfg.Moe.NormTopKProb);
        Assert.Equal(5632, cfg.Moe.SharedExpertIntermediateSize);
        Assert.True(cfg.Moe.HasSharedExpertGate);
        Assert.Equal(1, cfg.Moe.DecoderSparseStep);
        // decoder_sparse_step=1 ⇒ every layer is MoE.
        Assert.True(cfg.Moe.IsMoeLayer(0));
        Assert.True(cfg.Moe.IsMoeLayer(23));
    }

    /// <summary>
    /// Qwen-MoE <c>mlp_only_layers</c> override: forces listed layer indices
    /// to be dense MLPs even if the sparsity stride would otherwise mark
    /// them MoE.
    /// </summary>
    [Fact]
    public void QwenMoe_MlpOnlyLayersOverride_RespectedByIsMoeLayer()
    {
        const string json = """
        {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 32,
            "intermediate_size": 128, "vocab_size": 100,
            "max_position_embeddings": 128,
            "num_experts": 4, "num_experts_per_tok": 2,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [2]
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.NotNull(cfg.Moe);
        Assert.True(cfg.Moe!.IsMoeLayer(0));
        Assert.True(cfg.Moe.IsMoeLayer(1));
        Assert.False(cfg.Moe.IsMoeLayer(2));  // forced dense
        Assert.True(cfg.Moe.IsMoeLayer(3));
    }

    [Fact]
    public void DeepSeekStyleMoE_MultiSharedExpert_PopulatesNumSharedExperts()
    {
        // DeepSeek-V2/V3 MoE config: n_routed_experts + n_shared_experts +
        // moe_intermediate_size. We drive through a Llama-shaped attention
        // (the MLA attention path lands separately with the MLA chain); this
        // PR's contract is only that the MoE extractor maps n_shared_experts
        // into MoeConfig.NumSharedExperts, with SharedExpertIntermediateSize
        // remaining the per-shared-expert width (NOT the pre-folded total)
        // and HasSharedExpertGate disabled because DeepSeek does not gate.
        const string json = """
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "intermediate_size": 10944,
            "vocab_size": 102400,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "n_routed_experts": 64,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 1408,
            "n_shared_experts": 2,
            "norm_topk_prob": false
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);

        Assert.NotNull(cfg.Moe);
        Assert.Equal(64, cfg.Moe!.NumExperts);
        Assert.Equal(6, cfg.Moe.NumExpertsPerTok);
        Assert.Equal(1408, cfg.Moe.MoeIntermediateSize);
        Assert.False(cfg.Moe.NormTopKProb);
        // Each shared expert is moe_intermediate_size wide; count = n_shared_experts.
        Assert.Equal(1408, cfg.Moe.SharedExpertIntermediateSize);
        Assert.Equal(2, cfg.Moe.NumSharedExperts);
        Assert.False(cfg.Moe.HasSharedExpertGate); // DeepSeek does NOT gate
    }

    [Fact]
    public void QwenMoE_SingleSharedExpert_DefaultsNumSharedExpertsToOne()
    {
        // Qwen1.5-MoE convention: shared_expert_intermediate_size set,
        // n_shared_experts absent. Must default NumSharedExperts to 1 and
        // keep the sigmoid gate enabled.
        const string json = """
        {
            "architectures": ["Qwen2MoeForCausalLM"],
            "model_type": "qwen2_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "intermediate_size": 5632,
            "vocab_size": 151936,
            "max_position_embeddings": 8192,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "num_experts": 60,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 1408,
            "shared_expert_intermediate_size": 5632,
            "norm_topk_prob": false
        }
        """;

        var cfg = HfConfigExtractor.Extract(json);
        Assert.NotNull(cfg.Moe);
        Assert.Equal(5632, cfg.Moe!.SharedExpertIntermediateSize);
        Assert.Equal(1, cfg.Moe.NumSharedExperts);
        Assert.True(cfg.Moe.HasSharedExpertGate);
    }
}
