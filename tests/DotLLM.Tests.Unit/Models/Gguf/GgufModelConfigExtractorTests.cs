using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

public class GgufModelConfigExtractorTests
{
    private static GgufMetadata BuildMetadata(Action<GgufTestData> configure)
    {
        var data = new GgufTestData(version: 3);
        configure(data);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var raw = GgufReader.ReadMetadata(reader, header);
        return new GgufMetadata(raw);
    }

    private static GgufMetadata BuildLlamaMetadata(Action<GgufTestData>? extra = null)
    {
        return BuildMetadata(d =>
        {
            d.AddString("general.architecture", "llama");
            d.AddUInt32("llama.embedding_length", 4096);
            d.AddUInt32("llama.block_count", 32);
            d.AddUInt32("llama.feed_forward_length", 11008);
            d.AddUInt32("llama.attention.head_count", 32);
            d.AddUInt32("llama.attention.head_count_kv", 8);
            d.AddUInt32("llama.context_length", 4096);
            d.AddFloat32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
            d.AddUInt32("llama.vocab_size", 32000);
            d.AddFloat32("llama.rope.freq_base", 10000.0f);
            d.AddUInt32("llama.rope.dimension_count", 128);
            extra?.Invoke(d);
        });
    }

    [Fact]
    public void Extract_Llama_AllFields()
    {
        var metadata = BuildLlamaMetadata();
        var config = GgufModelConfigExtractor.Extract(metadata);

        Assert.Equal(Architecture.Llama, config.Architecture);
        Assert.Equal(4096, config.HiddenSize);
        Assert.Equal(32, config.NumLayers);
        Assert.Equal(11008, config.IntermediateSize);
        Assert.Equal(32, config.NumAttentionHeads);
        Assert.Equal(8, config.NumKvHeads);
        Assert.Equal(128, config.HeadDim); // 4096 / 32
        Assert.Equal(4096, config.MaxSequenceLength);
        Assert.Equal(1e-5f, config.NormEpsilon);
        Assert.Equal(32000, config.VocabSize);
    }

    [Fact]
    public void Extract_RoPEConfig_Populated()
    {
        var metadata = BuildLlamaMetadata();
        var config = GgufModelConfigExtractor.Extract(metadata);

        Assert.NotNull(config.RoPEConfig);
        Assert.Equal(10000.0f, config.RoPEConfig.Value.Theta);
        Assert.Equal(128, config.RoPEConfig.Value.DimensionCount);
        Assert.Equal(RoPEScalingType.None, config.RoPEConfig.Value.ScalingType);
        Assert.Equal(PositionEncodingType.RoPE, config.PositionEncodingType);
    }

    [Fact]
    public void Extract_RoPEScaling_YaRN()
    {
        var metadata = BuildLlamaMetadata(d =>
        {
            d.AddString("llama.rope.scaling.type", "yarn");
            d.AddFloat32("llama.rope.scaling.factor", 4.0f);
            d.AddUInt32("llama.rope.scaling.original_context_length", 8192);
            d.AddFloat32("llama.rope.scaling.attn_factor", 0.5f);
            d.AddFloat32("llama.rope.scaling.beta_fast", 64.0f);
            d.AddFloat32("llama.rope.scaling.beta_slow", 2.0f);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);

        Assert.NotNull(config.RoPEConfig);
        Assert.Equal(RoPEScalingType.YaRN, config.RoPEConfig.Value.ScalingType);
        Assert.Equal(4.0f, config.RoPEConfig.Value.ScalingFactor);
        Assert.Equal(8192, config.RoPEConfig.Value.OrigMaxSeqLen);
        Assert.Equal(0.5f, config.RoPEConfig.Value.AttnFactor);
        Assert.Equal(64.0f, config.RoPEConfig.Value.BetaFast);
        Assert.Equal(2.0f, config.RoPEConfig.Value.BetaSlow);
    }

    [Fact]
    public void Extract_KvHeads_DefaultsToAttentionHeads()
    {
        var metadata = BuildMetadata(d =>
        {
            d.AddString("general.architecture", "llama");
            d.AddUInt32("llama.embedding_length", 4096);
            d.AddUInt32("llama.block_count", 32);
            d.AddUInt32("llama.feed_forward_length", 11008);
            d.AddUInt32("llama.attention.head_count", 32);
            // No head_count_kv — should default to 32 (MHA)
            d.AddUInt32("llama.context_length", 4096);
            d.AddUInt32("llama.vocab_size", 32000);
            d.AddFloat32("llama.rope.freq_base", 10000.0f);
            d.AddUInt32("llama.rope.dimension_count", 128);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.Equal(32, config.NumKvHeads);
    }

    [Fact]
    public void Extract_VocabSize_FallbackToTokensArray()
    {
        var metadata = BuildMetadata(d =>
        {
            d.AddString("general.architecture", "llama");
            d.AddUInt32("llama.embedding_length", 4096);
            d.AddUInt32("llama.block_count", 32);
            d.AddUInt32("llama.feed_forward_length", 11008);
            d.AddUInt32("llama.attention.head_count", 32);
            d.AddUInt32("llama.context_length", 4096);
            d.AddFloat32("llama.rope.freq_base", 10000.0f);
            d.AddUInt32("llama.rope.dimension_count", 128);
            // No vocab_size, but tokens array present
            d.AddStringArray("tokenizer.ggml.tokens", ["<s>", "</s>", "hello"]);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.Equal(3, config.VocabSize);
    }

    [Fact]
    public void Extract_ChatTemplate_Extracted()
    {
        var metadata = BuildLlamaMetadata(d =>
            d.AddString("tokenizer.chat_template", "{% for msg in messages %}...{% endfor %}"));

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.NotNull(config.ChatTemplate);
        Assert.Contains("messages", config.ChatTemplate);
    }

    [Fact]
    public void Extract_ChatTemplate_NullWhenAbsent()
    {
        var metadata = BuildLlamaMetadata();
        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.Null(config.ChatTemplate);
    }

    [Fact]
    public void Extract_UnsupportedArchitecture_Throws()
    {
        var metadata = BuildMetadata(d => d.AddString("general.architecture", "unknown_arch"));

        var ex = Assert.Throws<InvalidDataException>(() => GgufModelConfigExtractor.Extract(metadata));
        Assert.Contains("unknown_arch", ex.Message);
    }

    [Fact]
    public void Extract_NoRoPEKeys_ReturnsNullRoPEConfig()
    {
        var metadata = BuildMetadata(d =>
        {
            d.AddString("general.architecture", "llama");
            d.AddUInt32("llama.embedding_length", 4096);
            d.AddUInt32("llama.block_count", 32);
            d.AddUInt32("llama.feed_forward_length", 11008);
            d.AddUInt32("llama.attention.head_count", 32);
            d.AddUInt32("llama.context_length", 4096);
            d.AddUInt32("llama.vocab_size", 32000);
            // No rope keys
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.Null(config.RoPEConfig);
        Assert.Equal(PositionEncodingType.None, config.PositionEncodingType);
    }

    [Fact]
    public void Extract_ContextLength_DefaultsTo2048()
    {
        var metadata = BuildMetadata(d =>
        {
            d.AddString("general.architecture", "llama");
            d.AddUInt32("llama.embedding_length", 4096);
            d.AddUInt32("llama.block_count", 32);
            d.AddUInt32("llama.feed_forward_length", 11008);
            d.AddUInt32("llama.attention.head_count", 32);
            d.AddUInt32("llama.vocab_size", 32000);
            // No context_length
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.Equal(2048, config.MaxSequenceLength);
    }

    [Theory]
    [InlineData("mistral", Architecture.Mistral)]
    [InlineData("phi", Architecture.Phi)]
    [InlineData("qwen", Architecture.Qwen)]
    [InlineData("qwen2", Architecture.Qwen)]
    [InlineData("qwen3", Architecture.Qwen)]
    [InlineData("deepseek", Architecture.DeepSeek)]
    [InlineData("deepseek2", Architecture.DeepSeek)]
    public void Extract_ArchitectureParsing(string archString, Architecture expected)
    {
        var metadata = BuildMetadata(d =>
        {
            d.AddString("general.architecture", archString);
            string arch = archString.ToLowerInvariant();
            d.AddUInt32($"{arch}.embedding_length", 4096);
            d.AddUInt32($"{arch}.block_count", 32);
            d.AddUInt32($"{arch}.feed_forward_length", 11008);
            d.AddUInt32($"{arch}.attention.head_count", 32);
            d.AddUInt32($"{arch}.context_length", 4096);
            d.AddUInt32($"{arch}.vocab_size", 32000);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.Equal(expected, config.Architecture);
    }

    [Fact]
    public void Extract_SlidingWindow_Extracted()
    {
        var metadata = BuildLlamaMetadata(d =>
            d.AddUInt32("llama.attention.sliding_window", 4096));

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.NotNull(config.SlidingWindowSize);
        Assert.Equal(4096, config.SlidingWindowSize!.Value);
    }

    [Fact]
    public void Extract_SlidingWindow_NullWhenAbsent()
    {
        var metadata = BuildLlamaMetadata();
        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.Null(config.SlidingWindowSize);
    }

    [Fact]
    public void Extract_SlidingWindow_NullWhenZero()
    {
        var metadata = BuildLlamaMetadata(d =>
            d.AddUInt32("llama.attention.sliding_window", 0));

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.Null(config.SlidingWindowSize);
    }

    [Fact]
    public void Extract_RoPEScaling_Su()
    {
        var metadata = BuildLlamaMetadata(d =>
        {
            d.AddString("llama.rope.scaling.type", "su");
            d.AddFloat32("llama.rope.scaling.factor", 2.0f);
            d.AddUInt32("llama.rope.scaling.original_context_length", 4096);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.NotNull(config.RoPEConfig);
        Assert.Equal(RoPEScalingType.Su, config.RoPEConfig.Value.ScalingType);
        Assert.Equal(2.0f, config.RoPEConfig.Value.ScalingFactor);
        Assert.Equal(4096, config.RoPEConfig.Value.OrigMaxSeqLen);
    }

    [Fact]
    public void Extract_RoPEScaling_LongRope()
    {
        var metadata = BuildLlamaMetadata(d =>
        {
            d.AddString("llama.rope.scaling.type", "longrope");
            d.AddFloat32("llama.rope.scaling.factor", 8.0f);
            d.AddUInt32("llama.rope.scaling.original_context_length", 8192);
            d.AddFloat32("llama.rope.scaling.attn_factor", 1.5f);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.NotNull(config.RoPEConfig);
        Assert.Equal(RoPEScalingType.Su, config.RoPEConfig.Value.ScalingType);
        Assert.Equal(8.0f, config.RoPEConfig.Value.ScalingFactor);
        Assert.Equal(8192, config.RoPEConfig.Value.OrigMaxSeqLen);
        Assert.Equal(1.5f, config.RoPEConfig.Value.AttnFactor);
    }

    [Fact]
    public void Extract_RoPEScaling_Linear()
    {
        var metadata = BuildLlamaMetadata(d =>
        {
            d.AddString("llama.rope.scaling.type", "linear");
            d.AddFloat32("llama.rope.scaling.factor", 4.0f);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.NotNull(config.RoPEConfig);
        Assert.Equal(RoPEScalingType.Linear, config.RoPEConfig.Value.ScalingType);
        Assert.Equal(4.0f, config.RoPEConfig.Value.ScalingFactor);
    }

    [Fact]
    public void Extract_RoPEScaling_DynamicNtk()
    {
        var metadata = BuildLlamaMetadata(d =>
        {
            d.AddString("llama.rope.scaling.type", "dynamic_ntk");
            d.AddFloat32("llama.rope.scaling.factor", 2.0f);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.NotNull(config.RoPEConfig);
        Assert.Equal(RoPEScalingType.DynamicNTK, config.RoPEConfig.Value.ScalingType);
    }

    [Fact]
    public void Extract_RoPEScaling_Dynamic()
    {
        var metadata = BuildLlamaMetadata(d =>
        {
            d.AddString("llama.rope.scaling.type", "dynamic");
            d.AddFloat32("llama.rope.scaling.factor", 2.0f);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.NotNull(config.RoPEConfig);
        Assert.Equal(RoPEScalingType.DynamicNTK, config.RoPEConfig.Value.ScalingType);
    }

    [Fact]
    public void Extract_RoPEScaling_NTK()
    {
        var metadata = BuildLlamaMetadata(d =>
        {
            d.AddString("llama.rope.scaling.type", "ntk");
            d.AddFloat32("llama.rope.scaling.factor", 2.0f);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.NotNull(config.RoPEConfig);
        Assert.Equal(RoPEScalingType.NTK, config.RoPEConfig.Value.ScalingType);
    }

    [Fact]
    public void Extract_RoPEScaling_UnknownType_FallsBackToNone()
    {
        var metadata = BuildLlamaMetadata(d =>
        {
            d.AddString("llama.rope.scaling.type", "unknown_scaling");
            d.AddFloat32("llama.rope.scaling.factor", 2.0f);
        });

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.NotNull(config.RoPEConfig);
        Assert.Equal(RoPEScalingType.None, config.RoPEConfig.Value.ScalingType);
    }

    [Fact]
    public void Extract_RoPEScaling_YaRN_AllDefaults()
    {
        var metadata = BuildLlamaMetadata(d =>
            d.AddString("llama.rope.scaling.type", "yarn"));

        var config = GgufModelConfigExtractor.Extract(metadata);
        Assert.NotNull(config.RoPEConfig);
        Assert.Equal(RoPEScalingType.YaRN, config.RoPEConfig.Value.ScalingType);
        // Verify defaults when optional keys are absent
        Assert.Equal(1.0f, config.RoPEConfig.Value.ScalingFactor);
        Assert.Equal(0, config.RoPEConfig.Value.OrigMaxSeqLen);
        Assert.Equal(1.0f, config.RoPEConfig.Value.AttnFactor);
        Assert.Equal(32.0f, config.RoPEConfig.Value.BetaFast);
        Assert.Equal(1.0f, config.RoPEConfig.Value.BetaSlow);
    }
}
