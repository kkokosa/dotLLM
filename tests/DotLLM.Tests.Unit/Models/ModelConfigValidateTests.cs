using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// Regression tests for <see cref="ModelConfig.Validate"/>. Without these checks
/// internally inconsistent GGUF metadata (e.g. <c>NumKvHeads</c> that does not
/// divide <c>NumAttentionHeads</c>) silently passes load and corrupts later
/// kernels that trust the values. See upstream issue #107 item 10.
/// </summary>
public class ModelConfigValidateTests
{
    private static ModelConfig ValidBase() => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 32_000,
        HiddenSize = 4096,
        IntermediateSize = 11008,
        NumLayers = 32,
        NumAttentionHeads = 32,
        NumKvHeads = 8, // 32 / 8 = 4 query heads per KV head
        HeadDim = 128, // 4096 / 32 = 128
        MaxSequenceLength = 4096,
        PositionEncodingType = PositionEncodingType.RoPE,
        RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: 128),
    };

    [Fact]
    public void Validate_ValidConfig_DoesNotThrow()
    {
        ValidBase().Validate();
    }

    [Fact]
    public void Validate_NumKvHeadsDoesNotDivideNumAttentionHeads_Throws()
    {
        // 7 does not divide 32.
        var config = ValidBase() with { NumAttentionHeads = 32, NumKvHeads = 7 };
        var ex = Assert.Throws<InvalidModelConfigException>(config.Validate);
        Assert.Equal(nameof(ModelConfig.NumKvHeads), ex.FieldName);
        Assert.Contains("NumAttentionHeads", ex.Message);
    }

    [Fact]
    public void Validate_HeadDimMismatchesDerived_Throws()
    {
        // 4096 / 32 = 128, but we claim 64.
        var config = ValidBase() with { HeadDim = 64 };
        var ex = Assert.Throws<InvalidModelConfigException>(config.Validate);
        Assert.Equal(nameof(ModelConfig.HeadDim), ex.FieldName);
        Assert.Contains("HiddenSize", ex.Message);
    }

    [Fact]
    public void Validate_RoPEConfigMissingOnRoPEEncoding_Throws()
    {
        var config = ValidBase() with { PositionEncodingType = PositionEncodingType.RoPE, RoPEConfig = null };
        var ex = Assert.Throws<InvalidModelConfigException>(config.Validate);
        Assert.Equal(nameof(ModelConfig.RoPEConfig), ex.FieldName);
        Assert.Contains("required", ex.Message);
    }

    [Fact]
    public void Validate_RoPEConfigPresentOnNonRoPEEncoding_Throws()
    {
        var config = ValidBase() with
        {
            PositionEncodingType = PositionEncodingType.ALiBi,
            RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: 128),
        };
        var ex = Assert.Throws<InvalidModelConfigException>(config.Validate);
        Assert.Equal(nameof(ModelConfig.RoPEConfig), ex.FieldName);
        Assert.Contains("must be null", ex.Message);
    }

    [Theory]
    [InlineData(nameof(ModelConfig.VocabSize))]
    [InlineData(nameof(ModelConfig.HiddenSize))]
    [InlineData(nameof(ModelConfig.IntermediateSize))]
    [InlineData(nameof(ModelConfig.NumLayers))]
    [InlineData(nameof(ModelConfig.NumAttentionHeads))]
    [InlineData(nameof(ModelConfig.NumKvHeads))]
    [InlineData(nameof(ModelConfig.HeadDim))]
    [InlineData(nameof(ModelConfig.MaxSequenceLength))]
    public void Validate_NonPositiveScalar_Throws(string fieldName)
    {
        var config = fieldName switch
        {
            nameof(ModelConfig.VocabSize) => ValidBase() with { VocabSize = 0 },
            nameof(ModelConfig.HiddenSize) => ValidBase() with { HiddenSize = 0 },
            nameof(ModelConfig.IntermediateSize) => ValidBase() with { IntermediateSize = -1 },
            nameof(ModelConfig.NumLayers) => ValidBase() with { NumLayers = 0 },
            nameof(ModelConfig.NumAttentionHeads) => ValidBase() with { NumAttentionHeads = 0 },
            nameof(ModelConfig.NumKvHeads) => ValidBase() with { NumKvHeads = 0 },
            nameof(ModelConfig.HeadDim) => ValidBase() with { HeadDim = -8 },
            nameof(ModelConfig.MaxSequenceLength) => ValidBase() with { MaxSequenceLength = 0 },
            _ => throw new InvalidOperationException(
                $"Test data error: no invalid-config case defined for field '{fieldName}'."),
        };
        var ex = Assert.Throws<InvalidModelConfigException>(config.Validate);
        Assert.Equal(fieldName, ex.FieldName);
    }

    [Fact]
    public void Validate_MqaConfig_AcceptsNumKvHeadsOne()
    {
        // Mistral-style MQA: KvHeads = 1, dividing NumAttentionHeads = 32 cleanly.
        var config = ValidBase() with { NumKvHeads = 1 };
        config.Validate();
    }
}
