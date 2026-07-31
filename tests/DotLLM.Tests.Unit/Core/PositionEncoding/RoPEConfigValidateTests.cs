using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using Xunit;

// Deliberately NOT `DotLLM.Tests.Unit.Core.PositionEncoding`, despite the folder path:
// introducing a `DotLLM.Tests.Unit.Core` namespace makes every fully-qualified
// `DotLLM.Core.*` reference elsewhere in this assembly bind to it instead of the real
// `DotLLM.Core` (C# name lookup walks enclosing namespaces first), breaking unrelated files.
namespace DotLLM.Tests.Unit.CorePositionEncoding;

/// <summary>
/// Regression tests for <see cref="RoPEConfig.Validate"/>. The YaRN-only fields
/// (<see cref="RoPEConfig.AttnFactor"/>, <see cref="RoPEConfig.BetaFast"/>,
/// <see cref="RoPEConfig.BetaSlow"/>) are silently ignored under non-YaRN scaling
/// strategies. Setting them while specifying NTK or Linear scaling is a config
/// mistake that previously passed validation and produced quietly wrong behaviour.
/// See upstream issue #107 item 11.
/// </summary>
public class RoPEConfigValidateTests
{
    [Fact]
    public void Validate_DefaultsAndNoneScaling_DoesNotThrow()
    {
        new RoPEConfig(DimensionCount: 128).Validate();
    }

    [Fact]
    public void Validate_YarnScalingWithCustomYarnFields_DoesNotThrow()
    {
        // Under YaRN scaling the YaRN fields are actively consumed — any value is fine.
        new RoPEConfig(
            DimensionCount: 128,
            ScalingType: RoPEScalingType.YaRN,
            AttnFactor: 0.5f,
            BetaFast: 64.0f,
            BetaSlow: 2.0f).Validate();
    }

    /// <summary>
    /// Headline discriminator: a non-default <see cref="RoPEConfig.BetaFast"/> under
    /// NTK scaling. Previously silently accepted; <see cref="RoPEConfig.Validate"/>
    /// must reject it.
    /// </summary>
    [Fact]
    public void Validate_NtkScalingWithCustomBetaFast_Throws()
    {
        var config = new RoPEConfig(
            DimensionCount: 128,
            ScalingType: RoPEScalingType.NTK,
            BetaFast: 64.0f); // non-default

        var ex = Assert.Throws<InvalidRoPEConfigException>(config.Validate);
        Assert.Equal(nameof(RoPEConfig.BetaFast), ex.FieldName);
        Assert.Contains("YaRN", ex.Message);
        Assert.Contains("NTK", ex.Message);
    }

    [Fact]
    public void Validate_LinearScalingWithCustomBetaSlow_Throws()
    {
        var config = new RoPEConfig(
            DimensionCount: 128,
            ScalingType: RoPEScalingType.Linear,
            BetaSlow: 2.0f); // non-default

        var ex = Assert.Throws<InvalidRoPEConfigException>(config.Validate);
        Assert.Equal(nameof(RoPEConfig.BetaSlow), ex.FieldName);
    }

    [Fact]
    public void Validate_NoneScalingWithCustomAttnFactor_Throws()
    {
        var config = new RoPEConfig(
            DimensionCount: 128,
            ScalingType: RoPEScalingType.None,
            AttnFactor: 0.7f); // non-default

        var ex = Assert.Throws<InvalidRoPEConfigException>(config.Validate);
        Assert.Equal(nameof(RoPEConfig.AttnFactor), ex.FieldName);
    }

    /// <summary>
    /// Non-YaRN, non-YaRN-fields configurations (e.g. setting just
    /// <see cref="RoPEConfig.ScalingFactor"/>) are valid.
    /// </summary>
    [Fact]
    public void Validate_NtkScalingWithOnlyScalingFactor_DoesNotThrow()
    {
        new RoPEConfig(
            DimensionCount: 128,
            ScalingType: RoPEScalingType.NTK,
            ScalingFactor: 4.0f).Validate();
    }
}
