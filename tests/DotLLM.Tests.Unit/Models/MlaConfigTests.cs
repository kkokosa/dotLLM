using DotLLM.Core.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

public sealed class MlaConfigTests
{
    private static MlaConfig BaseConfig() => new()
    {
        KvLoraRank = 512,
        QkNopeHeadDim = 128,
        QkRopeHeadDim = 64,
        VHeadDim = 128,
    };

    [Fact]
    public void ComputeYarnSoftmaxScaleMultiplier_NoYarnFields_ReturnsOne()
    {
        Assert.Equal(1.0f, BaseConfig().ComputeYarnSoftmaxScaleMultiplier());
    }

    [Fact]
    public void ComputeYarnSoftmaxScaleMultiplier_FactorLessOrEqualOne_ReturnsOne()
    {
        var cfg = BaseConfig() with
        {
            RopeScalingFactor = 1.0f,
            RopeScalingMscaleAllDim = 0.707f,
        };
        Assert.Equal(1.0f, cfg.ComputeYarnSoftmaxScaleMultiplier());
    }

    [Fact]
    public void ComputeYarnSoftmaxScaleMultiplier_ZeroMscaleAllDim_ReturnsOne()
    {
        var cfg = BaseConfig() with
        {
            RopeScalingFactor = 40.0f,
            RopeScalingMscaleAllDim = 0.0f,
        };
        Assert.Equal(1.0f, cfg.ComputeYarnSoftmaxScaleMultiplier());
    }

    [Fact]
    public void ComputeYarnSoftmaxScaleMultiplier_DeepSeekV2Lite_MatchesReferenceFormula()
    {
        // DeepSeek-V2-Lite config.json: rope_scaling.factor=40, mscale_all_dim=0.707.
        // Reference (HF modeling_deepseek.yarn_get_mscale):
        //   mscale = 0.1 * 0.707 * log(40) + 1.0
        //          = 0.1 * 0.707 * 3.688879 + 1.0
        //          ~= 1.260844
        //   result = mscale * mscale
        //          ~= 1.58973
        var cfg = BaseConfig() with
        {
            RopeScalingFactor = 40.0f,
            RopeScalingMscaleAllDim = 0.707f,
        };

        float expectedMscale = 0.1f * 0.707f * MathF.Log(40.0f) + 1.0f;
        float expected = expectedMscale * expectedMscale;

        float actual = cfg.ComputeYarnSoftmaxScaleMultiplier();
        Assert.Equal(expected, actual, precision: 5);
        Assert.InRange(actual, 1.58f, 1.60f);
    }

    [Fact]
    public void ComputeYarnSoftmaxScaleMultiplier_UsesMscaleAllDim_NotMscale()
    {
        // The softmax correction uses mscale_all_dim (not mscale). If we set
        // only mscale (not mscale_all_dim), the multiplier stays 1.0f.
        var cfg = BaseConfig() with
        {
            RopeScalingFactor = 40.0f,
            RopeScalingMscale = 0.707f,
            // RopeScalingMscaleAllDim is null
        };
        Assert.Equal(1.0f, cfg.ComputeYarnSoftmaxScaleMultiplier());
    }
}
