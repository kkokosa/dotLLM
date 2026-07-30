using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class LogitBiasStepTests
{
    [Fact]
    public void Apply_AddsBiasToConfiguredTokens()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        var step = new LogitBiasStep(new Dictionary<int, float>
        {
            [0] = 0.5f,
            [2] = -1.25f,
        });

        step.Apply(logits, default(SamplerContext));

        Assert.Equal(1.5f, logits[0], precision: 5);
        Assert.Equal(2.0f, logits[1], precision: 5);
        Assert.Equal(1.75f, logits[2], precision: 5);
    }

    [Fact]
    public void Apply_OutOfRangeTokenIds_AreIgnored()
    {
        float[] logits = [1.0f, 2.0f];
        var step = new LogitBiasStep(new Dictionary<int, float>
        {
            [-1] = 10f,
            [100] = -10f,
        });

        step.Apply(logits, default(SamplerContext));

        Assert.Equal(1.0f, logits[0]);
        Assert.Equal(2.0f, logits[1]);
    }

    [Fact]
    public void Apply_EmptyBiasMap_IsNoOp()
    {
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] original = [1.0f, 2.0f, 3.0f];
        var step = new LogitBiasStep(new Dictionary<int, float>());

        step.Apply(logits, default(SamplerContext));

        Assert.Equal(original, logits);
    }
}
