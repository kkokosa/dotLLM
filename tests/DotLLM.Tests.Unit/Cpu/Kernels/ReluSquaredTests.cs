using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed class ReluSquaredTests
{
    [Fact]
    public void Zero_ProducesZero()
    {
        float[] input = [0f];
        float[] result = new float[1];

        ReluSquared.Execute(input, result);

        Assert.Equal(0f, result[0], 1e-6f);
    }

    [Fact]
    public void NegativeInput_ProducesZero()
    {
        float[] input = [-3f];
        float[] result = new float[1];

        ReluSquared.Execute(input, result);

        Assert.Equal(0f, result[0], 1e-6f);
    }

    [Fact]
    public void PositiveInput_IsSquared()
    {
        // ReLU^2(2.5) = 6.25
        float[] input = [2.5f];
        float[] result = new float[1];

        ReluSquared.Execute(input, result);

        Assert.Equal(6.25f, result[0], 1e-5f);
    }

    [Fact]
    public void ScalarMatchesSimd()
    {
        var rng = new Random(17);
        const int n = 1024;
        float[] input = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = rng.NextSingle() * 20f - 10f; // [-10, 10]

        float[] scalar = new float[n];
        float[] simd = new float[n];

        ReluSquared.ExecuteScalar(input, scalar);
        ReluSquared.Execute(input, simd);

        for (int i = 0; i < n; i++)
            Assert.Equal(scalar[i], simd[i], 1e-5f);
    }

    [Fact]
    public void MixedValues_MatchExpected()
    {
        float[] input = [-2f, -1f, 0f, 0.5f, 1f, 3f];
        float[] expected = [0f, 0f, 0f, 0.25f, 1f, 9f];
        float[] result = new float[input.Length];

        ReluSquared.Execute(input, result);

        for (int i = 0; i < input.Length; i++)
            Assert.Equal(expected[i], result[i], 1e-5f);
    }
}
