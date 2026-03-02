using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed class RmsNormTests
{
    [Fact]
    public void AllOnes_WeightOnes_ProducesOnes()
    {
        // input = [1,1,1,1], weight = [1,1,1,1], epsilon = 0
        // sumSq = 4, rms = sqrt(4/4 + 0) = 1, result = [1,1,1,1]
        float[] input = [1f, 1f, 1f, 1f];
        float[] weight = [1f, 1f, 1f, 1f];
        float[] result = new float[4];

        RmsNorm.Execute(input, weight, 0f, result);

        for (int i = 0; i < 4; i++)
            Assert.Equal(1f, result[i], 1e-5f);
    }

    [Fact]
    public void KnownValues_HandCalculated()
    {
        // input = [3, 4], weight = [1, 1], epsilon = 0
        // sumSq = 9 + 16 = 25, rms = sqrt(25/2) = sqrt(12.5) ≈ 3.5355
        // result = [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
        float[] input = [3f, 4f];
        float[] weight = [1f, 1f];
        float[] result = new float[2];

        RmsNorm.Execute(input, weight, 0f, result);

        Assert.Equal(3f / MathF.Sqrt(12.5f), result[0], 1e-4f);
        Assert.Equal(4f / MathF.Sqrt(12.5f), result[1], 1e-4f);
    }

    [Fact]
    public void EpsilonPreventsDiv0_AllZeros()
    {
        float[] input = [0f, 0f, 0f, 0f];
        float[] weight = [1f, 1f, 1f, 1f];
        float[] result = new float[4];

        RmsNorm.Execute(input, weight, 1e-5f, result);

        // With all-zero input and epsilon, result should still be all zeros (0 / rms = 0).
        for (int i = 0; i < 4; i++)
            Assert.Equal(0f, result[i], 1e-5f);
    }

    [Fact]
    public void WeightScaling_Applied()
    {
        float[] input = [1f, 1f, 1f, 1f];
        float[] weight = [2f, 3f, 4f, 5f];
        float[] result = new float[4];

        RmsNorm.Execute(input, weight, 0f, result);

        // rms = 1, so result[i] = 1 * weight[i]
        for (int i = 0; i < 4; i++)
            Assert.Equal(weight[i], result[i], 1e-5f);
    }

    [Fact]
    public void ScalarMatchesTensorPrimitives()
    {
        var rng = new Random(42);
        const int n = 1024;
        float[] input = new float[n];
        float[] weight = new float[n];
        for (int i = 0; i < n; i++)
        {
            input[i] = rng.NextSingle() * 2f - 1f;
            weight[i] = rng.NextSingle() * 2f;
        }

        float[] scalarResult = new float[n];
        float[] simdResult = new float[n];

        RmsNorm.ExecuteScalar(input, weight, 1e-5f, scalarResult);
        RmsNorm.Execute(input, weight, 1e-5f, simdResult);

        for (int i = 0; i < n; i++)
            Assert.Equal(scalarResult[i], simdResult[i], 1e-4f);
    }

    [Fact]
    public void NegativeInput_Correct()
    {
        float[] input = [-2f, -2f];
        float[] weight = [1f, 1f];
        float[] result = new float[2];

        RmsNorm.Execute(input, weight, 0f, result);

        // sumSq = 8, rms = sqrt(8/2) = 2, result = [-1, -1]
        Assert.Equal(-1f, result[0], 1e-5f);
        Assert.Equal(-1f, result[1], 1e-5f);
    }
}
