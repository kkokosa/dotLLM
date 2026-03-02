using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed class MultiplyTests
{
    [Fact]
    public void KnownValues_MatchExpected()
    {
        float[] a = [2.0f, 3.0f, 4.0f, 5.0f];
        float[] b = [10.0f, 20.0f, 30.0f, 40.0f];
        float[] result = new float[4];

        Multiply.Execute(a, b, result);

        Assert.Equal([20.0f, 60.0f, 120.0f, 200.0f], result);
    }

    [Fact]
    public void Zeros_ProducesZeros()
    {
        float[] a = [0f, 0f, 0f, 0f];
        float[] b = [1f, 2f, 3f, 4f];
        float[] result = new float[4];

        Multiply.Execute(a, b, result);

        Assert.All(result, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void NegativeValues_Correct()
    {
        float[] a = [-1f, -2f, 3f];
        float[] b = [2f, -3f, -4f];
        float[] result = new float[3];

        Multiply.Execute(a, b, result);

        Assert.Equal([-2f, 6f, -12f], result);
    }

    [Fact]
    public void LargeSpan_Correct()
    {
        const int n = 4096;
        float[] a = new float[n];
        float[] b = new float[n];
        float[] result = new float[n];

        for (int i = 0; i < n; i++)
        {
            a[i] = 2.0f;
            b[i] = i;
        }

        Multiply.Execute(a, b, result);

        for (int i = 0; i < n; i++)
            Assert.Equal(2.0f * i, result[i]);
    }
}
