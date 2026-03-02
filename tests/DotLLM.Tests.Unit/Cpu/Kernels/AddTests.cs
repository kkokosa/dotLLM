using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed class AddTests
{
    [Fact]
    public void KnownValues_MatchExpected()
    {
        float[] a = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] b = [10.0f, 20.0f, 30.0f, 40.0f];
        float[] result = new float[4];

        Add.Execute(a, b, result);

        Assert.Equal([11.0f, 22.0f, 33.0f, 44.0f], result);
    }

    [Fact]
    public void Zeros_ProducesOther()
    {
        float[] a = [0f, 0f, 0f, 0f];
        float[] b = [1f, 2f, 3f, 4f];
        float[] result = new float[4];

        Add.Execute(a, b, result);

        Assert.Equal(b, result);
    }

    [Fact]
    public void NegativeValues_Correct()
    {
        float[] a = [-1f, -2f, 3f];
        float[] b = [1f, 2f, -3f];
        float[] result = new float[3];

        Add.Execute(a, b, result);

        Assert.Equal([0f, 0f, 0f], result);
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
            a[i] = i;
            b[i] = n - i;
        }

        Add.Execute(a, b, result);

        for (int i = 0; i < n; i++)
            Assert.Equal(n, result[i]);
    }
}
