using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public sealed class AddF32Tests
{
    [MetalTestFact]
    public void KnownValues_MatchExpected()
    {
        using var ctx = new MetalContext();
        float[] a      = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] b      = [10.0f, 20.0f, 30.0f, 40.0f];
        float[] result = new float[4];

        AddF32.Execute(ctx, a, b, result);

        Assert.Equal([11.0f, 22.0f, 33.0f, 44.0f], result);
    }

    [MetalTestFact]
    public void EmptyArrays_ReturnsSilently()
    {
        using var ctx = new MetalContext();
        AddF32.Execute(ctx, [], [], []);
    }

    [MetalTestFact]
    public void InputLengthMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            AddF32.Execute(ctx, [1.0f, 2.0f], [1.0f], new float[2]));
    }

    [MetalTestFact]
    public void ResultTooSmall_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            AddF32.Execute(ctx, [1.0f, 2.0f], [3.0f, 4.0f], new float[1]));
    }

    [MetalTestFact]
    public void LargeVector_MatchExpected()
    {
        using var ctx = new MetalContext();
        const int size = 1024 * 1024;
        float[] a      = Enumerable.Repeat(1.0f, size).ToArray();
        float[] b      = Enumerable.Repeat(2.0f, size).ToArray();
        float[] result = new float[size];

        AddF32.Execute(ctx, a, b, result);

        Assert.All(result, x => Assert.Equal(3.0f, x));
    }
}

// ── AddF16 ────────────────────────────────────────────────────────────────────

public sealed class AddF16Tests
{
    private static Half[] H(params float[] values) =>
        Array.ConvertAll(values, v => (Half)v);

    [MetalTestFact]
    public void KnownValues_MatchExpected()
    {
        using var ctx  = new MetalContext();
        Half[]   a      = H(1.0f, 2.0f, 3.0f, 4.0f);
        Half[]   b      = H(10.0f, 20.0f, 30.0f, 40.0f);
        Half[]   result = new Half[4];

        AddF16.Execute(ctx, a, b, result);

        Half[] expected = H(11.0f, 22.0f, 33.0f, 44.0f);
        Assert.Equal(expected, result);
    }

    [MetalTestFact]
    public void OddLength_HandlesTrailingElement()
    {
        // n=5: half2 covers elements [0,1] and [2,3]; element 4 is the odd tail
        using var ctx  = new MetalContext();
        Half[]   a      = H(1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        Half[]   b      = H(10.0f, 20.0f, 30.0f, 40.0f, 50.0f);
        Half[]   result = new Half[5];

        AddF16.Execute(ctx, a, b, result);

        Half[] expected = H(11.0f, 22.0f, 33.0f, 44.0f, 55.0f);
        Assert.Equal(expected, result);
    }

    [MetalTestFact]
    public void EmptyArrays_ReturnsSilently()
    {
        using var ctx = new MetalContext();
        AddF16.Execute(ctx, [], [], []);
    }

    [MetalTestFact]
    public void InputLengthMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            AddF16.Execute(ctx, [H(1.0f)[0], H(2.0f)[0]], [H(1.0f)[0]], new Half[2]));
    }

    [MetalTestFact]
    public void LargeVector_EvenLength_MatchExpected()
    {
        using var ctx = new MetalContext();
        const int size = 1024;
        Half[]   a      = Array.ConvertAll(new float[size], _ => (Half)1.0f);
        Half[]   b      = Array.ConvertAll(new float[size], _ => (Half)2.0f);
        Half[]   result = new Half[size];

        AddF16.Execute(ctx, a, b, result);

        Assert.All(result, x => Assert.Equal((Half)3.0f, x));
    }
}

// ── AddF32F16 ─────────────────────────────────────────────────────────────────

public sealed class AddF32F16Tests
{
    [MetalTestFact]
    public void KnownValues_MatchExpected()
    {
        using var ctx  = new MetalContext();
        float[]  a      = [1.0f, 2.0f, 3.0f, 4.0f];
        Half[]   b      = [(Half)10.0f, (Half)20.0f, (Half)30.0f, (Half)40.0f];
        float[]  result = new float[4];

        AddF32F16.Execute(ctx, a, b, result);

        Assert.Equal([11.0f, 22.0f, 33.0f, 44.0f], result);
    }

    [MetalTestFact]
    public void FP16_PartialPrecision_IsAccounted()
    {
        // Values chosen so that Half conversion is exact (powers of 2 / small integers)
        using var ctx  = new MetalContext();
        float[]  a      = [0.5f, 1.0f, 2.0f, 4.0f];
        Half[]   b      = [(Half)0.25f, (Half)0.5f, (Half)0.75f, (Half)1.0f];
        float[]  result = new float[4];

        AddF32F16.Execute(ctx, a, b, result);

        float[] expected = [0.75f, 1.5f, 2.75f, 5.0f];
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], result[i], 1e-3f);
    }

    [MetalTestFact]
    public void EmptyArrays_ReturnsSilently()
    {
        using var ctx = new MetalContext();
        AddF32F16.Execute(ctx, [], [], []);
    }

    [MetalTestFact]
    public void InputLengthMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            AddF32F16.Execute(ctx, [1.0f, 2.0f], [(Half)1.0f], new float[2]));
    }

    [MetalTestFact]
    public void LargeVector_MatchExpected()
    {
        using var ctx = new MetalContext();
        const int size = 1024;
        float[]  a      = Enumerable.Repeat(1.5f, size).ToArray();
        Half[]   b      = Enumerable.Repeat((Half)0.5f, size).ToArray();
        float[]  result = new float[size];

        AddF32F16.Execute(ctx, a, b, result);

        Assert.All(result, x => Assert.Equal(2.0f, x, 1e-3f));
    }
}
