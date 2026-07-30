using DotLLM.Cpu.Kernels;
using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

/// <summary>
/// Tests for <see cref="SoftmaxF16"/>.
///
/// CPU reference: scalar FP32 softmax (numerically stable).
/// Tolerance: 1e-2f — two sources of error vs FP32 reference:
///   1. float→half truncation on input and output (~0.1 % relative).
///   2. Different summation order (parallel GPU vs sequential CPU).
/// </summary>
public sealed class SoftmaxF16Tests
{
    private const float Tol = 1e-2f;

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static Half[] H(params float[] values) =>
        Array.ConvertAll(values, v => (Half)v);

    /// <summary>
    /// CPU reference: converts Half→float (lossless), then calls
    /// <see cref="Softmax.Execute"/> row by row (TensorPrimitives, numerically stable).
    /// </summary>
    private static float[] CpuRef(Half[] input, int rows, int cols)
    {
        float[] inputF = Array.ConvertAll(input, h => (float)h);
        float[] result = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            Softmax.Execute(inputF.AsSpan(r * cols, cols), result.AsSpan(r * cols, cols));
        return result;
    }

    private static void AssertClose(float[] expected, Half[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], (float)actual[i], Tol);
    }

    // ── Single-row tests ──────────────────────────────────────────────────────

    [MetalTestFact]
    public void SingleRow_Uniform_ProducesEqual()
    {
        Half[]  input  = H(1f, 1f, 1f, 1f);
        Half[]  output = new Half[4];
        float[] cpu    = CpuRef(input, rows: 1, cols: 4);

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output, rows: 1, cols: 4);

        AssertClose(cpu, output);
    }

    [MetalTestFact]
    public void SingleRow_SumsToOne()
    {
        Half[] input  = H(1f, 2f, 3f, 4f, 5f);
        Half[] output = new Half[5];

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output, rows: 1, cols: 5);

        float sum = 0f;
        foreach (var v in output) sum += (float)v;
        Assert.Equal(1f, sum, Tol);
    }

    [MetalTestFact]
    public void SingleRow_LargeValues_NumericallyStable()
    {
        // Values near FP16 max — subtracting max prevents overflow
        Half[] input  = H(100f, 101f, 102f);
        Half[] output = new Half[3];

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output, rows: 1, cols: 3);

        Assert.All(output, v => Assert.True(float.IsFinite((float)v)));
        float sum = 0f;
        foreach (var v in output) sum += (float)v;
        Assert.Equal(1f, sum, Tol);
        // Largest input → largest probability
        Assert.True((float)output[2] > (float)output[1]);
        Assert.True((float)output[1] > (float)output[0]);
    }

    [MetalTestFact]
    public void SingleRow_SingleElement_ProducesOne()
    {
        Half[] input  = H(42f);
        Half[] output = new Half[1];

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output, rows: 1, cols: 1);

        Assert.Equal((Half)1f, output[0]);
    }

    [MetalTestFact]
    public void SingleRow_AllNegative_StillValid()
    {
        // Values must stay within FP16 range after exp(x - max):
        // exp(-3 - (-1)) = exp(-2) ≈ 0.135 — well above FP16 subnormal minimum (~6e-8).
        // H(-10, -20, -30) would give exp(-20) ≈ 2e-9 which underflows to 0 in FP16.
        Half[] input  = H(-1f, -2f, -3f);
        Half[] output = new Half[3];

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output, rows: 1, cols: 3);

        Assert.All(output, v => Assert.True((float)v > 0f));
        float sum = 0f;
        foreach (var v in output) sum += (float)v;
        Assert.Equal(1f, sum, Tol);
    }

    [MetalTestFact]
    public void SingleRow_VectorOverload_MatchesRowCols()
    {
        // The convenience overload must produce the same result as rows=1, cols=n
        Half[] input   = H(0.5f, 1.5f, 2.5f, 3.5f);
        Half[] output1 = new Half[4];
        Half[] output2 = new Half[4];

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output1);
        SoftmaxF16.Execute(ctx, input, output2, rows: 1, cols: 4);

        Assert.Equal(output1, output2);
    }

    [MetalTestFact]
    public void SingleRow_MatchesCpuReference()
    {
        var rng = new Random(7);
        const int cols = 256;
        Half[] input  = new Half[cols];
        Half[] output = new Half[cols];
        for (int i = 0; i < cols; i++) input[i] = (Half)(rng.NextSingle() * 8f - 4f);

        float[] cpu = CpuRef(input, rows: 1, cols: cols);

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output, rows: 1, cols: cols);

        AssertClose(cpu, output);
    }

    // ── Multi-row (batch) tests ───────────────────────────────────────────────

    [MetalTestFact]
    public void MultiRow_EachRowSumsToOne()
    {
        // 3 tokens, 4 classes each
        Half[] input  = H(1f,2f,3f,4f,   5f,5f,5f,5f,   0f,-1f,-2f,-3f);
        Half[] output = new Half[12];

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output, rows: 3, cols: 4);

        for (int r = 0; r < 3; r++)
        {
            float sum = 0f;
            for (int i = 0; i < 4; i++) sum += (float)output[r * 4 + i];
            Assert.Equal(1f, sum, Tol);
        }
    }

    [MetalTestFact]
    public void MultiRow_UniformRow_AllEqual()
    {
        // Row 1 is all-same — should produce 0.25 each
        Half[] input  = H(1f,2f,3f,4f,   2f,2f,2f,2f);
        Half[] output = new Half[8];

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output, rows: 2, cols: 4);

        for (int i = 0; i < 4; i++)
            Assert.Equal(0.25f, (float)output[4 + i], Tol);
    }

    [MetalTestFact]
    public void MultiRow_MatchesCpuReference()
    {
        // Simulates a typical decode step: 8 tokens × 512 vocab
        var rng = new Random(42);
        const int rows = 8;
        const int cols = 512;
        Half[] input  = new Half[rows * cols];
        Half[] output = new Half[rows * cols];
        for (int i = 0; i < input.Length; i++) input[i] = (Half)(rng.NextSingle() * 6f - 3f);

        float[] cpu = CpuRef(input, rows, cols);

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output, rows, cols);

        AssertClose(cpu, output);
    }

    [MetalTestFact]
    public void LargeVocab_StressesColumnLoop_MatchesCpu()
    {
        // cols > 256 (tiles the inner loop multiple times per thread)
        var rng = new Random(13);
        const int rows = 4;
        const int cols = 1024;
        Half[] input  = new Half[rows * cols];
        Half[] output = new Half[rows * cols];
        for (int i = 0; i < input.Length; i++) input[i] = (Half)(rng.NextSingle() * 4f - 2f);

        float[] cpu = CpuRef(input, rows, cols);

        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, input, output, rows, cols);

        AssertClose(cpu, output);
    }

    // ── Validation ───────────────────────────────────────────────────────────

    [MetalTestFact]
    public void InputLengthMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            SoftmaxF16.Execute(ctx, new Half[6], new Half[8], rows: 2, cols: 4));
    }

    [MetalTestFact]
    public void OutputTooSmall_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        Assert.Throws<ArgumentException>(() =>
            SoftmaxF16.Execute(ctx, new Half[8], new Half[6], rows: 2, cols: 4));
    }

    [MetalTestFact]
    public void EmptyInput_ReturnsSilently()
    {
        using var ctx = new MetalContext();
        SoftmaxF16.Execute(ctx, [], [], rows: 0, cols: 0);
    }
}
