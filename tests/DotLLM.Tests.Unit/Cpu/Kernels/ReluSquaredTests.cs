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

    [Fact]
    public void OversizedDestination_LeavesTailUntouched()
    {
        // The contract allows result to be longer than input (pooled scratch
        // buffers). Only the first input.Length elements may be written.
        float[] input = [-2f, 0.5f, 3f];
        float[] result = [9f, 9f, 9f, -7f, -7f];

        ReluSquared.Execute(input, result);

        Assert.Equal(0f, result[0], 1e-6f);
        Assert.Equal(0.25f, result[1], 1e-6f);
        Assert.Equal(9f, result[2], 1e-5f);
        Assert.Equal(-7f, result[3], 1e-6f);
        Assert.Equal(-7f, result[4], 1e-6f);
    }

    [Fact]
    public void AliasedBuffers_ProducesCorrectOutput()
    {
        // Verifies the kernel is correct when src and dest spans alias
        // (same buffer for both). Production callers in some MLP paths
        // alias to save allocations — the implementation must apply
        // max(x, 0) before the multiply, not after.
        float[] expected = [0f, 0f, 0f, 0.25f, 1f, 9f];
        float[] buffer = [-2f, -1f, 0f, 0.5f, 1f, 3f];

        ReluSquared.Execute(buffer, buffer); // src == dest

        for (int i = 0; i < buffer.Length; i++)
            Assert.Equal(expected[i], buffer[i], 1e-5f);
    }
}
