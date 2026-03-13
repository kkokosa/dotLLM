using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed class FastMathTests
{
    [Fact]
    public void FastExp_MatchesMathF_AttentionRange()
    {
        // Attention range: x in [-88, 0] (scores after max-subtraction are always <= 0)
        var rng = new Random(42);
        for (int i = 0; i < 10_000; i++)
        {
            float x = rng.NextSingle() * -88f; // [-88, 0]
            float expected = MathF.Exp(x);
            float actual = FastMath.FastExp(x);

            if (expected < 1e-30f) continue; // skip near-zero where relative error is meaningless

            float relError = MathF.Abs(actual - expected) / expected;
            Assert.True(relError < 0.05f,
                $"FastExp({x}) = {actual}, MathF.Exp = {expected}, relative error = {relError:P2}");
        }
    }

    [Fact]
    public void FastExp_MatchesMathF_FullRange()
    {
        // Full useful range: [-87, 88]
        var rng = new Random(42);
        for (int i = 0; i < 10_000; i++)
        {
            float x = rng.NextSingle() * 175f - 87f; // [-87, 88]
            float expected = MathF.Exp(x);
            float actual = FastMath.FastExp(x);

            if (expected < 1e-30f || expected > 1e+30f) continue;

            float relError = MathF.Abs(actual - expected) / expected;
            Assert.True(relError < 0.05f,
                $"FastExp({x}) = {actual}, MathF.Exp = {expected}, relative error = {relError:P2}");
        }
    }

    [Fact]
    public void FastExp_ZeroReturnsApproxOne()
    {
        float result = FastMath.FastExp(0f);
        Assert.True(MathF.Abs(result - 1.0f) < 0.05f,
            $"FastExp(0) = {result}, expected ~1.0");
    }

    [Fact]
    public void FastExp_LargeNegativeReturnsNearZero()
    {
        float result = FastMath.FastExp(-100f);
        Assert.True(result >= 0f && result < 1e-30f,
            $"FastExp(-100) = {result}, expected ~0");
    }

    [Fact]
    public void ExpSumAndStore_MatchesSeparatePasses()
    {
        // Compare fused ExpSumAndStore vs separate Add+Exp+Sum
        var rng = new Random(42);
        const int n = 1024;
        float[] input = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = rng.NextSingle() * 10f - 5f;

        float offset = -3.5f;

        // Fused path
        float[] fusedOutput = new float[n];
        float fusedSum = FastMath.ExpSumAndStore(input, fusedOutput, offset);

        // Separate path using MathF.Exp (reference)
        float[] refOutput = new float[n];
        float refSum = 0f;
        for (int i = 0; i < n; i++)
        {
            refOutput[i] = MathF.Exp(input[i] + offset);
            refSum += refOutput[i];
        }

        // Allow ~2% relative error from fast exp approximation
        float sumRelError = MathF.Abs(fusedSum - refSum) / refSum;
        Assert.True(sumRelError < 0.02f,
            $"Sum mismatch: fused={fusedSum}, reference={refSum}, relative error={sumRelError:P2}");

        for (int i = 0; i < n; i++)
        {
            if (refOutput[i] < 1e-30f) continue;
            float relError = MathF.Abs(fusedOutput[i] - refOutput[i]) / refOutput[i];
            Assert.True(relError < 0.04f,
                $"Element {i}: fused={fusedOutput[i]}, reference={refOutput[i]}, relative error={relError:P2}");
        }
    }

    [Fact]
    public void ExpSumAndStore_InPlace()
    {
        // Verify in-place operation (input aliasing output)
        float[] data = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] backup = (float[])data.Clone();

        float sum = FastMath.ExpSumAndStore(data, data, -4f);

        Assert.True(sum > 0f);
        for (int i = 0; i < data.Length; i++)
        {
            float expected = MathF.Exp(backup[i] - 4f);
            float relError = MathF.Abs(data[i] - expected) / expected;
            Assert.True(relError < 0.04f,
                $"In-place element {i}: actual={data[i]}, expected={expected}");
        }
    }

    [Fact]
    public void ExpSumAndStore_SmallLength()
    {
        // Test with lengths smaller than SIMD vector width
        for (int len = 1; len <= 7; len++)
        {
            float[] input = new float[len];
            float[] output = new float[len];
            for (int i = 0; i < len; i++)
                input[i] = -i * 0.5f;

            float sum = FastMath.ExpSumAndStore(input, output, 0f);

            float refSum = 0f;
            for (int i = 0; i < len; i++)
                refSum += MathF.Exp(input[i]);

            float relError = MathF.Abs(sum - refSum) / refSum;
            Assert.True(relError < 0.05f,
                $"Length {len}: sum={sum}, refSum={refSum}, relError={relError:P2}");
        }
    }

    [Fact]
    public void FastSoftmax_MatchesStandard()
    {
        var rng = new Random(42);
        const int n = 512;
        float[] input = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = rng.NextSingle() * 20f - 10f;

        float[] fastResult = new float[n];
        float[] stdResult = new float[n];

        FastMath.Softmax(input, fastResult);
        Softmax.Execute(input, stdResult);

        for (int i = 0; i < n; i++)
            Assert.True(MathF.Abs(fastResult[i] - stdResult[i]) < 5e-3f,
                $"Softmax mismatch at {i}: fast={fastResult[i]}, std={stdResult[i]}");
    }

    [Fact]
    public void FastSoftmax_SumsToOne()
    {
        float[] input = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f];
        float[] result = new float[10];

        FastMath.Softmax(input, result);

        float sum = 0f;
        for (int i = 0; i < result.Length; i++)
            sum += result[i];

        Assert.True(MathF.Abs(sum - 1.0f) < 1e-5f,
            $"Softmax sum = {sum}, expected 1.0");
    }

    [Fact]
    public void FastSoftmax_AllPositive()
    {
        float[] input = [-10f, -5f, 0f, 5f, 10f];
        float[] result = new float[5];

        FastMath.Softmax(input, result);

        Assert.All(result, v => Assert.True(v > 0f, $"Expected positive, got {v}"));
    }

    [Fact]
    public void FastSoftmax_PreservesOrdering()
    {
        // input: [1, 3, 2, 5, 4] → indices by value: 0<2<1<4<3
        float[] input = [1f, 3f, 2f, 5f, 4f];
        float[] result = new float[5];

        FastMath.Softmax(input, result);

        // Larger input should get larger probability
        Assert.True(result[3] > result[4]); // 5 > 4
        Assert.True(result[4] > result[1]); // 4 > 3
        Assert.True(result[1] > result[2]); // 3 > 2
        Assert.True(result[2] > result[0]); // 2 > 1
    }

    [Fact]
    public void ExecuteFast_MatchesFastMathSoftmax()
    {
        // Softmax.ExecuteFast should produce same results as FastMath.Softmax
        var rng = new Random(42);
        const int n = 256;
        float[] input = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = rng.NextSingle() * 20f - 10f;

        float[] fromSoftmax = new float[n];
        float[] fromFastMath = new float[n];

        Softmax.ExecuteFast(input, fromSoftmax);
        FastMath.Softmax(input, fromFastMath);

        for (int i = 0; i < n; i++)
            Assert.Equal(fromFastMath[i], fromSoftmax[i], 1e-7f);
    }
}
