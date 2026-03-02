using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Cpu.Kernels;

[Collection("SmallModel")]
public sealed unsafe class MatMulIntegrationTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    private readonly SmallModelFixture _fixture;

    public MatMulIntegrationTests(SmallModelFixture fixture) => _fixture = fixture;

    [Fact]
    public void Q8_0Gemv_RealTensor_AllFinite()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);

        // Find a 2D Q8_0 tensor (weight matrix).
        var tensor = gguf.Tensors.First(t =>
            t.QuantizationType == QuantizationType.Q8_0 && t.Shape.Dimensions.Length == 2);

        int m = (int)tensor.Shape.Dimensions[0];
        int k = (int)tensor.Shape.Dimensions[1];
        byte* weightsPtr = (byte*)(gguf.DataBasePointer + (nint)tensor.DataOffset);

        // Use a small subset of rows to keep test fast.
        int testRows = Math.Min(m, 16);

        // Create random input vector.
        var rng = new Random(42);
        float[] x = new float[k];
        for (int i = 0; i < k; i++)
            x[i] = rng.NextSingle() * 2f - 1f;

        float[] result = new float[testRows];

        fixed (float* xp = x, rp = result)
            MatMul.GemvQ8_0(weightsPtr, xp, rp, testRows, k);

        for (int i = 0; i < testRows; i++)
            Assert.True(float.IsFinite(result[i]), $"result[{i}] = {result[i]}");
    }

    [Fact]
    public void Q8_0Gemv_CrossVerify_WithDequantizeF32Gemv()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);

        var tensor = gguf.Tensors.First(t =>
            t.QuantizationType == QuantizationType.Q8_0 && t.Shape.Dimensions.Length == 2);

        int m = (int)tensor.Shape.Dimensions[0];
        int k = (int)tensor.Shape.Dimensions[1];
        nint tensorDataPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

        // Use a small subset of rows for tractable dequantization.
        int testRows = Math.Min(m, 8);

        // Create input vector.
        var rng = new Random(42);
        float[] x = new float[k];
        for (int i = 0; i < k; i++)
            x[i] = rng.NextSingle() * 2f - 1f;

        // Method 1: Q8_0 GEMV (quantized path).
        float[] q8Result = new float[testRows];
        fixed (float* xp = x, rp = q8Result)
            MatMul.GemvQ8_0((byte*)tensorDataPtr, xp, rp, testRows, k);

        // Method 2: Dequantize to f32, then f32 GEMV.
        int blocksPerRow = k / Q8_0GroupSize;
        int rowQ8Bytes = blocksPerRow * Q8_0BlockBytes;

        float[] f32Weights = new float[testRows * k];
        for (int row = 0; row < testRows; row++)
        {
            nint rowPtr = tensorDataPtr + row * rowQ8Bytes;
            Dequantize.ToFloat32(rowPtr, k, QuantizationType.Q8_0,
                f32Weights.AsSpan(row * k, k));
        }

        float[] f32Result = new float[testRows];
        fixed (float* wp = f32Weights, xp = x, rp = f32Result)
            MatMul.GemvF32(wp, xp, rp, testRows, k);

        // Compare. Q8_0 GEMV introduces double quantization error (weights + activation),
        // so we allow ~2% relative error.
        for (int i = 0; i < testRows; i++)
        {
            float expected = f32Result[i];
            float actual = q8Result[i];

            if (MathF.Abs(expected) < 1e-3f)
            {
                Assert.True(MathF.Abs(actual) < 1f,
                    $"Row {i}: expected ≈ 0, got {actual}");
            }
            else
            {
                float relError = MathF.Abs(actual - expected) / MathF.Abs(expected);
                Assert.True(relError < 0.10f,
                    $"Row {i}: f32={expected}, q8={actual}, relError={relError:P1}");
            }
        }
    }
}
