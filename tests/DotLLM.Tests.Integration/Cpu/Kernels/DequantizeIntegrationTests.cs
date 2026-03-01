using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Cpu.Kernels;

[Collection("SmallModel")]
public sealed class DequantizeIntegrationTests
{
    private readonly SmallModelFixture _fixture;

    public DequantizeIntegrationTests(SmallModelFixture fixture) => _fixture = fixture;

    [Fact]
    public void Q8_0_RealTensor_AllFiniteAndReasonable()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);

        // Find first Q8_0 tensor.
        var tensor = gguf.Tensors.First(t => t.QuantizationType == QuantizationType.Q8_0);
        long elementCount = tensor.Shape.ElementCount;
        nint tensorPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

        float[] dest = new float[elementCount];
        Dequantize.ToFloat32(tensorPtr, elementCount, QuantizationType.Q8_0, dest);

        // All values must be finite.
        Assert.All(dest, v => Assert.True(float.IsFinite(v), $"Non-finite value: {v}"));

        // Quantized model weights should be in a reasonable range (not absurdly large).
        float maxAbs = dest.Max(MathF.Abs);
        Assert.True(maxAbs < 1000f, $"Max absolute value {maxAbs} exceeds reasonable range for model weights");
        Assert.True(maxAbs > 0f, "All values are zero — dequantization likely failed");
    }

    [Fact]
    public void Q8_0_ScalarMatchesSimd_RealTensorData()
    {
        if (!Avx2.IsSupported)
            return; // Nothing to compare on non-AVX2 machines.

        using var gguf = GgufFile.Open(_fixture.FilePath);

        var tensor = gguf.Tensors.First(t => t.QuantizationType == QuantizationType.Q8_0);
        nint tensorPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

        // Compare first 1024 elements (32 blocks).
        const int testElements = 1024;
        long elementCount = Math.Min(tensor.Shape.ElementCount, testElements);

        float[] scalarDest = new float[elementCount];
        float[] avx2Dest = new float[elementCount];

        Dequantize.DequantizeQ8_0Scalar(tensorPtr, elementCount, scalarDest);
        Dequantize.DequantizeQ8_0Avx2(tensorPtr, elementCount, avx2Dest);

        for (int i = 0; i < elementCount; i++)
        {
            Assert.Equal(scalarDest[i], avx2Dest[i], 1e-5f);
        }
    }
}
