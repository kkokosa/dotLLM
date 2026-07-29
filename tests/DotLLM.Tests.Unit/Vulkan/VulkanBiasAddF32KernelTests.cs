using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanBiasAddF32KernelTests
{
    [SkippableTheory]
    [InlineData(1, 16)]
    [InlineData(1, 576)]                // SmolLM hidden
    [InlineData(8, 1024)]
    [InlineData(192, 576)]              // prefill-ish
    [InlineData(1, 4096)]               // Llama-2-7B hidden
    [InlineData(3, 257)]                // odd dims
    public void Launch_MatchesCpuReference(int seqLen, int outputDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBEEF + seqLen * 13 + outputDim);
        float[] output = RandomFloats(rng, seqLen * outputDim);
        float[] bias = RandomFloats(rng, outputDim);

        // Reference: in-place add on a copy.
        float[] expected = (float[])output.Clone();
        for (int t = 0; t < seqLen; t++)
            for (int i = 0; i < outputDim; i++)
                expected[t * outputDim + i] += bias[i];

        using var device = VulkanDevice.Create();
        using var kernel = BiasAddF32Kernel.Create(device, spvDir);

        using var bufOut = device.Allocate((long)output.Length * sizeof(float));
        using var bufBias = device.Allocate((long)bias.Length * sizeof(float));
        device.Upload(output, bufOut);
        device.Upload(bias, bufBias);

        kernel.Launch(bufOut, bufBias, seqLen, outputDim);

        var actual = new float[output.Length];
        device.Download(bufOut, actual);

        // Pure addition — bit-identical to CPU reference (no reduction).
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }
}
