using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Vulkan FP32 SwiGLU kernel.
/// </summary>
/// <remarks>
/// Pointwise — compared against <see cref="FusedOps.SwiGLUScalar"/>, the
/// scalar reference that calls <c>MathF.Exp</c> directly (so it doesn't hide
/// GPU drift behind <c>TensorPrimitives.Sigmoid</c>'s hardened impl).
/// Tolerance follows the mandate (rel 1e-3 / abs 1e-4).
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanSwiGluF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(255)]            // not a multiple of workgroup
    [InlineData(256)]            // exact workgroup
    [InlineData(1536)]           // SmolLM intermediate-size
    [InlineData(11008)]          // Llama-style intermediate
    public void Launch_MatchesCpuReference(int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x5716 + n);
        float[] gate = RandomFloats(rng, n, range: 3.0f);
        float[] up   = RandomFloats(rng, n, range: 3.0f);
        float[] expected = new float[n];
        FusedOps.SwiGLUScalar(gate, up, expected);

        using var device = VulkanDevice.Create();
        using var kernel = SwiGluF32Kernel.Create(device, spvDir);

        using var bufGate = device.Allocate((long)n * sizeof(float));
        using var bufUp   = device.Allocate((long)n * sizeof(float));
        using var bufOut  = device.Allocate((long)n * sizeof(float));

        device.Upload(gate.AsSpan(), bufGate);
        device.Upload(up.AsSpan(),   bufUp);

        kernel.Launch(bufGate, bufUp, bufOut, n);

        float[] actual = new float[n];
        device.Download(bufOut, actual);

        AssertClose(expected, actual, n);
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual, int n)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"SwiGLU drift exceeded tolerance (n={n}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
