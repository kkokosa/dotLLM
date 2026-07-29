using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan FP32 RMS-norm kernel.
/// </summary>
/// <remarks>
/// Compares against a scalar CPU reference. The GPU uses a workgroup tree
/// reduction vs. the CPU's sequential accumulation — small drift is
/// expected at larger N. Tolerance: rel 1e-3 / abs 1e-4 per mandate.
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanRmsNormF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;
    private const float DefaultEps = 1e-5f;

    [SkippableTheory]
    [InlineData(1, 16, 1e-6f)]            // tiny
    [InlineData(1, 576, 1e-5f)]           // SmolLM hidden-size, one row
    [InlineData(4, 576, 1e-5f)]           // small prefill
    [InlineData(16, 1536, 1e-6f)]         // intermediate-size, batch
    [InlineData(1, 257, 1e-5f)]           // non-power-of-two, not a multiple of workgroup
    public void Launch_MatchesCpuReference(int rowCount, int n, float eps)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xFEED + rowCount * 31 + n);
        float[] input = RandomFloats(rng, rowCount * n, range: 1.0f);
        float[] weight = RandomFloats(rng, n, range: 1.0f);
        // Shift weights away from zero — real RMS-norm weights are typically ~1.
        for (int i = 0; i < n; i++) weight[i] = weight[i] * 0.5f + 1.0f;

        float[] expected = new float[rowCount * n];
        CpuReference(input, weight, expected, rowCount, n, eps);

        using var device = VulkanDevice.Create();
        using var kernel = RmsNormF32Kernel.Create(device, spvDir);

        using var bufIn = device.Allocate((long)rowCount * n * sizeof(float));
        using var bufW  = device.Allocate((long)n * sizeof(float));
        using var bufOut = device.Allocate((long)rowCount * n * sizeof(float));

        device.Upload(input, bufIn);
        device.Upload(weight, bufW);
        kernel.Launch(bufIn, bufW, bufOut, rowCount, n, eps);

        float[] actual = new float[rowCount * n];
        device.Download(bufOut, actual);

        AssertClose(expected, actual, rowCount, n);
    }

    // ─────────────────────────────────────────────────────────────

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static void CpuReference(float[] input, float[] weight, float[] output,
                                     int rowCount, int n, float eps)
    {
        for (int r = 0; r < rowCount; r++)
        {
            int rowBase = r * n;
            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                float v = input[rowBase + i];
                sumSq += (double)v * v;
            }
            float rinv = 1.0f / MathF.Sqrt((float)(sumSq / n) + eps);
            for (int i = 0; i < n; i++)
                output[rowBase + i] = input[rowBase + i] * rinv * weight[i];
        }
    }

    private static void AssertClose(float[] expected, float[] actual, int rowCount, int n)
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
            $"Numerical drift exceeded tolerance (rowCount={rowCount},n={n}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
