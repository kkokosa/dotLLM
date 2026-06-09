using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan F32 matmul kernel.
/// </summary>
/// <remarks>
/// Compares against a scalar CPU reference (not <c>TensorPrimitives</c>) so the
/// comparison does not mask drift that might originate from SIMD reduction order
/// differences. Tolerances follow the mandate: relative 1e-3 / absolute 1e-4.
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanMatMulF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 1, 1)]                // degenerate
    [InlineData(8, 16, 1)]               // tiny GEMV
    [InlineData(64, 128, 1)]             // small GEMV
    [InlineData(17, 33, 5)]              // non-multiple-of-workgroup sizes
    [InlineData(256, 576, 1)]            // SmolLM hidden-size GEMV
    [InlineData(576, 1536, 1)]           // SmolLM up_proj GEMV
    [InlineData(128, 64, 8)]             // batched matmul
    [InlineData(576, 576, 4)]            // prefill-ish
    public void Launch_MatchesCpuReference(int m, int k, int n)
    {
        SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xABCDEF + m * 31 + k * 17 + n);
        float[] a = RandomFloats(rng, m * k);
        float[] b = RandomFloats(rng, n * k);
        float[] expected = new float[n * m];
        ReferenceGemm(a, b, expected, m, k, n);

        float[] actual = new float[n * m];

        using var device = VulkanDevice.Create();
        using var kernel = MatMulF32Kernel.Create(device, spvDir);

        using var bufA = device.Allocate((long)m * k * sizeof(float));
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(a, bufA);
        device.Upload(b, bufB);
        kernel.Launch(bufA, bufB, bufC, m, k, n);
        device.Download(bufC, actual);

        AssertClose(expected, actual, m, k, n);
    }

    [SkippableFact]
    public void Launch_NonTrivial_SmolLmAttentionProjectionShape()
    {
        // SmolLM-135M q/k/v projection: hidden 576 -> 576, single token (decode).
        SkipIfUnavailable(out string spvDir);

        const int m = 576;
        const int k = 576;
        const int n = 1;

        var rng = new Random(7);
        float[] a = RandomFloats(rng, m * k);
        float[] b = RandomFloats(rng, n * k);
        float[] expected = new float[n * m];
        ReferenceGemm(a, b, expected, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulF32Kernel.Create(device, spvDir);

        using var bufA = device.Allocate((long)m * k * sizeof(float));
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(a, bufA);
        device.Upload(b, bufB);
        kernel.Launch(bufA, bufB, bufC, m, k, n);

        float[] actual = new float[n * m];
        device.Download(bufC, actual);
        AssertClose(expected, actual, m, k, n);
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    internal static void SkipIfUnavailable(out string spvDir)
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string? found = FindSpvDir();
        Skip.If(
            found == null,
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
        spvDir = found!;
    }

    private static string? FindSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        return null;
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0); // [-1, 1]
        return arr;
    }

    /// <summary>
    /// Scalar reference GEMM: <c>C[N,M] = B[N,K] @ A[M,K]^T</c>,
    /// matching the CPU <c>GemvF32Scalar</c> reduction order.
    /// </summary>
    private static void ReferenceGemm(float[] a, float[] b, float[] c, int m, int k, int n)
    {
        for (int t = 0; t < n; t++)
        {
            int bRow = t * k;
            for (int row = 0; row < m; row++)
            {
                int aRow = row * k;
                float sum = 0;
                for (int j = 0; j < k; j++)
                    sum += a[aRow + j] * b[bRow + j];
                c[t * m + row] = sum;
            }
        }
    }

    internal static void AssertClose(float[] expected, float[] actual, int m, int k, int n)
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
            $"Numerical drift exceeded tolerance (m={m},k={k},n={n}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
