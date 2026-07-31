using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Smoke test for the Vulkan compute scaffold. Runs if a Vulkan loader + driver
/// are present on the host; skips cleanly otherwise.
/// </summary>
/// <remarks>
/// Opt-out via <c>DOTLLM_SKIP_VULKAN=1</c> for CI environments where a Vulkan
/// loader is installed but no usable driver (e.g. swiftshader-free
/// headless Linux VMs) — the <see cref="VulkanDevice.IsAvailable"/> probe
/// catches most such cases but the env var is a belt-and-braces escape hatch.
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanAddKernelTests
{
    [SkippableFact]
    public void AddKernel_ProducesElementwiseSum()
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string? spvDir = FindSpvDir();
        Skip.If(
            spvDir == null,
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");

        const int n = 1024;
        var a = new float[n];
        var b = new float[n];
        var expected = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = i * 0.5f;
            b[i] = i * -0.25f + 3.0f;
            expected[i] = a[i] + b[i];
        }

        using var device = VulkanDevice.Create();
        using var kernel = AddKernel.Create(device, spvDir!);

        using var bufA = device.Allocate(n * sizeof(float));
        using var bufB = device.Allocate(n * sizeof(float));
        using var bufC = device.Allocate(n * sizeof(float));

        device.Upload(a, bufA);
        device.Upload(b, bufB);

        kernel.Launch(bufA, bufB, bufC, n);

        var result = new float[n];
        device.Download(bufC, result);

        // Exact equality — float addition is deterministic, no reduction here.
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(expected[i], result[i]);
        }
    }

    [SkippableFact]
    public void AddKernel_LaunchesRepeatedly_OnSameInstance()
    {
        // Regression: the descriptor set used to be allocated inside Launch from a
        // maxSets = 1 pool, so a second dispatch on the same instance failed with
        // VK_ERROR_OUT_OF_POOL_MEMORY. The set is now allocated once and re-written.
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string? spvDir = FindSpvDir();
        Skip.If(
            spvDir == null,
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");

        const int n = 256;
        using var device = VulkanDevice.Create();
        using var kernel = AddKernel.Create(device, spvDir!);

        using var bufA = device.Allocate(n * sizeof(float));
        using var bufB = device.Allocate(n * sizeof(float));
        using var bufC = device.Allocate(n * sizeof(float));

        var a = new float[n];
        var b = new float[n];
        var result = new float[n];

        // Three dispatches with different data, re-binding the same descriptor set each time.
        for (int pass = 1; pass <= 3; pass++)
        {
            for (int i = 0; i < n; i++)
            {
                a[i] = i * pass;
                b[i] = -i * 0.5f;
            }
            device.Upload(a, bufA);
            device.Upload(b, bufB);

            kernel.Launch(bufA, bufB, bufC, n);

            device.Download(bufC, result);
            for (int i = 0; i < n; i++)
                Assert.Equal(a[i] + b[i], result[i]);
        }
    }

    [SkippableFact]
    public void Device_ReportsName()
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        using var device = VulkanDevice.Create();
        Assert.False(string.IsNullOrWhiteSpace(device.DeviceName));
        Assert.True(device.VendorId != 0);
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
}
