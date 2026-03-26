using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Tests for CUDA device detection and basic operations.
/// Skip if no GPU is available.
/// </summary>
[Trait("Category", "GPU")]
public class CudaDeviceTests
{
    [SkippableFact]
    public void IsAvailable_ReturnsTrue_WhenGpuPresent()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        Assert.True(CudaDevice.IsAvailable());
    }

    [SkippableFact]
    public void GetDeviceCount_ReturnsAtLeastOne()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        int count = CudaDevice.GetDeviceCount();
        Assert.True(count >= 1);
    }

    [SkippableFact]
    public void GetDevice_ReturnsValidInfo()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        var device = CudaDevice.GetDevice(0);

        Assert.False(string.IsNullOrEmpty(device.Name));
        Assert.True(device.TotalMemoryBytes > 0);
        Assert.True(device.ComputeCapabilityMajor >= 3);
        Assert.True(device.MultiprocessorCount > 0);
    }
}
