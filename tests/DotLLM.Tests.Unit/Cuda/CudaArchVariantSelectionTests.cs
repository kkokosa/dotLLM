using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Tests arch-tiered PTX variant selection (<see cref="CudaModule.ResolveArchVariantPath"/>).
/// Pure file-system logic — no GPU or CUDA driver required.
/// </summary>
public class CudaArchVariantSelectionTests : IDisposable
{
    private readonly string _dir;

    public CudaArchVariantSelectionTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "dotllm_ptx_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    private string Touch(string fileName)
    {
        string path = Path.Combine(_dir, fileName);
        File.WriteAllText(path, "// stub ptx");
        return path;
    }

    [Fact]
    public void NoVariantsPresent_ReturnsBaseFile()
    {
        Touch("foo.ptx");

        string resolved = CudaModule.ResolveArchVariantPath(_dir, "foo.ptx", 8, 6);

        Assert.Equal(Path.Combine(_dir, "foo.ptx"), resolved);
    }

    [Fact]
    public void VariantPresent_SelectedAtMatchingArch()
    {
        Touch("foo.ptx");
        string sm80 = Touch("foo.sm_80.ptx");

        // sm_86 device: sm_80 variant (80 <= 86) preferred over base.
        string resolved = CudaModule.ResolveArchVariantPath(_dir, "foo.ptx", 8, 6);

        Assert.Equal(sm80, resolved);
    }

    [Fact]
    public void VariantPresent_FallsBackToBaseOnLowerArch()
    {
        Touch("foo.ptx");
        Touch("foo.sm_80.ptx");

        // sm_61 device: sm_80 variant (80 > 61) is NOT eligible → base fallback.
        string resolved = CudaModule.ResolveArchVariantPath(_dir, "foo.ptx", 6, 1);

        Assert.Equal(Path.Combine(_dir, "foo.ptx"), resolved);
    }

    [Fact]
    public void MultipleVariants_SelectsHighestNotExceedingDeviceArch()
    {
        Touch("foo.ptx");
        Touch("foo.sm_75.ptx");
        string sm80 = Touch("foo.sm_80.ptx");
        Touch("foo.sm_90.ptx");

        // sm_86 device: eligible variants are 75 and 80; pick the highest (80).
        string resolved = CudaModule.ResolveArchVariantPath(_dir, "foo.ptx", 8, 6);

        Assert.Equal(sm80, resolved);
    }

    [Fact]
    public void ZeroComputeCapability_AlwaysSelectsBase()
    {
        Touch("foo.ptx");
        Touch("foo.sm_80.ptx");

        // CC 0.0 (default, unknown device) → universal fallback, never a variant.
        string resolved = CudaModule.ResolveArchVariantPath(_dir, "foo.ptx", 0, 0);

        Assert.Equal(Path.Combine(_dir, "foo.ptx"), resolved);
    }

    [Fact]
    public void MalformedVariantToken_Ignored()
    {
        Touch("foo.ptx");
        Touch("foo.sm_80a.ptx"); // non-numeric arch token — must be ignored

        string resolved = CudaModule.ResolveArchVariantPath(_dir, "foo.ptx", 8, 6);

        Assert.Equal(Path.Combine(_dir, "foo.ptx"), resolved);
    }

    [Fact]
    public void OtherKernelVariants_DoNotLeakAcrossBaseNames()
    {
        Touch("foo.ptx");
        Touch("bar.sm_80.ptx"); // belongs to a different kernel

        string resolved = CudaModule.ResolveArchVariantPath(_dir, "foo.ptx", 8, 6);

        Assert.Equal(Path.Combine(_dir, "foo.ptx"), resolved);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); }
        catch { /* best-effort temp cleanup */ }
    }
}
