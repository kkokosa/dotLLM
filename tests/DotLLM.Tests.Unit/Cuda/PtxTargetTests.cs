using System.Text.RegularExpressions;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Guards the compatibility baseline of the checked-in PTX. These files are build
/// artifacts, so it is easy to regenerate one with a newer toolkit's default
/// architecture and silently drop support for older GPUs and drivers — which has
/// happened before (a CUDA 13.1 regeneration shipped <c>.target sm_75</c>).
/// </summary>
/// <remarks>
/// The baseline is <c>compute_61</c>, matching <c>native/build.ps1</c> /
/// <c>native/build.sh</c>. PTX is forward-compatible, so sm_61 PTX runs on every
/// GPU from Pascal onward; sm_75 PTX does not load on Pascal at all.
/// </remarks>
public sealed class PtxTargetTests
{
    [Fact]
    public void CheckedInPtx_TargetsBaselineArchitecture()
    {
        string ptxDir = FindPtxDir();
        string[] files = Directory.GetFiles(ptxDir, "*.ptx");
        Assert.NotEmpty(files);

        foreach (string file in files)
        {
            string text = File.ReadAllText(file);
            Match target = Regex.Match(text, @"^\.target\s+(\S+)", RegexOptions.Multiline);
            Assert.True(target.Success, $"{Path.GetFileName(file)} declares no .target directive.");
            Assert.True(
                target.Groups[1].Value == "sm_61",
                $"{Path.GetFileName(file)} targets '{target.Groups[1].Value}', not the sm_61 baseline. " +
                "Regenerate with native/build.ps1 (or build.sh), which pins -arch=compute_61.");
        }
    }

    private static string FindPtxDir()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            string candidate = Path.Combine(dir.FullName, "native", "ptx");
            if (Directory.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException("Could not locate native/ptx from the test output directory.");
    }
}
