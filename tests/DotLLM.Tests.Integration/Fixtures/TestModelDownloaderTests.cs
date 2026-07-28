using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Cache-root resolution for <see cref="TestModelDownloader"/>. Deliberately declares no
/// <c>[Collection]</c> so it pulls in no model fixture and downloads nothing — these are
/// pure path assertions.
/// </summary>
public class TestModelDownloaderTests
{
    private static string DefaultCacheDir => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache");

    [Fact]
    public void ResolveCacheDirectory_WhenUnset_UsesDefault()
    {
        Assert.Equal(DefaultCacheDir, TestModelDownloader.ResolveCacheDirectory(null));
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("\t")]
    public void ResolveCacheDirectory_WhenBlank_UsesDefault(string envValue)
    {
        Assert.Equal(DefaultCacheDir, TestModelDownloader.ResolveCacheDirectory(envValue));
    }

    [Fact]
    public void ResolveCacheDirectory_WhenSet_UsesOverride()
    {
        string target = Path.Combine(Path.GetTempPath(), "dotllm-test-cache-override");

        Assert.Equal(Path.GetFullPath(target), TestModelDownloader.ResolveCacheDirectory(target));
    }

    [Fact]
    public void ResolveCacheDirectory_WhenRelative_IsMadeAbsolute()
    {
        string resolved = TestModelDownloader.ResolveCacheDirectory("some-relative-cache");

        Assert.True(Path.IsPathFullyQualified(resolved));
        Assert.Equal(Path.GetFullPath("some-relative-cache"), resolved);
    }

    [Fact]
    public void ResolveCacheDirectory_Override_DoesNotAppendTestCacheSegment()
    {
        // Pointing the override at ~/.dotllm/models must use that directory verbatim,
        // so the layout stays interchangeable with the CLI's model directory.
        string models = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "models");

        Assert.Equal(models, TestModelDownloader.ResolveCacheDirectory(models));
    }

    [Fact]
    public void CacheDirectory_ReflectsEnvironmentVariable()
    {
        // Reads the ambient value rather than mutating it, so this stays safe under
        // xUnit's parallel collection execution.
        string expected = TestModelDownloader.ResolveCacheDirectory(
            Environment.GetEnvironmentVariable(TestModelDownloader.CacheDirEnvVar));

        Assert.Equal(expected, TestModelDownloader.CacheDirectory);
    }
}
