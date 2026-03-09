using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads Bielik-1.5B-v3.0-Instruct Q8_0 GGUF (~1.6 GB) for bias support integration tests.
/// Bielik has linear layer biases on all 7 projections per layer — exercises the bias code path.
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class BielikQ8ModelFixture : IAsyncLifetime
{
    private const string RepoId = "second-state/Bielik-1.5B-v3.0-Instruct-GGUF";
    private const string Filename = "Bielik-1.5B-v3.0-Instruct-Q8_0.gguf";

    /// <summary>Full local path to the downloaded GGUF file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    public async Task InitializeAsync()
    {
        string cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache");

        string cachedPath = Path.Combine(cacheDir, RepoId.Replace('/', Path.DirectorySeparatorChar), Filename);

        if (File.Exists(cachedPath))
        {
            FilePath = cachedPath;
            return;
        }

        using var downloader = new HuggingFaceDownloader();
        FilePath = await downloader.DownloadFileAsync(RepoId, Filename, cacheDir);
    }

    public Task DisposeAsync() => Task.CompletedTask;
}

[CollectionDefinition("BielikQ8Model")]
public class BielikQ8ModelCollection : ICollectionFixture<BielikQ8ModelFixture>;
