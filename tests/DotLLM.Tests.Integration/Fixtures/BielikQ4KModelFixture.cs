using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads Bielik-1.5B-v3.0-Instruct Q4_K_M GGUF (~0.9 GB) for bias + K-quant integration tests.
/// Bielik has linear layer biases on all 7 projections per layer — exercises bias + mixed-quant paths.
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class BielikQ4KModelFixture : IAsyncLifetime
{
    private const string RepoId = "second-state/Bielik-1.5B-v3.0-Instruct-GGUF";
    private const string Filename = "Bielik-1.5B-v3.0-Instruct-Q4_K_M.gguf";

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

[CollectionDefinition("BielikQ4KModel")]
public class BielikQ4KModelCollection : ICollectionFixture<BielikQ4KModelFixture>;
