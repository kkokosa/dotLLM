using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads Qwen2.5-0.5B-Instruct Q8_0 (~530 MB) for Qwen2 architecture tests.
/// Tests tied embeddings, Q/K biases, and architecture auto-detection.
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class QwenModelFixture : IAsyncLifetime
{
    private const string RepoId = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
    private const string Filename = "qwen2.5-0.5b-instruct-q8_0.gguf";

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

[CollectionDefinition("QwenModel")]
public class QwenModelCollection : ICollectionFixture<QwenModelFixture>;
