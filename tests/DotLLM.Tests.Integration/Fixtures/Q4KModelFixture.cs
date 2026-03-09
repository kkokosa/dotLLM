using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads SmolLM-135M Q4_K_M GGUF (~100 MB) for K-quant integration tests.
/// Uses Q4_K for FFN layers and Q6_K for attention layers — exercises the full
/// mixed-quantization dispatch path.
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class Q4KModelFixture : IAsyncLifetime
{
    private const string RepoId = "QuantFactory/SmolLM-135M-GGUF";
    private const string Filename = "SmolLM-135M.Q4_K_M.gguf";

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

[CollectionDefinition("Q4KModel")]
public class Q4KModelCollection : ICollectionFixture<Q4KModelFixture>;
