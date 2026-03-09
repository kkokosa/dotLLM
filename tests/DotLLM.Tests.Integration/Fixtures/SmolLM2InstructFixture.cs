using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads SmolLM2-135M-Instruct Q8_0 (~145 MB) for chat template integration tests.
/// Small instruct model with ChatML template — exercises the Jinja2 engine end-to-end.
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class SmolLM2InstructFixture : IAsyncLifetime
{
    private const string RepoId = "bartowski/SmolLM2-135M-Instruct-GGUF";
    private const string Filename = "SmolLM2-135M-Instruct-Q8_0.gguf";

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

[CollectionDefinition("SmolLM2Instruct")]
public class SmolLM2InstructCollection : ICollectionFixture<SmolLM2InstructFixture>;
