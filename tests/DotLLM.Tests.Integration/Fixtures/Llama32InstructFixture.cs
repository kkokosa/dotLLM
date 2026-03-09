using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads Llama-3.2-1B-Instruct Q8_0 (~1.1 GB) for chat template integration tests.
/// Exercises the complex Llama 3.2 Jinja2 template with dict literals, slicing,
/// strftime_now, and tool-use formatting.
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class Llama32InstructFixture : IAsyncLifetime
{
    private const string RepoId = "bartowski/Llama-3.2-1B-Instruct-GGUF";
    private const string Filename = "Llama-3.2-1B-Instruct-Q8_0.gguf";

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

[CollectionDefinition("Llama32Instruct")]
public class Llama32InstructCollection : ICollectionFixture<Llama32InstructFixture>;
