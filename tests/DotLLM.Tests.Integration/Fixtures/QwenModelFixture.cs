using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads Qwen2.5-0.5B-Instruct Q8_0 (~530 MB) for Qwen2 architecture tests.
/// Tests tied embeddings, Q/K biases, and architecture auto-detection.
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class QwenModelFixture : IAsyncLifetime
{
    /// <summary>Full local path to the downloaded GGUF file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    public async Task InitializeAsync() =>
        FilePath = await TestModelDownloader.EnsureModelAsync("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf");

    public Task DisposeAsync() => Task.CompletedTask;
}

[CollectionDefinition("QwenModel")]
public class QwenModelCollection : ICollectionFixture<QwenModelFixture>;
