using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads a small GGUF model once and caches it for all tests in the collection.
/// Uses <c>QuantFactory/SmolLM-135M-GGUF</c> Q8_0 (~145 MB) — llama architecture,
/// so both GGUF parsing and <c>GgufModelConfigExtractor</c> work against it.
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class SmallModelFixture : IAsyncLifetime
{
    /// <summary>Full local path to the downloaded GGUF file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    public async Task InitializeAsync() =>
        FilePath = await TestModelDownloader.EnsureModelAsync("QuantFactory/SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");

    public Task DisposeAsync() => Task.CompletedTask;
}

[CollectionDefinition("SmallModel")]
public class SmallModelCollection : ICollectionFixture<SmallModelFixture>;
