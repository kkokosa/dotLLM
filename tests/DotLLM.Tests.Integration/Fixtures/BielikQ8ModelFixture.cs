using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads Bielik-1.5B-v3.0-Instruct Q8_0 GGUF (~1.6 GB) for bias support integration tests.
/// Bielik has linear layer biases on all 7 projections per layer — exercises the bias code path.
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class BielikQ8ModelFixture : IAsyncLifetime
{
    /// <summary>Full local path to the downloaded GGUF file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    public async Task InitializeAsync() =>
        FilePath = await TestModelDownloader.EnsureModelAsync("second-state/Bielik-1.5B-v3.0-Instruct-GGUF", "Bielik-1.5B-v3.0-Instruct-Q8_0.gguf");

    public Task DisposeAsync() => Task.CompletedTask;
}

[CollectionDefinition("BielikQ8Model")]
public class BielikQ8ModelCollection : ICollectionFixture<BielikQ8ModelFixture>;
