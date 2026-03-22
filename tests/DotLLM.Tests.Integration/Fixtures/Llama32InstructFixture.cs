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
    /// <summary>Full local path to the downloaded GGUF file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    public async Task InitializeAsync() =>
        FilePath = await TestModelDownloader.EnsureModelAsync("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");

    public Task DisposeAsync() => Task.CompletedTask;
}

[CollectionDefinition("Llama32Instruct")]
public class Llama32InstructCollection : ICollectionFixture<Llama32InstructFixture>;
