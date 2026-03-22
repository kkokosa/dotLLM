using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads SmolLM2-135M-Instruct Q8_0 (~145 MB) for chat template integration tests.
/// Small instruct model with ChatML template — exercises the Jinja2 engine end-to-end.
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class SmolLM2InstructFixture : IAsyncLifetime
{
    /// <summary>Full local path to the downloaded GGUF file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    public async Task InitializeAsync() =>
        FilePath = await TestModelDownloader.EnsureModelAsync("bartowski/SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q8_0.gguf");

    public Task DisposeAsync() => Task.CompletedTask;
}

[CollectionDefinition("SmolLM2Instruct")]
public class SmolLM2InstructCollection : ICollectionFixture<SmolLM2InstructFixture>;
