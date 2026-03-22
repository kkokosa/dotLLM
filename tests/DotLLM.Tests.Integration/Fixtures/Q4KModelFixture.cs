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
    /// <summary>Full local path to the downloaded GGUF file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    public async Task InitializeAsync() =>
        FilePath = await TestModelDownloader.EnsureModelAsync("QuantFactory/SmolLM-135M-GGUF", "SmolLM-135M.Q4_K_M.gguf");

    public Task DisposeAsync() => Task.CompletedTask;
}

[CollectionDefinition("Q4KModel")]
public class Q4KModelCollection : ICollectionFixture<Q4KModelFixture>;
