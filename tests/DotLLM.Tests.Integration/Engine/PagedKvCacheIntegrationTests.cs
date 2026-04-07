using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Integration tests for paged KV-cache with <see cref="TextGenerator"/>.
/// Verifies that <see cref="PagedKvCache"/> produces identical output to <see cref="SimpleKvCache"/>
/// and that prefix caching works correctly with paged allocation.
/// </summary>
[Collection("SmallModel")]
public class PagedKvCacheIntegrationTests
{
    private readonly SmallModelFixture _fixture;

    public PagedKvCacheIntegrationTests(SmallModelFixture fixture)
    {
        _fixture = fixture;
    }

    private (TransformerModel model, GgufFile gguf, BpeTokenizer tokenizer, ModelConfig config) LoadModel()
    {
        var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        return (model, gguf, tokenizer, config);
    }

    [Fact]
    public void PagedKvCache_MatchesSimpleKvCache_GreedyOutput()
    {
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        // Generate with default SimpleKvCache
        var simpleGenerator = new TextGenerator(model, tokenizer);
        var simpleResponse = simpleGenerator.Generate("The capital of France is", options);

        // Generate with PagedKvCache
        using var pagedFactory = new PagedKvCacheFactory(
            config.NumLayers, config.NumKvHeads, config.HeadDim);
        var pagedGenerator = new TextGenerator(model, tokenizer,
            kvCacheFactory: (cfg, size) => pagedFactory.Create(size));
        var pagedResponse = pagedGenerator.Generate("The capital of France is", options);

        Assert.Equal(simpleResponse.Text, pagedResponse.Text);
        Assert.Equal(simpleResponse.GeneratedTokenIds, pagedResponse.GeneratedTokenIds);
    }

    [Fact]
    public void PagedKvCache_WithPrefixCache_ReusesCacheOnSecondCall()
    {
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        using var pagedFactory = new PagedKvCacheFactory(
            config.NumLayers, config.NumKvHeads, config.HeadDim);
        using var prefixCache = new PrefixCache(maxEntries: 2);

        var generator = new TextGenerator(model, tokenizer,
            kvCacheFactory: (cfg, size) => pagedFactory.Create(size),
            prefixCache: prefixCache);

        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 5 };

        // First call — no cached tokens
        var response1 = generator.Generate("The capital of France is", options);
        Assert.Equal(0, response1.Timings.CachedTokenCount);
        Assert.True(response1.GeneratedTokenCount > 0);

        // Second call with extended prompt (shares prefix with first call's full sequence)
        // The prefix cache stores prompt + generated tokens from call 1.
        // Call 2's prompt = same prompt → shares that prefix.
        var response2 = generator.Generate("The capital of France is", options);
        Assert.True(response2.Timings.CachedTokenCount > 0,
            $"Expected cached tokens > 0 on second call, got {response2.Timings.CachedTokenCount}. " +
            "PrefixCache should reuse the PagedKvCache from the first call.");
    }

    [Fact]
    public async Task PagedKvCache_StreamingMatchesSynchronous()
    {
        var (model, gguf, tokenizer, config) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        using var pagedFactory = new PagedKvCacheFactory(
            config.NumLayers, config.NumKvHeads, config.HeadDim);

        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        var generator = new TextGenerator(model, tokenizer,
            kvCacheFactory: (cfg, size) => pagedFactory.Create(size));

        // Synchronous
        var syncResponse = generator.Generate("The capital of France is", options);

        // Streaming
        var sb = new System.Text.StringBuilder();
        await foreach (var token in generator.GenerateStreamingTokensAsync("The capital of France is", options))
            sb.Append(token.Text);

        Assert.Equal(syncResponse.Text, sb.ToString());
    }
}
