using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Integration tests for <see cref="TextGenerator"/> against SmolLM-135M Q8_0.
/// </summary>
[Collection("SmallModel")]
public class TextGeneratorTests
{
    private readonly SmallModelFixture _fixture;

    public TextGeneratorTests(SmallModelFixture fixture)
    {
        _fixture = fixture;
    }

    private (LlamaModel model, GgufFile gguf, BpeTokenizer tokenizer) LoadModel()
    {
        var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = LlamaModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        return (model, gguf, tokenizer);
    }

    [Fact]
    public void GreedyGeneration_ProducesNonEmptyOutput()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        var response = generator.Generate("Hello", options);

        Assert.False(string.IsNullOrEmpty(response.Text), "Generated text should not be empty.");
        Assert.True(response.GeneratedTokenCount > 0);
        Assert.True(response.GeneratedTokenIds.Length > 0);
    }

    [Fact]
    public void GreedyGeneration_PredictsParis()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 5 };

        var response = generator.Generate("The capital of France is", options);

        // The first generated token should be "Paris"
        Assert.True(response.GeneratedTokenIds.Length > 0);
        string firstToken = tokenizer.DecodeToken(response.GeneratedTokenIds[0]).Trim();
        Assert.Equal("Paris", firstToken);
    }

    [Fact]
    public void SeededSampling_IsDeterministic()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var options = new InferenceOptions { Temperature = 0.8f, Seed = 42, MaxTokens = 10 };

        var generator1 = new TextGenerator(model, tokenizer);
        var response1 = generator1.Generate("Once upon a time", options);

        var generator2 = new TextGenerator(model, tokenizer);
        var response2 = generator2.Generate("Once upon a time", options);

        Assert.Equal(response1.Text, response2.Text);
        Assert.Equal(response1.GeneratedTokenIds, response2.GeneratedTokenIds);
    }

    [Fact]
    public void MaxTokens_StopsAtLimit()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 5 };

        var response = generator.Generate("Hello world", options);

        Assert.True(response.GeneratedTokenIds.Length <= 5,
            $"Expected at most 5 tokens, got {response.GeneratedTokenIds.Length}.");
    }

    [Fact]
    public void EosStop_ReportsCorrectFinishReason()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        // Generate enough tokens that we're likely to hit EOS or Length
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 200 };

        var response = generator.Generate("The capital of France is", options);

        // Should be either Stop (EOS hit) or Length (max tokens hit)
        Assert.True(
            response.FinishReason == FinishReason.Stop || response.FinishReason == FinishReason.Length,
            $"Expected Stop or Length, got {response.FinishReason}.");
    }

    [Fact]
    public void Timings_ArePopulatedAfterGeneration()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        var response = generator.Generate("The capital of France is", options);
        var timings = response.Timings;

        // Prefill timing should be positive
        Assert.True(timings.PrefillTimeMs > 0,
            $"PrefillTimeMs should be > 0, got {timings.PrefillTimeMs}");

        // PrefillTokenCount should match prompt length
        Assert.Equal(response.PromptTokenCount, timings.PrefillTokenCount);

        // With multiple generated tokens, decode time should be positive
        if (response.GeneratedTokenCount > 1)
        {
            Assert.True(timings.DecodeTimeMs > 0,
                $"DecodeTimeMs should be > 0 when multiple tokens generated, got {timings.DecodeTimeMs}");
            Assert.Equal(response.GeneratedTokenCount - 1, timings.DecodeTokenCount);
        }

        // Sampling time should be positive when tokens were generated
        if (response.GeneratedTokenCount > 0)
        {
            Assert.True(timings.SamplingTimeMs > 0,
                $"SamplingTimeMs should be > 0, got {timings.SamplingTimeMs}");
        }

        // Derived tok/s should be positive
        Assert.True(timings.PrefillTokensPerSec > 0);
        if (timings.DecodeTokenCount > 0)
            Assert.True(timings.DecodeTokensPerSec > 0);
    }

    [Fact]
    public void OnTokenGenerated_CallbackIsInvoked()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 5 };
        var callbackTokens = new List<int>();

        var response = generator.Generate("Hello", options,
            onTokenGenerated: tokenId => callbackTokens.Add(tokenId));

        // Callback should have been invoked for each generated token
        Assert.Equal(response.GeneratedTokenCount, callbackTokens.Count);
        Assert.Equal(response.GeneratedTokenIds, callbackTokens.ToArray());
    }
}
