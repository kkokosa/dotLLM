using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// End-to-end text generation tests for Qwen2 architecture via <see cref="TextGenerator"/>.
/// Validates that the full pipeline works through the <see cref="TransformerModel"/> interface.
/// </summary>
[Collection("QwenModel")]
public class QwenTextGeneratorTests
{
    private readonly QwenModelFixture _fixture;

    public QwenTextGeneratorTests(QwenModelFixture fixture)
    {
        _fixture = fixture;
    }

    private (TransformerModel model, GgufFile gguf, BpeTokenizer tokenizer) LoadModel()
    {
        var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = TransformerModel.LoadFromGguf(gguf, config);
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

        Assert.True(response.GeneratedTokenIds.Length > 0);
        string firstToken = tokenizer.DecodeToken(response.GeneratedTokenIds[0]).Trim();
        Assert.Equal("Paris", firstToken);
    }

    [Fact]
    public void Timings_ArePopulated()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 5 };

        var response = generator.Generate("The capital of France is", options);
        var timings = response.Timings;

        Assert.True(timings.PrefillTimeMs > 0);
        Assert.True(timings.PrefillTokensPerSec > 0);

        if (response.GeneratedTokenCount > 1)
        {
            Assert.True(timings.DecodeTimeMs > 0);
            Assert.True(timings.DecodeTokensPerSec > 0);
        }
    }

    [Fact]
    public async Task StreamingOutput_MatchesSynchronousOutput()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        // Synchronous generation
        var syncResponse = generator.Generate("The capital of France is", options);

        // Streaming generation — concatenate all text
        var sb = new StringBuilder();
        await foreach (var token in generator.GenerateStreamingTokensAsync("The capital of France is", options))
            sb.Append(token.Text);

        Assert.Equal(syncResponse.Text, sb.ToString());
    }
}
