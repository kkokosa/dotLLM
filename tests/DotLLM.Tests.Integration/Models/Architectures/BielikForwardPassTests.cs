using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Integration tests for Bielik-1.5B-v3.0-Instruct — a Llama variant with linear layer biases
/// on all 7 projections per layer (attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down).
/// Verifies that bias support produces correct output for both Q8_0 and Q4_K_M quantizations.
/// </summary>
[Collection("BielikQ8Model")]
public class BielikQ8ForwardPassTests
{
    private readonly BielikQ8ModelFixture _fixture;

    public BielikQ8ForwardPassTests(BielikQ8ModelFixture fixture)
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
    public void GreedyGeneration_PredictsWarszawa()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        var response = generator.Generate("Stolicą Polski jest", options);

        Assert.True(response.GeneratedTokenIds.Length > 0);
        Assert.Contains("Warszawa", response.Text);
    }
}

[Collection("BielikQ4KModel")]
public class BielikQ4KForwardPassTests
{
    private readonly BielikQ4KModelFixture _fixture;

    public BielikQ4KForwardPassTests(BielikQ4KModelFixture fixture)
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
    public void GreedyGeneration_PredictsWarszawa()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        var response = generator.Generate("Stolicą Polski jest", options);

        Assert.True(response.GeneratedTokenIds.Length > 0);
        Assert.Contains("Warszawa", response.Text);
    }
}
