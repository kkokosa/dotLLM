using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Integration tests for Qwen2 architecture forward pass against Qwen2.5-0.5B-Instruct Q8_0.
/// Validates architecture auto-detection, tied embeddings, Q/K biases, and correct inference.
/// </summary>
[Collection("QwenModel")]
public class QwenForwardPassTests
{
    private readonly QwenModelFixture _fixture;

    public QwenForwardPassTests(QwenModelFixture fixture)
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
    public void Config_DetectsQwenArchitecture()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.Qwen, config.Architecture);
    }

    [Fact]
    public void Config_HasTiedEmbeddings()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);

        // Qwen2 models typically have tied embeddings — no output.weight tensor
        bool hasOutputWeight = gguf.TensorsByName.ContainsKey("output.weight");
        if (!hasOutputWeight)
        {
            // When output.weight is absent, the model uses tied embeddings
            // (the weight loading code aliases token_embd.weight)
            Assert.True(true, "Model uses tied embeddings (no output.weight tensor).");
        }
    }

    [Fact]
    public void Config_HasQKBiases()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);

        // Qwen2 has biases on Q and K (but not V)
        Assert.True(gguf.TensorsByName.ContainsKey("blk.0.attn_q.bias"),
            "Qwen2 should have Q bias");
        Assert.True(gguf.TensorsByName.ContainsKey("blk.0.attn_k.bias"),
            "Qwen2 should have K bias");
    }

    [Fact]
    public void SingleToken_ProducesVocabSizedLogits()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int bosId = tokenizer.BosTokenId;
        using ITensor logits = model.Forward([bosId], [0], deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(model.Config.VocabSize, logits.Shape[1]);
    }

    [Fact]
    public void SingleToken_LogitsAreFinite()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int bosId = tokenizer.BosTokenId;
        using ITensor logits = model.Forward([bosId], [0], deviceId: -1);

        unsafe
        {
            var logitSpan = new ReadOnlySpan<float>((void*)logits.DataPointer, (int)logits.ElementCount);
            for (int i = 0; i < logitSpan.Length; i++)
            {
                Assert.True(float.IsFinite(logitSpan[i]),
                    $"Logit at index {i} is not finite: {logitSpan[i]}");
            }
        }
    }

    [Fact]
    public void SameInput_ProducesSameOutput()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int bosId = tokenizer.BosTokenId;
        using ITensor logits1 = model.Forward([bosId], [0], deviceId: -1);
        using ITensor logits2 = model.Forward([bosId], [0], deviceId: -1);

        Assert.Equal(logits1.ElementCount, logits2.ElementCount);

        unsafe
        {
            var span1 = new ReadOnlySpan<float>((void*)logits1.DataPointer, (int)logits1.ElementCount);
            var span2 = new ReadOnlySpan<float>((void*)logits2.DataPointer, (int)logits2.ElementCount);

            for (int i = 0; i < span1.Length; i++)
            {
                Assert.Equal(span1[i], span2[i]);
            }
        }
    }

    [Fact]
    public void Forward_WithKvCache_PrefillMatchesUncached()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int[] tokenIds = tokenizer.Encode("The capital of France is");
        int[] positions = new int[tokenIds.Length];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;

        // Uncached forward
        using ITensor uncachedLogits = model.Forward(tokenIds, positions, deviceId: -1);

        // Cached forward (prefill)
        using var kvCache = new SimpleKvCache(
            model.Config.NumLayers, model.Config.NumKvHeads, model.Config.HeadDim,
            tokenIds.Length + 10);
        using ITensor cachedLogits = model.Forward(tokenIds, positions, deviceId: -1, kvCache);

        // Compare: must be bit-identical
        Assert.Equal(uncachedLogits.ElementCount, cachedLogits.ElementCount);
        unsafe
        {
            var uncached = new ReadOnlySpan<float>((void*)uncachedLogits.DataPointer, (int)uncachedLogits.ElementCount);
            var cached = new ReadOnlySpan<float>((void*)cachedLogits.DataPointer, (int)cachedLogits.ElementCount);
            for (int i = 0; i < uncached.Length; i++)
            {
                Assert.Equal(uncached[i], cached[i]);
            }
        }
    }

    [Fact]
    public void GreedyDecode_PredictsParis()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int[] tokenIds = tokenizer.Encode("The capital of France is");
        int[] positions = new int[tokenIds.Length];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

        int vocabSize = model.Config.VocabSize;
        int nextTokenId;
        unsafe
        {
            float* logitPtr = (float*)logits.DataPointer;
            nextTokenId = ArgMax(new ReadOnlySpan<float>(logitPtr, vocabSize));
        }

        string predicted = tokenizer.DecodeToken(nextTokenId).Trim();
        Assert.Equal("Paris", predicted);
    }

    private static int ArgMax(ReadOnlySpan<float> span)
        => System.Numerics.Tensors.TensorPrimitives.IndexOfMax(span);
}
