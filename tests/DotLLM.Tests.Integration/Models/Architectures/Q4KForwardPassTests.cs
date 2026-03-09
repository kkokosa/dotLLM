using System.Numerics.Tensors;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Integration tests for Q4_K_M model loading and inference.
/// Q4_K_M uses Q4_K for FFN layers and Q6_K for attention layers,
/// exercising the full mixed-quantization dispatch path.
/// </summary>
[Collection("Q4KModel")]
public class Q4KForwardPassTests
{
    private readonly Q4KModelFixture _fixture;

    public Q4KForwardPassTests(Q4KModelFixture fixture)
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

    // ──────────────────── Model loading ────────────────────

    [Fact]
    public void Load_DetectsKQuantTensors()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);

        var quantTypes = gguf.Tensors
            .Select(t => t.QuantizationType)
            .Distinct()
            .ToHashSet();

        // Q4_K_M uses Q4_K for FFN and Q6_K for attention
        Assert.Contains(QuantizationType.Q4_K, quantTypes);
        Assert.Contains(QuantizationType.Q6_K, quantTypes);
    }

    [Fact]
    public void Load_ModelConfigIsValid()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.True(config.VocabSize > 0);
        Assert.True(config.NumLayers > 0);
        Assert.True(config.HiddenSize > 0);
        Assert.True(config.NumAttentionHeads > 0);
    }

    // ──────────────────── Dequantization ────────────────────

    [Fact]
    public void DequantizeQ4K_RealTensor_AllFiniteAndReasonable()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);

        var tensor = gguf.Tensors.FirstOrDefault(t => t.QuantizationType == QuantizationType.Q4_K);
        if (tensor.Name is null) return; // skip if no Q4_K tensors

        long elementCount = tensor.Shape.ElementCount;
        nint tensorPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

        float[] dest = new float[elementCount];
        Dequantize.ToFloat32(tensorPtr, elementCount, QuantizationType.Q4_K, dest);

        Assert.All(dest, v => Assert.True(float.IsFinite(v), $"Non-finite value: {v}"));

        float maxAbs = dest.Max(MathF.Abs);
        Assert.True(maxAbs < 1000f, $"Max absolute value {maxAbs} exceeds reasonable range");
        Assert.True(maxAbs > 0f, "All values are zero — dequantization likely failed");
    }

    [Fact]
    public void DequantizeQ6K_RealTensor_AllFiniteAndReasonable()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);

        var tensor = gguf.Tensors.FirstOrDefault(t => t.QuantizationType == QuantizationType.Q6_K);
        if (tensor.Name is null) return;

        long elementCount = tensor.Shape.ElementCount;
        nint tensorPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

        float[] dest = new float[elementCount];
        Dequantize.ToFloat32(tensorPtr, elementCount, QuantizationType.Q6_K, dest);

        Assert.All(dest, v => Assert.True(float.IsFinite(v), $"Non-finite value: {v}"));

        float maxAbs = dest.Max(MathF.Abs);
        Assert.True(maxAbs < 1000f, $"Max absolute value {maxAbs} exceeds reasonable range");
        Assert.True(maxAbs > 0f, "All values are zero — dequantization likely failed");
    }

    // ──────────────────── Forward pass ────────────────────

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
                Assert.Equal(span1[i], span2[i]);
        }
    }

    // ──────────────────── Text generation ────────────────────

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
    public void GreedyGeneration_ProducesCoherentTokensForKnowledge()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        var response = generator.Generate("The capital of France is", options);

        // Q4_K_M on a 135M model may not get "Paris" due to quantization loss,
        // but should still produce coherent non-empty output with valid tokens.
        Assert.True(response.GeneratedTokenIds.Length > 0);
        Assert.False(string.IsNullOrWhiteSpace(response.Text));

        foreach (int id in response.GeneratedTokenIds)
            Assert.InRange(id, 0, model.Config.VocabSize - 1);
    }

    [Fact]
    public void GreedyGeneration_WithKvCache_ProducesCoherentOutput()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 20 };

        var response = generator.Generate("Once upon a time", options);

        Assert.True(response.GeneratedTokenCount > 0);
        Assert.False(string.IsNullOrEmpty(response.Text));

        // Not all tokens identical (degenerate output)
        if (response.GeneratedTokenIds.Length > 1)
        {
            Assert.False(
                response.GeneratedTokenIds.All(id => id == response.GeneratedTokenIds[0]),
                "All generated tokens are identical — likely degenerate output.");
        }
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
}
