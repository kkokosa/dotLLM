using DotLLM.Core.Tensors;
using DotLLM.Cpu;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Architectures;

/// <summary>
/// Integration tests for the Llama forward pass against SmolLM-135M Q8_0.
/// Validates the full pipeline: embedding → transformer blocks → final norm → LM head → logits.
/// </summary>
[Collection("SmallModel")]
public class LlamaForwardPassTests
{
    private readonly SmallModelFixture _fixture;

    public LlamaForwardPassTests(SmallModelFixture fixture)
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
    public void MultipleTokens_ProducesCorrectShape()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        int[] tokenIds = tokenizer.Encode("Hello");
        Assert.True(tokenIds.Length > 0, "Tokenizer should produce at least one token for 'Hello'.");

        int[] positions = new int[tokenIds.Length];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(1, logits.Shape[0]); // Only last token's logits returned
        Assert.Equal(model.Config.VocabSize, logits.Shape[1]);
    }

    [Fact]
    public void GreedyDecode_ProducesCoherentTokens()
    {
        var (model, gguf, tokenizer) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        // Encode prompt
        int[] promptIds = tokenizer.Encode("The capital of France is");
        int vocabSize = model.Config.VocabSize;

        // Prefill: run full prompt
        int[] positions = new int[promptIds.Length];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;

        // For this test, we run the full forward on all tokens and take
        // the argmax of the last token's logits as the next token.
        // We then decode single tokens iteratively (simplified — no KV cache).
        var generatedIds = new List<int>();
        int[] currentIds = promptIds;
        int[] currentPositions = positions;

        for (int step = 0; step < 5; step++)
        {
            using ITensor logits = model.Forward(currentIds, currentPositions, deviceId: -1);

            // Take argmax of logits (already last-token-only: [1, vocabSize])
            int nextTokenId;
            unsafe
            {
                float* logitPtr = (float*)logits.DataPointer;
                nextTokenId = ArgMax(new ReadOnlySpan<float>(logitPtr, vocabSize));
            }

            generatedIds.Add(nextTokenId);

            // Verify token is within vocab range
            Assert.InRange(nextTokenId, 0, vocabSize - 1);

            // Next step: full context (simplified — re-run entire sequence without KV cache)
            int newLen = promptIds.Length + generatedIds.Count;
            currentIds = new int[newLen];
            currentPositions = new int[newLen];
            Array.Copy(promptIds, currentIds, promptIds.Length);
            for (int g = 0; g < generatedIds.Count; g++)
                currentIds[promptIds.Length + g] = generatedIds[g];
            for (int i = 0; i < newLen; i++)
                currentPositions[i] = i;
        }

        // Decode generated tokens
        string generated = tokenizer.Decode(generatedIds.ToArray());
        Assert.False(string.IsNullOrEmpty(generated), "Generated text should not be empty.");

        // Verify not all tokens are the same (degenerate output)
        Assert.False(
            generatedIds.TrueForAll(id => id == generatedIds[0]),
            "All 5 generated tokens are identical — likely degenerate output.");
        Assert.Equal(5, generatedIds.Count);
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
    {
        int maxIdx = 0;
        float maxVal = span[0];
        for (int i = 1; i < span.Length; i++)
        {
            if (span[i] > maxVal)
            {
                maxVal = span[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
