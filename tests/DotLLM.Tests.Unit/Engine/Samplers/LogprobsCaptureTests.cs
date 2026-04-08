using System.Buffers;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Samplers;

public class LogprobsCaptureTests
{
    /// <summary>Minimal tokenizer stub for testing — decodes token IDs as string representations.</summary>
    private sealed class StubTokenizer : ITokenizer
    {
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public int VocabSize => 10;
        public int[] Encode(string text) => [];
        public string Decode(ReadOnlySpan<int> tokenIds) =>
            tokenIds.Length == 1 ? $"tok_{tokenIds[0]}" : string.Join("", tokenIds.ToArray().Select(id => $"tok_{id}"));
        public string DecodeToken(int tokenId) => $"tok_{tokenId}";
        public int CountTokens(string text) => text.Split(' ').Length;
    }

    [Fact]
    public void ComputeLogSoftmax_UniformLogits_ProducesEqualProbabilities()
    {
        float[] logits = [1.0f, 1.0f, 1.0f, 1.0f];
        float[] buffer = LogprobsCapture.ComputeLogSoftmax(logits);

        try
        {
            float expected = MathF.Log(0.25f); // ln(1/4)
            for (int i = 0; i < 4; i++)
                Assert.InRange(buffer[i], expected - 0.001f, expected + 0.001f);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    [Fact]
    public void ComputeLogSoftmax_DominantLogit_HasHighestLogprob()
    {
        float[] logits = [10.0f, 0.0f, 0.0f, 0.0f];
        float[] buffer = LogprobsCapture.ComputeLogSoftmax(logits);

        try
        {
            // Token 0 should have logprob close to 0 (probability ~1)
            Assert.True(buffer[0] > buffer[1]);
            Assert.True(buffer[0] > buffer[2]);
            Assert.True(buffer[0] > buffer[3]);
            Assert.InRange(buffer[0], -0.01f, 0.0f);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    [Fact]
    public void ComputeLogSoftmax_SumsToOne()
    {
        float[] logits = [2.0f, 1.0f, 0.5f, -1.0f, 3.0f];
        float[] buffer = LogprobsCapture.ComputeLogSoftmax(logits);

        try
        {
            // exp(log_softmax) should sum to ~1
            double sum = 0;
            for (int i = 0; i < 5; i++)
                sum += Math.Exp(buffer[i]);
            Assert.InRange(sum, 0.999, 1.001);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    [Fact]
    public void BuildInfo_CorrectTokenAndLogprob()
    {
        var tokenizer = new StubTokenizer();
        float[] logits = [0.0f, 5.0f, 1.0f, 0.0f];
        float[] buffer = LogprobsCapture.ComputeLogSoftmax(logits);

        try
        {
            var info = LogprobsCapture.BuildInfo(buffer.AsSpan(0, 4), 4, sampledTokenId: 1, topK: 0, tokenizer);

            Assert.Equal(1, info.TokenId);
            Assert.Equal("tok_1", info.Token);
            Assert.True(info.Logprob > -1.0f); // Token 1 has highest logit, should have high logprob
            Assert.NotNull(info.Bytes);
            Assert.Null(info.TopLogprobs); // topK=0 means no alternatives
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    [Fact]
    public void BuildInfo_TopK_ReturnsCorrectEntries()
    {
        var tokenizer = new StubTokenizer();
        float[] logits = [0.0f, 5.0f, 3.0f, 1.0f, 0.5f];
        float[] buffer = LogprobsCapture.ComputeLogSoftmax(logits);

        try
        {
            var info = LogprobsCapture.BuildInfo(buffer.AsSpan(0, 5), 5, sampledTokenId: 1, topK: 3, tokenizer);

            Assert.NotNull(info.TopLogprobs);
            Assert.Equal(3, info.TopLogprobs!.Length);

            // Top 3 should be tokens 1, 2, 3 (highest logits in order)
            Assert.Equal(1, info.TopLogprobs[0].TokenId); // logit=5
            Assert.Equal(2, info.TopLogprobs[1].TokenId); // logit=3
            Assert.Equal(3, info.TopLogprobs[2].TokenId); // logit=1

            // Logprobs should be in descending order
            Assert.True(info.TopLogprobs[0].Logprob >= info.TopLogprobs[1].Logprob);
            Assert.True(info.TopLogprobs[1].Logprob >= info.TopLogprobs[2].Logprob);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    [Fact]
    public void BuildInfo_TopK_LargerThanVocab_ReturnsAllTokens()
    {
        var tokenizer = new StubTokenizer();
        float[] logits = [1.0f, 2.0f, 3.0f];
        float[] buffer = LogprobsCapture.ComputeLogSoftmax(logits);

        try
        {
            var info = LogprobsCapture.BuildInfo(buffer.AsSpan(0, 3), 3, sampledTokenId: 2, topK: 10, tokenizer);

            Assert.NotNull(info.TopLogprobs);
            Assert.Equal(3, info.TopLogprobs!.Length); // Only 3 tokens in vocab
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    [Fact]
    public void ComputeLogSoftmax_WithMaskedTokens_HandlesNegativeInfinity()
    {
        float[] logits = [2.0f, float.NegativeInfinity, 1.0f, float.NegativeInfinity];
        float[] buffer = LogprobsCapture.ComputeLogSoftmax(logits);

        try
        {
            // Masked tokens should remain -inf or very negative
            Assert.True(float.IsNegativeInfinity(buffer[1]) || buffer[1] < -30f);
            Assert.True(float.IsNegativeInfinity(buffer[3]) || buffer[3] < -30f);

            // Unmasked tokens should have valid logprobs
            Assert.False(float.IsNaN(buffer[0]));
            Assert.False(float.IsNaN(buffer[2]));

            // exp(logprobs) of unmasked tokens should sum to ~1
            double sum = Math.Exp(buffer[0]) + Math.Exp(buffer[2]);
            Assert.InRange(sum, 0.999, 1.001);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    [Fact]
    public void BuildInfo_WithMaskedTokens_ExcludesFromTopK()
    {
        var tokenizer = new StubTokenizer();
        float[] logits = [2.0f, float.NegativeInfinity, 1.0f, float.NegativeInfinity];
        float[] buffer = LogprobsCapture.ComputeLogSoftmax(logits);

        try
        {
            var info = LogprobsCapture.BuildInfo(buffer.AsSpan(0, 4), 4, sampledTokenId: 0, topK: 5, tokenizer);

            Assert.NotNull(info.TopLogprobs);
            // Only 2 non-masked tokens should appear
            Assert.Equal(2, info.TopLogprobs!.Length);
            Assert.Equal(0, info.TopLogprobs[0].TokenId);
            Assert.Equal(2, info.TopLogprobs[1].TokenId);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }
}
