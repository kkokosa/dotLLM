using DotLLM.Server;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for <see cref="RequestValidator"/> input validation.
/// </summary>
public class RequestValidatorTests
{
    // ── Chat request validation ──

    [Fact]
    public void ValidateChatRequest_EmptyMessages_ReturnsError()
    {
        var request = new ChatCompletionRequest { Messages = [] };
        var error = RequestValidator.ValidateChatRequest(request);
        Assert.Equal("messages array must not be empty", error);
    }

    [Fact]
    public void ValidateChatRequest_TooManyMessages_ReturnsError()
    {
        var messages = new ChatMessageDto[RequestValidator.MaxMessages + 1];
        for (int i = 0; i < messages.Length; i++)
            messages[i] = new ChatMessageDto { Role = "user", Content = "hi" };

        var request = new ChatCompletionRequest { Messages = messages };
        var error = RequestValidator.ValidateChatRequest(request);
        Assert.Contains("exceeds maximum", error);
    }

    [Fact]
    public void ValidateChatRequest_MaxTokensZero_ReturnsError()
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = "hi" }],
            MaxTokens = 0,
        };
        var error = RequestValidator.ValidateChatRequest(request);
        Assert.Equal("max_tokens must be a positive integer", error);
    }

    [Fact]
    public void ValidateChatRequest_MaxTokensNegative_ReturnsError()
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = "hi" }],
            MaxTokens = -5,
        };
        var error = RequestValidator.ValidateChatRequest(request);
        Assert.Equal("max_tokens must be a positive integer", error);
    }

    [Fact]
    public void ValidateChatRequest_ValidRequest_ReturnsNull()
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = "Hello" }],
            MaxTokens = 100,
        };
        Assert.Null(RequestValidator.ValidateChatRequest(request));
    }

    [Fact]
    public void ValidateChatRequest_NoMaxTokens_ReturnsNull()
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = "Hello" }],
        };
        Assert.Null(RequestValidator.ValidateChatRequest(request));
    }

    // ── Completion request validation ──

    [Fact]
    public void ValidateCompletionRequest_EmptyPrompt_ReturnsError()
    {
        var request = new CompletionRequest { Prompt = "" };
        var error = RequestValidator.ValidateCompletionRequest(request);
        Assert.Equal("prompt must not be empty", error);
    }

    [Fact]
    public void ValidateCompletionRequest_MaxTokensNegative_ReturnsError()
    {
        var request = new CompletionRequest { Prompt = "hello", MaxTokens = -1 };
        var error = RequestValidator.ValidateCompletionRequest(request);
        Assert.Equal("max_tokens must be a positive integer", error);
    }

    [Fact]
    public void ValidateCompletionRequest_Valid_ReturnsNull()
    {
        var request = new CompletionRequest { Prompt = "Hello world" };
        Assert.Null(RequestValidator.ValidateCompletionRequest(request));
    }

    // ── Prompt length validation ──

    [Fact]
    public void ValidatePromptLength_PromptExceedsContext_ReturnsError()
    {
        var tokenizer = new FakeTokenizer(tokenCount: 520);
        var error = RequestValidator.ValidatePromptLength(
            "long prompt", tokenizer, maxSequenceLength: 512,
            requestedMaxTokens: 100, out _, out int promptTokenCount);

        Assert.NotNull(error);
        Assert.Contains("exceeds model context length", error);
        Assert.Equal(520, promptTokenCount);
    }

    [Fact]
    public void ValidatePromptLength_PromptFits_ClampsMaxTokens()
    {
        var tokenizer = new FakeTokenizer(tokenCount: 400);
        var error = RequestValidator.ValidatePromptLength(
            "prompt", tokenizer, maxSequenceLength: 512,
            requestedMaxTokens: 200, out int effectiveMaxTokens, out int promptTokenCount);

        Assert.Null(error);
        Assert.Equal(400, promptTokenCount);
        Assert.Equal(112, effectiveMaxTokens); // 512 - 400 = 112 < 200
    }

    [Fact]
    public void ValidatePromptLength_MaxTokensWithinBudget_NotClamped()
    {
        var tokenizer = new FakeTokenizer(tokenCount: 100);
        var error = RequestValidator.ValidatePromptLength(
            "prompt", tokenizer, maxSequenceLength: 2048,
            requestedMaxTokens: 256, out int effectiveMaxTokens, out _);

        Assert.Null(error);
        Assert.Equal(256, effectiveMaxTokens); // 2048 - 100 = 1948 > 256
    }

    /// <summary>
    /// Minimal tokenizer stub that returns a fixed token count.
    /// </summary>
    private sealed class FakeTokenizer(int tokenCount) : ITokenizer
    {
        public int[] Encode(string text) => new int[tokenCount];
        public string Decode(ReadOnlySpan<int> tokenIds) => "";
        public string DecodeToken(int tokenId) => "";
        public int VocabSize => 32000;
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public int CountTokens(string text) => tokenCount;
    }
}
