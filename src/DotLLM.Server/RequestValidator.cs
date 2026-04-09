using DotLLM.Server.Models;
using DotLLM.Tokenizers;

namespace DotLLM.Server;

/// <summary>
/// Validates incoming inference requests and clamps parameters to model limits.
/// </summary>
public static class RequestValidator
{
    /// <summary>Maximum number of messages allowed in a chat completion request.</summary>
    public const int MaxMessages = 1024;

    /// <summary>
    /// Validates a chat completion request before inference.
    /// Returns an error message if invalid, or null if valid.
    /// </summary>
    public static string? ValidateChatRequest(ChatCompletionRequest request)
    {
        if (request.Messages is null || request.Messages.Length == 0)
            return "messages array must not be empty";

        if (request.Messages.Length > MaxMessages)
            return $"messages array exceeds maximum of {MaxMessages}";

        if (request.MaxTokens.HasValue && request.MaxTokens.Value <= 0)
            return "max_tokens must be a positive integer";

        return null;
    }

    /// <summary>
    /// Validates a raw completion request before inference.
    /// Returns an error message if invalid, or null if valid.
    /// </summary>
    public static string? ValidateCompletionRequest(CompletionRequest request)
    {
        if (string.IsNullOrEmpty(request.Prompt))
            return "prompt must not be empty";

        if (request.MaxTokens.HasValue && request.MaxTokens.Value <= 0)
            return "max_tokens must be a positive integer";

        return null;
    }

    /// <summary>
    /// Validates prompt length against the model's context window and clamps max_tokens.
    /// Returns an error message if the prompt alone exceeds context, or null if valid.
    /// When valid, <paramref name="effectiveMaxTokens"/> is clamped to remaining context.
    /// </summary>
    public static string? ValidatePromptLength(
        string prompt, ITokenizer tokenizer, int maxSequenceLength,
        int requestedMaxTokens, out int effectiveMaxTokens, out int promptTokenCount)
    {
        // TODO: BpeTokenizer.CountTokens currently delegates to Encode().Length (same allocation).
        // When a streaming/counting-only tokenizer path is added, this will benefit automatically.
        promptTokenCount = tokenizer.CountTokens(prompt);

        if (promptTokenCount >= maxSequenceLength)
        {
            effectiveMaxTokens = 0;
            return $"prompt ({promptTokenCount} tokens) exceeds model context length ({maxSequenceLength})";
        }

        int remaining = maxSequenceLength - promptTokenCount;
        effectiveMaxTokens = Math.Min(requestedMaxTokens, remaining);
        return null;
    }
}
