namespace DotLLM.Tokenizers;

/// <summary>
/// A single message in a chat conversation.
/// </summary>
public record ChatMessage
{
    /// <summary>Role of the message sender: "system", "user", "assistant", or "tool".</summary>
    public required string Role { get; init; }

    /// <summary>Text content of the message.</summary>
    public required string Content { get; init; }

    /// <summary>Tool calls made by the assistant. Null if not a tool-calling response.</summary>
    public ToolCall[]? ToolCalls { get; init; }

    /// <summary>ID of the tool call this message is responding to. Null if not a tool result.</summary>
    public string? ToolCallId { get; init; }
}
