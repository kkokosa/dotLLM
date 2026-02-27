namespace DotLLM.Tokenizers;

/// <summary>
/// Applies a chat template to format conversation messages into a prompt string.
/// Implementations interpret Jinja2-subset templates from model metadata.
/// </summary>
public interface IChatTemplate
{
    /// <summary>
    /// Applies the template to a list of chat messages.
    /// </summary>
    /// <param name="messages">Conversation messages in order.</param>
    /// <param name="options">Template options (generation prompt, tools).</param>
    /// <returns>Formatted prompt string ready for tokenization.</returns>
    string Apply(IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options);
}
