namespace DotLLM.Tokenizers;

/// <summary>
/// Options for applying a chat template.
/// </summary>
public record ChatTemplateOptions
{
    /// <summary>Whether to append the assistant turn prefix for generation.</summary>
    public bool AddGenerationPrompt { get; init; } = true;

    /// <summary>Tool definitions available to the model. Null if tool calling is not enabled.</summary>
    public ToolDefinition[]? Tools { get; init; }
}
