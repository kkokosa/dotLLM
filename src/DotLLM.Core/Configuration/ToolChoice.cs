namespace DotLLM.Core.Configuration;

/// <summary>
/// Specifies how the model should select tool calls.
/// Follows the OpenAI API convention for <c>tool_choice</c>.
/// </summary>
public abstract record ToolChoice
{
    /// <summary>Model decides whether to call tools or generate text (default when tools are present).</summary>
    public sealed record Auto : ToolChoice;

    /// <summary>Model must not call any tools — generate text only.</summary>
    public sealed record None : ToolChoice;

    /// <summary>Model must call at least one tool.</summary>
    public sealed record Required : ToolChoice;

    /// <summary>Model must call the specified function.</summary>
    /// <param name="Name">The function name to force.</param>
    public sealed record Function(string Name) : ToolChoice;
}
