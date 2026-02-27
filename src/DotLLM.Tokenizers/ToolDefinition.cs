namespace DotLLM.Tokenizers;

/// <summary>
/// Defines a tool/function that the model can call.
/// </summary>
/// <param name="Name">Function name.</param>
/// <param name="Description">Human-readable description of what the function does.</param>
/// <param name="ParametersSchema">JSON Schema describing the function parameters.</param>
public record ToolDefinition(string Name, string Description, string ParametersSchema);
