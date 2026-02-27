namespace DotLLM.Tokenizers;

/// <summary>
/// Represents a tool/function call extracted from model output.
/// </summary>
/// <param name="Id">Unique identifier for this tool call.</param>
/// <param name="FunctionName">Name of the function to invoke.</param>
/// <param name="Arguments">JSON string of function arguments.</param>
public record ToolCall(string Id, string FunctionName, string Arguments);
