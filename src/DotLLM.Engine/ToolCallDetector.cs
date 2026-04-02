using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Post-generation tool call detection. Parses generated text using an <see cref="IToolCallParser"/>
/// and enriches the <see cref="InferenceResponse"/> with tool call data.
/// </summary>
public static class ToolCallDetector
{
    /// <summary>
    /// Checks generated text for tool calls and returns an enriched response.
    /// </summary>
    /// <param name="response">The original inference response.</param>
    /// <param name="parser">The model-specific tool call parser.</param>
    /// <returns>
    /// A new response with <see cref="InferenceResponse.ToolCalls"/> populated
    /// and <see cref="InferenceResponse.FinishReason"/> set to <see cref="FinishReason.ToolCalls"/>
    /// if tool calls were detected. Otherwise, the original response unchanged.
    /// </returns>
    public static InferenceResponse DetectToolCalls(InferenceResponse response, IToolCallParser parser)
    {
        if (string.IsNullOrEmpty(response.Text))
            return response;

        var toolCalls = parser.TryParse(response.Text);
        if (toolCalls is not { Length: > 0 })
            return response;

        return response with
        {
            ToolCalls = toolCalls,
            FinishReason = FinishReason.ToolCalls
        };
    }
}
