namespace DotLLM.Engine;

/// <summary>
/// Reason why token generation stopped.
/// </summary>
public enum FinishReason
{
    /// <summary>A stop condition was met (EOS token, stop sequence).</summary>
    Stop,

    /// <summary>Maximum token limit was reached.</summary>
    Length,

    /// <summary>Generation stopped to execute tool calls.</summary>
    ToolCalls
}
