namespace DotLLM.Core.Sampling;

/// <summary>
/// Result of a stop condition check.
/// </summary>
public enum StopResult
{
    /// <summary>Generation should continue.</summary>
    Continue,

    /// <summary>Generation should stop. The triggering token is excluded from output.</summary>
    Stop,

    /// <summary>Generation should stop. The triggering token is included in output.</summary>
    StopInclude
}
