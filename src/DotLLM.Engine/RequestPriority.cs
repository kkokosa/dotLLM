namespace DotLLM.Engine;

/// <summary>
/// Priority level for inference requests in the scheduler queue.
/// </summary>
public enum RequestPriority
{
    /// <summary>Low priority — may be preempted.</summary>
    Low,

    /// <summary>Normal priority (default).</summary>
    Normal,

    /// <summary>High priority — preempts normal and low.</summary>
    High,

    /// <summary>Critical priority — never preempted.</summary>
    Critical
}
