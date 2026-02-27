namespace DotLLM.Engine;

/// <summary>
/// Snapshot of scheduler state for monitoring and observability.
/// </summary>
/// <param name="ActiveSequences">Number of sequences currently being processed.</param>
/// <param name="QueueDepth">Number of requests waiting in the queue.</param>
/// <param name="PreemptionCount">Total number of preemptions since startup.</param>
public readonly record struct SchedulerMetrics(
    int ActiveSequences,
    int QueueDepth,
    long PreemptionCount);
