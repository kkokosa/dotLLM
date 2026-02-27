namespace DotLLM.Core.Telemetry;

/// <summary>
/// Records inference performance metrics via <c>System.Diagnostics.Metrics</c>.
/// Zero overhead when no metric listener is attached.
/// </summary>
public interface IInferenceMetrics
{
    /// <summary>Records decode (generation) throughput.</summary>
    /// <param name="tokensPerSecond">Tokens generated per second.</param>
    void RecordDecodeTokensPerSecond(double tokensPerSecond);

    /// <summary>Records prefill (prompt processing) throughput.</summary>
    /// <param name="tokensPerSecond">Prompt tokens processed per second.</param>
    void RecordPrefillTokensPerSecond(double tokensPerSecond);

    /// <summary>Increments the completed request counter.</summary>
    void IncrementRequestsCompleted();

    /// <summary>Increments the failed request counter.</summary>
    void IncrementRequestsFailed();

    /// <summary>Records time-to-first-token latency.</summary>
    /// <param name="latencyMs">Latency in milliseconds.</param>
    void RecordTimeToFirstToken(double latencyMs);

    /// <summary>Records current active sequence count.</summary>
    /// <param name="count">Number of active sequences.</param>
    void RecordActiveSequences(int count);

    /// <summary>Records current KV-cache utilization.</summary>
    /// <param name="utilizationPercent">Cache utilization as a percentage (0-100).</param>
    void RecordKvCacheUtilization(double utilizationPercent);
}
