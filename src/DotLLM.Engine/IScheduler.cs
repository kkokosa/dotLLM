namespace DotLLM.Engine;

/// <summary>
/// Manages the inference request queue, continuous batching, and scheduling loop.
/// </summary>
public interface IScheduler
{
    /// <summary>
    /// Enqueues an inference request and returns a task that completes with the response.
    /// </summary>
    /// <param name="request">The inference request.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The inference response when generation completes.</returns>
    Task<InferenceResponse> EnqueueAsync(InferenceRequest request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Runs the continuous batching loop. Call once at startup; runs until cancelled.
    /// </summary>
    /// <param name="cancellationToken">Cancellation token to stop the loop.</param>
    Task RunLoopAsync(CancellationToken cancellationToken);

    /// <summary>
    /// Returns a snapshot of current scheduler metrics.
    /// </summary>
    SchedulerMetrics GetMetrics();
}
