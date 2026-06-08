namespace DotLLM.Engine.Scheduler;

/// <summary>
/// Iteration-level scheduler for continuous batching.
/// </summary>
/// <remarks>
/// <para>This is the synchronous lower-level seam used by the engine and by tests.
/// The async <see cref="IScheduler"/> interface wraps this with a queue and a
/// <see cref="IScheduler.RunLoopAsync(System.Threading.CancellationToken)"/> driver
/// task for ASP.NET server integration.</para>
/// <para>Each <see cref="Step"/> call performs one iteration:</para>
/// <list type="number">
///   <item>Admit newly-submitted sequences subject to KV-cache capacity, running prompt prefill.</item>
///   <item>Run one decode forward per active (decoding) sequence; sample + apply stop conditions.</item>
///   <item>Evict completed sequences and release their KV-cache blocks.</item>
/// </list>
/// <para>The scheduler is intentionally backend-agnostic — it drives the model through
/// the <see cref="DotLLM.Core.Models.IModel.Forward(System.ReadOnlySpan{int},System.ReadOnlySpan{int},int,DotLLM.Core.Attention.IKvCache?)"/>
/// API and uses the supplied KV-cache factory, so CPU/CUDA/Vulkan all work uniformly.</para>
/// </remarks>
public interface IBatchScheduler
{
    /// <summary>
    /// Submits a new request. The returned object becomes <see cref="SequenceState.Queued"/>
    /// and will be admitted on a future <see cref="Step"/> call when KV-cache capacity allows.
    /// </summary>
    /// <param name="request">The inference request.</param>
    /// <param name="cancellationToken">Optional cancellation token. If signalled before the
    /// sequence completes, the scheduler evicts it and propagates cancellation via
    /// <see cref="ISchedulerRequest.Completion"/>.</param>
    /// <returns>Handle for the submitted sequence; await <see cref="ISchedulerRequest.Completion"/>
    /// for the result.</returns>
    ISchedulerRequest Submit(InferenceRequest request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Runs one scheduling iteration. Safe to call when idle (returns immediately).
    /// </summary>
    /// <returns><see langword="true"/> if any work was performed (admission, prefill, decode,
    /// or eviction); <see langword="false"/> if the scheduler is idle.</returns>
    bool Step();

    /// <summary>
    /// True when there are no queued or active sequences.
    /// </summary>
    bool IsIdle { get; }

    /// <summary>Number of sequences currently being processed (prefilling or decoding).</summary>
    int ActiveCount { get; }

    /// <summary>Number of requests queued and awaiting admission.</summary>
    int QueueDepth { get; }

    /// <summary>Returns a snapshot of scheduler counters.</summary>
    SchedulerMetrics GetMetrics();
}
