namespace DotLLM.Engine.Scheduler;

/// <summary>
/// State machine for a sequence in the continuous-batch scheduler.
/// </summary>
/// <remarks>
/// Transitions:
/// <list type="bullet">
///   <item><c>Queued</c> → <c>Prefilling</c> when admitted by the scheduler.</item>
///   <item><c>Prefilling</c> → <c>Decoding</c> after the prompt prefill completes
///         (or in one step for non-chunked prefill).</item>
///   <item><c>Decoding</c> → <c>Completed</c> when a stop condition fires or the
///         max-token / cache limit is hit.</item>
///   <item><c>Queued</c>/<c>Prefilling</c>/<c>Decoding</c> → <c>Cancelled</c> when
///         the caller cancels the request.</item>
/// </list>
/// </remarks>
public enum SequenceState
{
    /// <summary>Submitted but not yet admitted; no KV-cache allocated.</summary>
    Queued,

    /// <summary>Admitted; prompt prefill in progress.</summary>
    Prefilling,

    /// <summary>Prefill complete; emitting tokens one per scheduler iteration.</summary>
    Decoding,

    /// <summary>Stop condition fired or max-tokens reached. KV-cache may still be held briefly.</summary>
    Completed,

    /// <summary>Cancelled by the caller. KV-cache released, completion observed via TCS cancellation.</summary>
    Cancelled,
}
