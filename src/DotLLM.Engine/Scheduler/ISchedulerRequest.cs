using DotLLM.Core.Attention;

namespace DotLLM.Engine.Scheduler;

/// <summary>
/// An in-flight sequence tracked by an <see cref="IBatchScheduler"/>.
/// Encapsulates the per-sequence state the scheduler needs across iterations:
/// prompt, generated tokens, position cursor, KV-cache handle, and the awaitable
/// that completes when the sequence finishes (or fails).
/// </summary>
/// <remarks>
/// <para>This is the public seam: callers see <see cref="ISchedulerRequest"/>; the
/// scheduler implementation owns the concrete <c>SchedulerRequest</c> and casts back
/// when it needs mutation. Keeping the public surface read-only prevents callers
/// from corrupting scheduler invariants between iterations.</para>
/// <para>Streaming is not part of this MVP — the scheduler accumulates generated
/// tokens and exposes them only when the sequence completes via the awaitable.
/// A future iteration may add a per-request <c>ChannelWriter&lt;GenerationToken&gt;</c>.</para>
/// </remarks>
public interface ISchedulerRequest
{
    /// <summary>The original request submitted by the caller.</summary>
    InferenceRequest Request { get; }

    /// <summary>Current state in the sequence lifecycle.</summary>
    SequenceState State { get; }

    /// <summary>Number of prompt tokens (set at submission time, immutable after).</summary>
    int PromptLength { get; }

    /// <summary>Number of tokens generated so far (excluding the prompt).</summary>
    int GeneratedCount { get; }

    /// <summary>Logical position of the next token to be appended (= prompt length + generated count).</summary>
    int Position { get; }

    /// <summary>
    /// KV-cache handle owned by this sequence. Null until the scheduler admits the request
    /// and allocates its cache. Released to the pool when the sequence completes.
    /// </summary>
    IKvCache? KvCache { get; }

    /// <summary>Task that completes with the inference response when the sequence finishes.</summary>
    Task<InferenceResponse> Completion { get; }
}
