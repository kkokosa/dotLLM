namespace DotLLM.Engine.Scheduler;

/// <summary>
/// Configuration for <see cref="ContinuousBatchScheduler"/>.
/// </summary>
public sealed record ContinuousBatchSchedulerOptions
{
    /// <summary>
    /// Maximum number of sequences that may be actively prefilling or decoding at once.
    /// Acts as a soft upper bound — KV-cache capacity is the hard constraint.
    /// Default 64 matches small-server throughput; servers with large pools can raise this.
    /// </summary>
    public int MaxActiveSequences { get; init; } = 64;

    /// <summary>
    /// Maximum number of prompt tokens admitted (across all newly-admitted sequences) in
    /// a single scheduler iteration. Bounds the worst-case prefill latency before a decode
    /// iteration is allowed to run. Set to 0 to disable the bound.
    /// </summary>
    /// <remarks>
    /// MVP: this is a per-iteration admission cap, not chunked prefill. A long prompt that
    /// exceeds the cap simply waits for a later iteration to be admitted in full. Chunked
    /// prefill (splitting one sequence's prefill across iterations) is a future enhancement.
    /// </remarks>
    public int MaxPrefillTokensPerStep { get; init; } = 0;

    /// <summary>
    /// Optional cap on KV-cache blocks reserved for in-flight sequences. When non-zero,
    /// admission is gated on (free blocks ≥ blocks-required) so a single oversize prompt
    /// can't drain the pool.
    /// </summary>
    /// <remarks>
    /// MVP gates only on raw free-block count; the pool itself is the source of truth via
    /// <see cref="DotLLM.Engine.KvCache.KvBlockPool.FreeBlocks"/>. <see cref="ContinuousBatchScheduler"/>
    /// works with non-paged caches too — in that case, admission is governed by
    /// <see cref="MaxActiveSequences"/> alone.
    /// </remarks>
    public int ReserveBlocksPerSequence { get; init; } = 0;
}
