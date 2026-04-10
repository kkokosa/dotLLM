namespace DotLLM.Core.Configuration;

/// <summary>
/// Configuration for CPU thread parallelism during inference.
/// </summary>
/// <param name="ThreadCount">
/// Number of threads. 1 = single-threaded (default), 0 = <see cref="Environment.ProcessorCount"/>.
/// </param>
/// <param name="DecodeThreadCount">
/// Number of threads for decode phase. 0 = auto (memory channel heuristic).
/// Decode is memory-bandwidth-bound, so more threads than memory channels don't help.
/// </param>
/// <param name="EnableNumaPinning">
/// Pin worker threads to NUMA-local cores on multi-socket systems.
/// </param>
/// <param name="EnablePCorePinning">
/// Pin worker threads to P-cores only on Intel hybrid (P-core/E-core) architectures.
/// </param>
/// <param name="EnableCallerPinning">
/// Pin the caller thread (the inference thread that invokes <c>Dispatch</c>) to the first
/// candidate core when any pinning is active. Default <c>true</c> — a no-op when no other
/// pinning is configured. Prevents the scenario where pinned P-core workers idle at the
/// barrier waiting for an unpinned caller that the OS scheduled onto an E-core.
/// </param>
public readonly record struct ThreadingConfig(
    int ThreadCount = 1,
    int DecodeThreadCount = 0,
    bool EnableNumaPinning = false,
    bool EnablePCorePinning = false,
    bool EnableCallerPinning = true)
{
    /// <summary>Single-threaded configuration (no thread pool overhead).</summary>
    public static ThreadingConfig SingleThreaded => new(1);

    /// <summary>Use all available processors.</summary>
    public static ThreadingConfig Auto => new(0);

    /// <summary>Resolved thread count, clamped to at least 1.</summary>
    public int EffectiveThreadCount => Math.Max(1, ThreadCount == 0 ? Environment.ProcessorCount : ThreadCount);

    /// <summary>Whether parallelism is enabled (more than one thread).</summary>
    public bool IsParallel => EffectiveThreadCount > 1;

    /// <summary>
    /// Resolves the effective decode thread count.
    /// Uses explicit <see cref="DecodeThreadCount"/> if set, otherwise caps at the
    /// estimated memory channel count (since decode is bandwidth-bound).
    /// </summary>
    /// <param name="memoryChannelEstimate">Heuristic memory channel count from NUMA topology.</param>
    /// <returns>Effective decode thread count, clamped to [2, EffectiveThreadCount].</returns>
    public int EffectiveDecodeThreadCount(int memoryChannelEstimate)
        => DecodeThreadCount > 0
            ? Math.Min(DecodeThreadCount, EffectiveThreadCount)
            : Math.Min(EffectiveThreadCount, Math.Max(2, memoryChannelEstimate));
}
