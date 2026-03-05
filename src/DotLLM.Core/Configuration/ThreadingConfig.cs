namespace DotLLM.Core.Configuration;

/// <summary>
/// Configuration for CPU thread parallelism during inference.
/// </summary>
/// <param name="ThreadCount">
/// Number of threads. 1 = single-threaded (default), 0 = <see cref="Environment.ProcessorCount"/>.
/// </param>
public readonly record struct ThreadingConfig(int ThreadCount = 1)
{
    /// <summary>Single-threaded configuration (no thread pool overhead).</summary>
    public static ThreadingConfig SingleThreaded => new(1);

    /// <summary>Use all available processors.</summary>
    public static ThreadingConfig Auto => new(0);

    /// <summary>Resolved thread count, clamped to at least 1.</summary>
    public int EffectiveThreadCount => Math.Max(1, ThreadCount == 0 ? Environment.ProcessorCount : ThreadCount);

    /// <summary>Whether parallelism is enabled (more than one thread).</summary>
    public bool IsParallel => EffectiveThreadCount > 1;
}
