namespace DotLLM.Engine;

/// <summary>
/// Timing measurements captured during inference (prefill, decode, sampling).
/// </summary>
public readonly record struct InferenceTimings
{
    /// <summary>Time spent on the prefill forward pass, in milliseconds.</summary>
    public double PrefillTimeMs { get; init; }

    /// <summary>Time spent on decode forward passes, in milliseconds.</summary>
    public double DecodeTimeMs { get; init; }

    /// <summary>Time spent on sampling, in milliseconds.</summary>
    public double SamplingTimeMs { get; init; }

    /// <summary>Number of prompt tokens processed during prefill.</summary>
    public int PrefillTokenCount { get; init; }

    /// <summary>Number of decode forward passes (generated tokens minus 1, since the first token comes from prefill).</summary>
    public int DecodeTokenCount { get; init; }

    /// <summary>Prefill throughput in tokens per second.</summary>
    public double PrefillTokensPerSec => PrefillTokenCount > 0 && PrefillTimeMs > 0
        ? PrefillTokenCount / (PrefillTimeMs / 1000.0) : 0;

    /// <summary>Decode throughput in tokens per second.</summary>
    public double DecodeTokensPerSec => DecodeTokenCount > 0 && DecodeTimeMs > 0
        ? DecodeTokenCount / (DecodeTimeMs / 1000.0) : 0;
}
