using System.Diagnostics;

namespace DotLLM.Telemetry;

/// <summary>
/// Helpers that start the dotLLM engine's tracing spans with consistent names.
/// Every method returns <c>null</c> when no <see cref="ActivityListener"/> is
/// subscribed — the calling code stays branch-free apart from a single null check.
/// </summary>
/// <remarks>
/// Span hierarchy:
/// <code>
/// dotllm.request           (root — per Generate / GenerateStreamingTokensAsync call)
///  ├── dotllm.prefill      (one per call; tokens, duration_ms)
///  ├── dotllm.sample       (sampler pipeline)
///  └── dotllm.decode_step  (sampled — per-decode step, only when sampling enabled)
/// </code>
/// </remarks>
public static class EngineActivities
{
    /// <summary>Root request span — covers prompt encode through final token.</summary>
    public const string Request = "dotllm.request";

    /// <summary>Prefill forward pass span.</summary>
    public const string Prefill = "dotllm.prefill";

    /// <summary>Sampler pipeline span. Created per token when verbose decode-step tracing is enabled.</summary>
    public const string Sample = "dotllm.sample";

    /// <summary>Per-decode-step span. Sampled (see <see cref="DecodeStepSamplePermille"/>).</summary>
    public const string DecodeStep = "dotllm.decode_step";

    /// <summary>
    /// Decode-step sampling rate in per-mille (1/1000). The default 10 ‰ matches the spec
    /// "sample 1%" guidance — only ~1% of decode steps will produce a span when an
    /// <see cref="ActivityListener"/> is attached, keeping the trace volume bounded for
    /// long generations.
    /// </summary>
    public const int DecodeStepSamplePermille = 10;

    /// <summary>Starts the root request span. Returns null when no listener is attached.</summary>
    public static Activity? StartRequest()
        => EngineTelemetry.ActivitySource.StartActivity(Request, ActivityKind.Internal);

    /// <summary>Starts the prefill child span. Returns null when no listener is attached.</summary>
    public static Activity? StartPrefill()
        => EngineTelemetry.ActivitySource.StartActivity(Prefill, ActivityKind.Internal);

    /// <summary>Starts the sampler-pipeline child span. Returns null when no listener is attached.</summary>
    public static Activity? StartSample()
        => EngineTelemetry.ActivitySource.StartActivity(Sample, ActivityKind.Internal);

    /// <summary>
    /// Starts a decode-step child span — but only when an <see cref="ActivityListener"/> is
    /// attached and the step falls inside the configured sample rate. Returns null otherwise.
    /// </summary>
    /// <param name="step">Zero-based decode step index used for deterministic sampling.</param>
    public static Activity? StartDecodeStep(int step)
    {
        if (!EngineTelemetry.ActivitySource.HasListeners())
            return null;
        // Deterministic: every (1000 / permille)-th step gets a span — keeps trace volume bounded
        // for long generations regardless of how many listeners are attached.
        if (step % (1000 / DecodeStepSamplePermille) != 0)
            return null;
        return EngineTelemetry.ActivitySource.StartActivity(DecodeStep, ActivityKind.Internal);
    }
}
