using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Reflection;

namespace DotLLM.Telemetry;

/// <summary>
/// Central <see cref="Meter"/> and <see cref="ActivitySource"/> for dotLLM engine instrumentation.
/// <para>
/// Naming follows the OpenTelemetry semantic conventions — both surfaces are named
/// <c>DotLLM.Engine</c>. Instrument names use the <c>dotllm.engine.*</c> namespace.
/// </para>
/// <para>
/// All instruments are static so the hot path is a single <c>Instrument.Enabled</c> /
/// <see cref="ActivitySource.HasListeners"/> branch when nothing is subscribed.
/// </para>
/// </summary>
public static class EngineTelemetry
{
    /// <summary>OpenTelemetry-friendly name for both the <see cref="Meter"/> and <see cref="ActivitySource"/>.</summary>
    public const string Name = "DotLLM.Engine";

    private static readonly string s_version =
        typeof(EngineTelemetry).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? typeof(EngineTelemetry).Assembly.GetName().Version?.ToString()
        ?? "0.0.0";

    /// <summary>Shared <see cref="Meter"/> for all engine instruments.</summary>
    public static readonly Meter Meter = new(Name, s_version);

    /// <summary>Shared <see cref="ActivitySource"/> for engine spans.</summary>
    public static readonly ActivitySource ActivitySource = new(Name, s_version);

    /// <summary>Counter — prompt tokens processed during prefill, tagged by <c>model</c>.</summary>
    public static readonly Counter<long> PrefillTokens = Meter.CreateCounter<long>(
        "dotllm.engine.tokens.prefill", unit: "tokens",
        description: "Prompt tokens processed during prefill (excludes prefix-cache hits).");

    /// <summary>Counter — tokens generated during decode, tagged by <c>model</c>.</summary>
    public static readonly Counter<long> DecodeTokens = Meter.CreateCounter<long>(
        "dotllm.engine.tokens.decode", unit: "tokens",
        description: "Tokens generated during the decode phase.");

    /// <summary>Histogram — prefill throughput in tokens/second.</summary>
    public static readonly Histogram<double> PrefillTokensPerSecond = Meter.CreateHistogram<double>(
        "dotllm.engine.tokens_per_second.prefill", unit: "tokens/s",
        description: "Prefill throughput in tokens per second.");

    /// <summary>Histogram — decode throughput in tokens/second.</summary>
    public static readonly Histogram<double> DecodeTokensPerSecond = Meter.CreateHistogram<double>(
        "dotllm.engine.tokens_per_second.decode", unit: "tokens/s",
        description: "Decode throughput in tokens per second.");

    /// <summary>Histogram — time to first token in milliseconds.</summary>
    public static readonly Histogram<double> TimeToFirstTokenMs = Meter.CreateHistogram<double>(
        "dotllm.engine.time_to_first_token_ms", unit: "ms",
        description: "Time from request start to first generated token.");

    private static Func<long>? s_queueDepthProvider;
    private static Func<double>? s_kvCacheUtilizationProvider;

    static EngineTelemetry()
    {
        _ = Meter.CreateObservableGauge<long>(
            "dotllm.engine.request.queue_depth", ObserveQueueDepth,
            unit: "{request}",
            description: "Requests waiting for admission in the scheduler queue. -1 until the scheduler (Step 35) is wired.");

        _ = Meter.CreateObservableGauge<double>(
            "dotllm.engine.kvcache.utilization", ObserveKvCacheUtilization,
            unit: "1",
            description: "Fraction of paged KV-cache blocks in use (0..1). -1 until paged scheduler (Step 35) is wired.");
    }

    /// <summary>
    /// Registers a callback that returns the current scheduler queue depth.
    /// Without a callback the observable gauge emits <c>-1</c> as a sentinel.
    /// </summary>
    public static void SetQueueDepthProvider(Func<long>? provider) => s_queueDepthProvider = provider;

    /// <summary>
    /// Registers a callback returning paged KV-cache utilization in the range [0, 1].
    /// Without a callback the observable gauge emits <c>-1</c> as a sentinel.
    /// </summary>
    public static void SetKvCacheUtilizationProvider(Func<double>? provider) => s_kvCacheUtilizationProvider = provider;

    private static long ObserveQueueDepth()
    {
        var provider = s_queueDepthProvider;
        // TODO(step-35): wire to IScheduler.QueueDepth once continuous batching lands.
        return provider is null ? -1L : provider();
    }

    private static double ObserveKvCacheUtilization()
    {
        var provider = s_kvCacheUtilizationProvider;
        // TODO(step-35): wire to paged KV allocator usage once continuous batching lands.
        return provider is null ? -1.0 : provider();
    }
}
