using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Telemetry;

namespace DotLLM.Engine;

/// <summary>
/// Thin wrapper that emits dotLLM engine metrics and tracing for a single inference call.
/// All operations no-op when nothing is subscribed — the cost is one branch per call site.
/// </summary>
/// <remarks>
/// Owned by the calling generation method; not thread-safe. The recorder caches the
/// model tag once so per-token sites avoid the architecture-to-string conversion.
/// </remarks>
internal struct TelemetryRecorder
{
    private readonly KeyValuePair<string, object?> _modelTag;
    private readonly bool _metricsEnabled;
    private readonly bool _activitiesEnabled;
    private long _requestStartTicks;
    private bool _firstTokenRecorded;

    public Activity? RequestSpan;

    public TelemetryRecorder(ModelConfig config, InferenceOptions options)
    {
        _metricsEnabled = EngineTelemetry.PrefillTokens.Enabled
                          || EngineTelemetry.DecodeTokens.Enabled
                          || EngineTelemetry.PrefillTokensPerSecond.Enabled
                          || EngineTelemetry.DecodeTokensPerSecond.Enabled
                          || EngineTelemetry.TimeToFirstTokenMs.Enabled;
        _activitiesEnabled = EngineTelemetry.ActivitySource.HasListeners();

        _modelTag = new KeyValuePair<string, object?>(TelemetryTags.Model, config.Architecture.ToString());
        _requestStartTicks = 0;
        _firstTokenRecorded = false;
        RequestSpan = null;

        if (_activitiesEnabled)
        {
            RequestSpan = EngineActivities.StartRequest();
            if (RequestSpan is { IsAllDataRequested: true })
            {
                RequestSpan.SetTag(TelemetryTags.Model, _modelTag.Value);
                RequestSpan.SetTag(TelemetryTags.MaxTokens, options.MaxTokens);
                RequestSpan.SetTag(TelemetryTags.Temperature, options.Temperature);
                RequestSpan.SetTag(TelemetryTags.TopK, options.TopK);
                RequestSpan.SetTag(TelemetryTags.TopP, options.TopP);
            }
        }

        if (_metricsEnabled || _activitiesEnabled)
            _requestStartTicks = Stopwatch.GetTimestamp();
    }

    public bool ActivitiesEnabled => _activitiesEnabled;

    public Activity? StartPrefill()
        => _activitiesEnabled ? EngineActivities.StartPrefill() : null;

    public Activity? StartSample()
        => _activitiesEnabled ? EngineActivities.StartSample() : null;

    public Activity? StartDecodeStep(int step)
        => _activitiesEnabled ? EngineActivities.StartDecodeStep(step) : null;

    /// <summary>
    /// Records first-token latency, called once after the first token leaves the sampler.
    /// </summary>
    public void RecordFirstToken()
    {
        if (_firstTokenRecorded || !_metricsEnabled) return;
        _firstTokenRecorded = true;
        if (EngineTelemetry.TimeToFirstTokenMs.Enabled)
        {
            double ms = (Stopwatch.GetTimestamp() - _requestStartTicks) * 1000.0 / Stopwatch.Frequency;
            EngineTelemetry.TimeToFirstTokenMs.Record(ms, _modelTag);
        }
    }

    /// <summary>
    /// Records final inference counters/histograms and finishes the request span.
    /// Safe to call multiple times — only the first call emits.
    /// </summary>
    public void Complete(int promptTokens, int cachedTokens, int generatedTokens,
        double prefillMs, double decodeMs, FinishReason finishReason)
    {
        if (_metricsEnabled)
        {
            int prefillEffective = promptTokens - cachedTokens;
            if (prefillEffective > 0)
            {
                EngineTelemetry.PrefillTokens.Add(prefillEffective, _modelTag);
                if (prefillMs > 0)
                    EngineTelemetry.PrefillTokensPerSecond.Record(prefillEffective / (prefillMs / 1000.0), _modelTag);
            }
            int decodeEffective = generatedTokens > 1 ? generatedTokens - 1 : 0;
            if (decodeEffective > 0)
            {
                EngineTelemetry.DecodeTokens.Add(decodeEffective, _modelTag);
                if (decodeMs > 0)
                    EngineTelemetry.DecodeTokensPerSecond.Record(decodeEffective / (decodeMs / 1000.0), _modelTag);
            }
        }

        if (RequestSpan is { IsAllDataRequested: true })
        {
            RequestSpan.SetTag(TelemetryTags.PromptTokens, promptTokens);
            RequestSpan.SetTag(TelemetryTags.CachedTokens, cachedTokens);
            RequestSpan.SetTag(TelemetryTags.GeneratedTokens, generatedTokens);
            RequestSpan.SetTag(TelemetryTags.FinishReason, finishReason.ToString());
        }

        RequestSpan?.Dispose();
        RequestSpan = null;
    }
}
