namespace DotLLM.Telemetry;

/// <summary>
/// Tag keys for engine metrics and tracing spans.
/// </summary>
public static class TelemetryTags
{
    /// <summary>Model identifier (architecture name).</summary>
    public const string Model = "model";

    /// <summary>Prompt token count.</summary>
    public const string PromptTokens = "dotllm.prompt_tokens";

    /// <summary>Maximum tokens requested for generation.</summary>
    public const string MaxTokens = "dotllm.max_tokens";

    /// <summary>Tokens actually generated.</summary>
    public const string GeneratedTokens = "dotllm.generated_tokens";

    /// <summary>Tokens served from prefix cache.</summary>
    public const string CachedTokens = "dotllm.cached_tokens";

    /// <summary>Sampler temperature.</summary>
    public const string Temperature = "dotllm.sampler.temperature";

    /// <summary>Top-k sampler value.</summary>
    public const string TopK = "dotllm.sampler.top_k";

    /// <summary>Top-p sampler value.</summary>
    public const string TopP = "dotllm.sampler.top_p";

    /// <summary>Finish reason string (stop, length, tool_calls).</summary>
    public const string FinishReason = "dotllm.finish_reason";

    /// <summary>Tokens processed during this prefill span.</summary>
    public const string PrefillTokenCount = "dotllm.prefill.token_count";

    /// <summary>Prefill duration in milliseconds.</summary>
    public const string PrefillDurationMs = "dotllm.prefill.duration_ms";
}
