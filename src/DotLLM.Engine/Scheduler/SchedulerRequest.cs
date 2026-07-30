using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Constraints;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;

namespace DotLLM.Engine.Scheduler;

/// <summary>
/// Concrete in-flight sequence state owned by <see cref="ContinuousBatchScheduler"/>.
/// </summary>
/// <remarks>
/// This type holds mutable scheduler state — generated tokens, KV-cache handle,
/// sampler pipeline, stop conditions, completion source. Callers see this through
/// <see cref="ISchedulerRequest"/> which exposes only safe read-only properties.
/// </remarks>
internal sealed class SchedulerRequest : ISchedulerRequest
{
    public InferenceRequest Request { get; }
    public InferenceOptionsLike Options { get; }
    public SequenceState State { get; set; }
    public int PromptLength { get; }

    /// <summary>Maximum tokens to generate (already clamped to model context length).</summary>
    public int MaxTokens { get; }

    /// <summary>The prompt token IDs. Borrowed from the request — never mutated.</summary>
    public int[] PromptTokenIds => Request.TokenIds;

    /// <summary>Tokens generated since prefill ended. Lazily allocated.</summary>
    public List<int> GeneratedTokens { get; } = new();

    public int GeneratedCount => GeneratedTokens.Count;
    public int Position => PromptLength + GeneratedCount;

    /// <summary>KV-cache assigned at admission time. Released when the sequence is evicted.</summary>
    public IKvCache? KvCache { get; set; }

    /// <summary>Sampler pipeline built from the request's <c>InferenceOptions</c>.</summary>
    public SamplerPipeline SamplerPipeline { get; }

    /// <summary>Stop conditions (EOS + max-tokens + user-supplied stop strings).</summary>
    public IReadOnlyList<IStopCondition> StopConditions { get; }

    /// <summary>Optional decoding constraint for structured output (JSON / schema / regex / grammar).</summary>
    public IDecodingConstraint? Constraint { get; }

    /// <summary>Reason this sequence stopped (set when transitioning to <see cref="SequenceState.Completed"/>).</summary>
    public FinishReason FinishReason { get; set; } = FinishReason.Length;

    /// <summary>Wall-clock prefill ticks (set during admission's prefill pass).</summary>
    public long PrefillTicks { get; set; }

    /// <summary>Cumulative decode forward-pass ticks across all iterations.</summary>
    public long DecodeTicks { get; set; }

    /// <summary>Cumulative sampling ticks across all iterations.</summary>
    public long SamplerTicks { get; set; }

    /// <summary>Completion source resolved when the sequence finishes.</summary>
    public TaskCompletionSource<InferenceResponse> CompletionSource { get; }

    public Task<InferenceResponse> Completion => CompletionSource.Task;

    /// <summary>Cancellation registration on the caller's token; disposed on completion.</summary>
    public CancellationTokenRegistration CancellationRegistration { get; set; }

    /// <summary>Monotonic submission counter used for FIFO tie-breaking among same-priority requests.</summary>
    public long SubmissionOrder { get; }

    public SchedulerRequest(
        InferenceRequest request,
        InferenceOptionsLike options,
        int promptLength,
        int maxTokens,
        SamplerPipeline samplerPipeline,
        IReadOnlyList<IStopCondition> stopConditions,
        IDecodingConstraint? constraint,
        long submissionOrder,
        TaskCompletionSource<InferenceResponse> tcs)
    {
        Debug.Assert(request != null);
        Debug.Assert(promptLength > 0);
        Debug.Assert(maxTokens > 0);

        Request = request;
        Options = options;
        PromptLength = promptLength;
        MaxTokens = maxTokens;
        SamplerPipeline = samplerPipeline;
        StopConditions = stopConditions;
        Constraint = constraint;
        SubmissionOrder = submissionOrder;
        CompletionSource = tcs;
        State = SequenceState.Queued;
    }
}

/// <summary>
/// Minimal projection of <see cref="DotLLM.Core.Configuration.InferenceOptions"/> the scheduler
/// needs after pipeline construction. Avoids carrying the full options record per sequence.
/// </summary>
internal readonly record struct InferenceOptionsLike(int MaxTokens, bool Logprobs, int TopLogprobs);
