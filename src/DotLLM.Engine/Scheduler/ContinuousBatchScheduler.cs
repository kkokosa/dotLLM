using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Constraints;
using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Constraints;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Engine.Samplers;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Scheduler;

/// <summary>
/// Continuous-batching scheduler — MVP implementation.
/// </summary>
/// <remarks>
/// <para>This MVP performs <em>iteration-level</em> scheduling: every <see cref="Step"/> processes
/// admissions and one decode token across all active sequences. The KV-cache pool is shared
/// across sequences (via a caller-supplied factory such as <see cref="PagedKvCacheFactory"/>),
/// so sequences can be admitted as soon as blocks become free.</para>
///
/// <para><b>What this MVP does NOT do</b> (deferred to follow-on roadmap steps):</para>
/// <list type="bullet">
///   <item>Kernel-level batched forward across sequences. We currently call
///   <see cref="IModel.Forward(System.ReadOnlySpan{int},System.ReadOnlySpan{int},int,IKvCache?)"/>
///   once per sequence per iteration. The model still benefits from KV-cache pooling and
///   request-admission overlap, but per-sequence GEMV is not yet batched. Step 59
///   (advanced scheduling) is the right place for that.</item>
///   <item>Chunked prefill. A long prompt is prefilled in a single forward pass during admission.
///   See <see cref="ContinuousBatchSchedulerOptions.MaxPrefillTokensPerStep"/> for the partial
///   admission cap that mitigates head-of-line blocking at admission time only.</item>
///   <item>Preemption / swap. Sequences run to completion once admitted. Step 59 will add
///   priority-based preemption for VRAM-constrained workloads.</item>
///   <item>Streaming yield. Generated tokens accumulate inside the scheduler and surface only
///   when <see cref="ISchedulerRequest.Completion"/> resolves. Streaming through a
///   <c>ChannelWriter&lt;GenerationToken&gt;</c> per request is straightforward to add but
///   out of scope for the MVP.</item>
/// </list>
///
/// <para>Thread-safety: <see cref="Submit"/> may be called from any thread (queue is
/// <see cref="ConcurrentQueue{T}"/>). <see cref="Step"/> must be driven by a single thread —
/// typically the server's run-loop task.</para>
/// </remarks>
public sealed class ContinuousBatchScheduler : IBatchScheduler, IDisposable
{
    private readonly IModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly Func<ModelConfig, int, IKvCache> _kvCacheFactory;
    private readonly ContinuousBatchSchedulerOptions _options;
    private readonly KvBlockPool? _pagedPool;
    private readonly PrefixTrieManager? _prefixCache;
    private long _cachedPromptTokens;
    private long _prefilledPromptTokens;

    private readonly ConcurrentQueue<SchedulerRequest> _pendingQueue = new();
    private readonly List<SchedulerRequest> _active = new();

    private long _submissionCounter;
    // Preemption is intentionally not implemented in the MVP (deferred to Step 59 / advanced
    // scheduling). The counter exists so the metrics surface and the eventual implementation
    // can plug in without an API change.
#pragma warning disable CS0649
    private long _preemptionCount;
#pragma warning restore CS0649
    private bool _disposed;

    /// <summary>Number of sequences ever submitted to this scheduler.</summary>
    public long TotalSubmitted => Interlocked.Read(ref _submissionCounter);

    /// <inheritdoc/>
    public int ActiveCount => _active.Count;

    /// <inheritdoc/>
    public int QueueDepth => _pendingQueue.Count;

    /// <inheritdoc/>
    public bool IsIdle => _active.Count == 0 && _pendingQueue.IsEmpty;

    /// <summary>
    /// Creates a new continuous-batch scheduler.
    /// </summary>
    /// <param name="model">The transformer model to run. Forward is invoked once per active
    /// sequence per <see cref="Step"/>.</param>
    /// <param name="tokenizer">Tokenizer for decoding generated tokens into the final response text.</param>
    /// <param name="kvCacheFactory">Factory returning a fresh per-sequence KV-cache. For paged
    /// caching, pass <see cref="PagedKvCacheFactory.Create(int)"/> wrapped in a delegate.</param>
    /// <param name="options">Optional scheduler options.</param>
    /// <param name="pagedPool">Optional reference to the underlying paged-block pool. When provided,
    /// admission uses pool free-block count for capacity gating in addition to
    /// <see cref="ContinuousBatchSchedulerOptions.MaxActiveSequences"/>.</param>
    /// <param name="prefixCache">Optional cross-request prefix cache. When provided, admission
    /// seeds new KV-caches from the trie and routes completions back to it (Step 37).</param>
    public ContinuousBatchScheduler(
        IModel model,
        ITokenizer tokenizer,
        Func<ModelConfig, int, IKvCache> kvCacheFactory,
        ContinuousBatchSchedulerOptions? options = null,
        KvBlockPool? pagedPool = null,
        PrefixTrieManager? prefixCache = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(kvCacheFactory);
        _model = model;
        _tokenizer = tokenizer;
        _kvCacheFactory = kvCacheFactory;
        _options = options ?? new ContinuousBatchSchedulerOptions();
        _pagedPool = pagedPool;
        _prefixCache = prefixCache;
    }

    /// <summary>Cumulative prompt tokens served from the prefix cache (no prefill needed).</summary>
    public long CachedPromptTokens => Interlocked.Read(ref _cachedPromptTokens);

    /// <summary>Cumulative prompt tokens that required prefill compute.</summary>
    public long PrefilledPromptTokens => Interlocked.Read(ref _prefilledPromptTokens);

    /// <inheritdoc/>
    public ISchedulerRequest Submit(InferenceRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (_disposed) throw new ObjectDisposedException(nameof(ContinuousBatchScheduler));

        // Validate prompt non-empty (mirrors TextGenerator guard but throws here — scheduler
        // callers are server endpoints that have already validated input).
        var promptIds = request.TokenIds;
        if (promptIds is null || promptIds.Length == 0)
            throw new ArgumentException("Request prompt must contain at least one token.", nameof(request));

        var options = request.Options;
        int maxTokens = Math.Max(1, options.MaxTokens);

        // Sampler pipeline (per-sequence — pipelines hold RNG state, so we can't share).
        var pipeline = new SamplerPipeline(options);

        // Build stop conditions. Mirrors TextGenerator: explicit list if set, else EOS + MaxTokens + stop strings.
        IReadOnlyList<IStopCondition> stops;
        if (options.StopConditions is not null)
        {
            stops = options.StopConditions;
        }
        else
        {
            var list = new List<IStopCondition>(capacity: 2 + options.StopSequences.Count)
            {
                new EosStopCondition(_tokenizer.EosTokenId),
                new MaxTokensStopCondition(maxTokens),
            };
            foreach (var stopSeq in options.StopSequences)
                list.Add(new StopStringCondition(stopSeq));
            stops = list;
        }

        // Decoding constraint for structured output (JSON / schema / regex / grammar).
        IDecodingConstraint? constraint = options.ResponseFormat switch
        {
            ResponseFormat.JsonObject => new JsonConstraint(_tokenizer),
            ResponseFormat.JsonSchema js => new JsonSchemaConstraint(_tokenizer, js.Schema),
            ResponseFormat.Regex rx => new RegexConstraint(_tokenizer, rx.Pattern),
            ResponseFormat.Grammar gr => new GrammarConstraint(_tokenizer, gr.GbnfGrammar),
            _ => null,
        };

        var tcs = new TaskCompletionSource<InferenceResponse>(
            TaskCreationOptions.RunContinuationsAsynchronously);

        var seq = new SchedulerRequest(
            request,
            new InferenceOptionsLike(maxTokens, options.Logprobs, options.TopLogprobs),
            promptLength: promptIds.Length,
            maxTokens: maxTokens,
            samplerPipeline: pipeline,
            stopConditions: stops,
            constraint: constraint,
            submissionOrder: Interlocked.Increment(ref _submissionCounter),
            tcs: tcs);

        if (cancellationToken.CanBeCanceled)
        {
            seq.CancellationRegistration = cancellationToken.Register(static state =>
            {
                var s = (SchedulerRequest)state!;
                // We cannot safely free the KV-cache here — the scheduler thread may be in the
                // middle of decoding this sequence. Mark for cancellation; Step() will collect it.
                s.State = SequenceState.Cancelled;
                s.CompletionSource.TrySetCanceled();
            }, seq);
        }

        _pendingQueue.Enqueue(seq);
        return seq;
    }

    /// <inheritdoc/>
    public bool Step()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ContinuousBatchScheduler));

        bool didWork = false;

        // 1. Sweep cancelled sequences (caller-side cancellation may have flipped state).
        for (int i = _active.Count - 1; i >= 0; i--)
        {
            var s = _active[i];
            if (s.State == SequenceState.Cancelled)
            {
                ReleaseKvCache(s);
                _active.RemoveAt(i);
                didWork = true;
            }
        }

        // 2. Admit new sequences from the queue, subject to capacity. Each admitted sequence
        //    runs prompt prefill in this same iteration (chunked prefill is a stretch goal).
        int admittedThisStep = 0;
        int prefillTokensThisStep = 0;
        while (_active.Count < _options.MaxActiveSequences && _pendingQueue.TryPeek(out var head))
        {
            // Cancelled while queued — drop without admission.
            if (head.State == SequenceState.Cancelled)
            {
                _pendingQueue.TryDequeue(out _);
                continue;
            }

            // Bound prefill cost per step. Once we've started admitting, finishing the head is fine
            // unless this step has already done meaningful prefill work.
            if (_options.MaxPrefillTokensPerStep > 0 &&
                admittedThisStep > 0 &&
                prefillTokensThisStep + head.PromptLength > _options.MaxPrefillTokensPerStep)
            {
                break;
            }

            // Block-pool gating: refuse admission if the paged pool can't fit the worst-case
            // footprint. Try to relieve pressure by evicting zero-refcount trie blocks first.
            if (_pagedPool is not null && _options.ReserveBlocksPerSequence > 0 &&
                _pagedPool.FreeBlocks < _options.ReserveBlocksPerSequence)
            {
                int short_ = _options.ReserveBlocksPerSequence - _pagedPool.FreeBlocks;
                if (_prefixCache is not null)
                    _prefixCache.TryEvict(short_);
                if (_pagedPool.FreeBlocks < _options.ReserveBlocksPerSequence)
                    break;
            }

            if (!_pendingQueue.TryDequeue(out var seq))
                break;

            try
            {
                AdmitAndPrefill(seq);
                admittedThisStep++;
                prefillTokensThisStep += seq.PromptLength;
                didWork = true;

                // If the first sampled token already triggered a stop condition, AdmitAndPrefill
                // sets state to Completed — finish out the response and skip the decoding queue.
                if (seq.State == SequenceState.Completed)
                {
                    CompleteSequence(seq);
                }
                else
                {
                    _active.Add(seq);
                }
            }
            catch (OperationCanceledException)
            {
                seq.State = SequenceState.Cancelled;
                ReleaseKvCache(seq);
                seq.CompletionSource.TrySetCanceled();
            }
            catch (Exception ex)
            {
                seq.State = SequenceState.Completed;
                ReleaseKvCache(seq);
                seq.CompletionSource.TrySetException(ex);
            }
        }

        // 3. Decode one token for every actively-decoding sequence.
        for (int i = _active.Count - 1; i >= 0; i--)
        {
            var seq = _active[i];
            if (seq.State != SequenceState.Decoding) continue;

            try
            {
                bool finished = DecodeOneStep(seq);
                didWork = true;
                if (finished)
                {
                    seq.State = SequenceState.Completed;
                    CompleteSequence(seq);
                    _active.RemoveAt(i);
                }
            }
            catch (OperationCanceledException)
            {
                seq.State = SequenceState.Cancelled;
                ReleaseKvCache(seq);
                seq.CompletionSource.TrySetCanceled();
                _active.RemoveAt(i);
            }
            catch (Exception ex)
            {
                seq.State = SequenceState.Completed;
                ReleaseKvCache(seq);
                seq.CompletionSource.TrySetException(ex);
                _active.RemoveAt(i);
            }
        }

        return didWork;
    }

    /// <inheritdoc/>
    public SchedulerMetrics GetMetrics() => new(
        ActiveSequences: _active.Count,
        QueueDepth: _pendingQueue.Count,
        PreemptionCount: Interlocked.Read(ref _preemptionCount));

    // ── Admission & prefill ──

    private void AdmitAndPrefill(SchedulerRequest seq)
    {
        Debug.Assert(seq.State == SequenceState.Queued);

        int promptLen = seq.PromptLength;
        int cacheSize = Math.Min(promptLen + seq.MaxTokens, _model.Config.MaxSequenceLength);
        var promptIds = seq.PromptTokenIds;

        // Prefix-cache-aware admission: when the manager can seed a prefix, only the
        // suffix is run through the model. Falls back to the configured factory when
        // the cache is disabled, missed, or no manager is wired.
        int cachedTokens = 0;
        if (_prefixCache is not null)
        {
            var admission = _prefixCache.Admit(promptIds, cacheSize);
            seq.KvCache = admission.Cache;
            seq.IsPrefixCached = true;
            cachedTokens = admission.CachedTokens;
        }
        else
        {
            seq.KvCache = _kvCacheFactory(_model.Config, cacheSize);
        }

        seq.PrefixCachedTokens = cachedTokens;
        Interlocked.Add(ref _cachedPromptTokens, cachedTokens);
        Interlocked.Add(ref _prefilledPromptTokens, promptLen - cachedTokens);
        seq.State = SequenceState.Prefilling;

        int vocabSize = _model.Config.VocabSize;

        int prefillStart = cachedTokens;
        int prefillLen = promptLen - prefillStart;

        int[] positionsArray = ArrayPool<int>.Shared.Rent(Math.Max(1, prefillLen));
        try
        {
            ReadOnlySpan<int> forwardTokens;
            Span<int> positions;
            if (prefillLen > 0)
            {
                positions = positionsArray.AsSpan(0, prefillLen);
                for (int i = 0; i < prefillLen; i++)
                    positions[i] = prefillStart + i;
                forwardTokens = promptIds.AsSpan(prefillStart);
            }
            else
            {
                // 100% cache hit — re-forward last prompt token to obtain its logits.
                positions = positionsArray.AsSpan(0, 1);
                positions[0] = promptLen - 1;
                forwardTokens = promptIds.AsSpan(promptLen - 1, 1);
            }

            long ts0 = Stopwatch.GetTimestamp();
            using ITensor prefillLogits = _model.Forward(forwardTokens, positions, deviceId: -1, seq.KvCache);
            long ts1 = Stopwatch.GetTimestamp();
            seq.PrefillTicks = ts1 - ts0;

            // Sample first token from last-position logits.
            unsafe
            {
                float* logitPtr = (float*)prefillLogits.DataPointer;
                int logitRows = prefillLogits.Shape[0];
                var logitSpan = new Span<float>(logitPtr + (long)(logitRows - 1) * vocabSize, vocabSize);

                if (seq.Constraint is not null)
                    TokenMaskApplier.Apply(logitSpan, seq.Constraint.GetAllowedTokens());

                long sStart = Stopwatch.GetTimestamp();
                int firstToken = seq.SamplerPipeline.Sample(logitSpan, seq.GeneratedTokens);
                seq.SamplerTicks += Stopwatch.GetTimestamp() - sStart;

                seq.Constraint?.Advance(firstToken);
                seq.GeneratedTokens.Add(firstToken);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(positionsArray);
        }

        // Check stop conditions on the first generated token. If satisfied, sequence completes
        // without entering the decoding phase.
        if (CheckStopAfterAppend(seq, out var result))
        {
            seq.FinishReason = result == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
            seq.State = SequenceState.Completed;
            return;
        }

        seq.State = SequenceState.Decoding;
    }

    // ── Decode ──

    private bool DecodeOneStep(SchedulerRequest seq)
    {
        Debug.Assert(seq.State == SequenceState.Decoding);

        int cacheSize = seq.KvCache!.MaxLength;
        int pos = seq.Position - 1; // position of the last token appended (the one whose successor we generate)

        // Capacity / max-tokens gates.
        // Position is appended at seq.PromptLength + seq.GeneratedCount - 1 already. The NEXT
        // forward consumes that token at position (PromptLength + GeneratedCount - 1).
        int nextPos = pos;
        if (nextPos >= cacheSize)
        {
            seq.FinishReason = FinishReason.Length;
            return true;
        }
        if (seq.GeneratedCount >= seq.MaxTokens)
        {
            seq.FinishReason = FinishReason.Length;
            return true;
        }

        int vocabSize = _model.Config.VocabSize;
        int lastToken = seq.GeneratedTokens[^1];
        Span<int> tokenSpan = stackalloc int[1] { lastToken };
        Span<int> posSpan = stackalloc int[1] { nextPos };

        int nextTokenId;
        long fwdStart = Stopwatch.GetTimestamp();
        using (ITensor logits = _model.Forward(tokenSpan, posSpan, deviceId: -1, seq.KvCache))
        {
            seq.DecodeTicks += Stopwatch.GetTimestamp() - fwdStart;

            unsafe
            {
                var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
                if (seq.Constraint is not null)
                    TokenMaskApplier.Apply(logitSpan, seq.Constraint.GetAllowedTokens());

                long sStart = Stopwatch.GetTimestamp();
                nextTokenId = seq.SamplerPipeline.Sample(logitSpan, seq.GeneratedTokens);
                seq.SamplerTicks += Stopwatch.GetTimestamp() - sStart;
            }
        }

        seq.Constraint?.Advance(nextTokenId);
        seq.GeneratedTokens.Add(nextTokenId);

        if (CheckStopAfterAppend(seq, out var result))
        {
            seq.FinishReason = result == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
            return true;
        }

        return false;
    }

    // ── Helpers ──

    /// <summary>
    /// Runs stop conditions over the latest appended token. If the result is <see cref="StopResult.Stop"/>,
    /// removes the trailing token from the output (matching TextGenerator semantics).
    /// </summary>
    private static bool CheckStopAfterAppend(SchedulerRequest seq, out StopResult result)
    {
        result = StopResult.Continue;
        int last = seq.GeneratedTokens[^1];

        // MVP: we do not pass a decoded-text tail. Stop-string conditions therefore won't fire.
        // EOS and MaxTokens both work on tokenId / count alone, which covers the contract
        // documented in CLAUDE.md. Tail-aware stop strings are a near-term enhancement —
        // we'd need a per-sequence IncrementalDetokenizer (see DEFERRED note in the test class).
        ReadOnlySpan<char> emptyTail = ReadOnlySpan<char>.Empty;

        for (int i = 0; i < seq.StopConditions.Count; i++)
        {
            var r = seq.StopConditions[i].ShouldStop(last, seq.GeneratedTokens, emptyTail);
            if (r != StopResult.Continue)
            {
                result = r;
                if (r == StopResult.Stop)
                {
                    // Stop semantics exclude the triggering token from output.
                    seq.GeneratedTokens.RemoveAt(seq.GeneratedTokens.Count - 1);
                }
                return true;
            }
        }
        return false;
    }

    private void CompleteSequence(SchedulerRequest seq)
    {
        Debug.Assert(seq.State == SequenceState.Completed);

        // Build response.
        string text = seq.GeneratedTokens.Count > 0
            ? _tokenizer.Decode(CollectionsMarshal.AsSpan(seq.GeneratedTokens), stripBosSpace: false)
            : string.Empty;

        long kvBytes = seq.KvCache is not null ? TextGenerator.GetKvCacheBytes(seq.KvCache) : 0;

        var timings = BuildTimings(
            seq.PromptLength,
            seq.GeneratedTokens.Count,
            seq.PrefillTicks,
            seq.DecodeTicks,
            seq.SamplerTicks,
            kvBytes);

        var response = new InferenceResponse
        {
            GeneratedTokenIds = seq.GeneratedTokens.ToArray(),
            Text = text,
            FinishReason = seq.FinishReason,
            PromptTokenCount = seq.PromptLength,
            GeneratedTokenCount = seq.GeneratedTokens.Count,
            Timings = timings,
        };

        ReleaseKvCache(seq);
        seq.CancellationRegistration.Dispose();
        seq.CompletionSource.TrySetResult(response);
    }

    private static InferenceTimings BuildTimings(
        int promptLen, int generatedCount,
        long prefillTicks, long decodeTicks, long samplerTicks, long kvBytes)
    {
        double tickFreq = Stopwatch.Frequency;
        int decodeSteps = generatedCount > 1 ? generatedCount - 1 : 0;
        return new InferenceTimings
        {
            PrefillTimeMs = prefillTicks / tickFreq * 1000.0,
            DecodeTimeMs = decodeTicks / tickFreq * 1000.0,
            SamplingTimeMs = samplerTicks / tickFreq * 1000.0,
            PrefillTokenCount = promptLen,
            DecodeTokenCount = decodeSteps,
            KvCacheBytes = kvBytes,
        };
    }

    private void ReleaseKvCache(SchedulerRequest seq)
    {
        var cache = seq.KvCache;
        if (cache is null) return;
        seq.KvCache = null;

        // Push back into the prefix trie before disposal so freshly-computed blocks
        // can be reused by future requests.
        if (seq.IsPrefixCached && _prefixCache is not null && cache is PagedKvCache paged)
        {
            try
            {
                // Build the full token sequence (prompt + generated) covered by the cache.
                int promptLen = seq.PromptLength;
                int genCount = seq.GeneratedTokens.Count;
                int totalLen = promptLen + genCount;
                var full = ArrayPool<int>.Shared.Rent(totalLen);
                try
                {
                    Array.Copy(seq.PromptTokenIds, full, promptLen);
                    for (int i = 0; i < genCount; i++)
                        full[promptLen + i] = seq.GeneratedTokens[i];
                    _prefixCache.RecordCompletion(paged, full.AsSpan(0, totalLen));
                }
                finally
                {
                    ArrayPool<int>.Shared.Return(full);
                }
            }
            catch
            {
                // Telemetry-only failure; never block the scheduler loop.
            }
        }

        try { cache.Dispose(); }
        catch
        {
            // Disposal failures must not derail the scheduler loop. Future: telemetry hook.
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Cancel everything in flight.
        foreach (var active in _active)
        {
            ReleaseKvCache(active);
            active.CompletionSource.TrySetCanceled();
            active.CancellationRegistration.Dispose();
        }
        _active.Clear();

        while (_pendingQueue.TryDequeue(out var pending))
        {
            pending.CompletionSource.TrySetCanceled();
            pending.CancellationRegistration.Dispose();
        }
    }
}
