using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Constraints;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Constraints;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Engine.Samplers;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Engine.Strategies;
using DotLLM.Telemetry;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Autoregressive text generator: encodes a prompt, runs prefill + decode loop
/// with sampling and stop conditions, and returns the generated text.
/// </summary>
public sealed class TextGenerator
{
    private readonly IModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly Func<ModelConfig, int, Core.Attention.IKvCache>? _kvCacheFactory;
    private readonly PrefixCache? _prefixCache;
    private readonly PrefixTrieManager? _prefixTrieManager;
    private readonly IModel? _draftModel;
    private readonly Func<ModelConfig, int, Core.Attention.IKvCache>? _draftKvCacheFactory;
    private readonly int _speculativeCandidates;
    private readonly HybridPrefillDecodeStrategy? _hybridStrategy;

    /// <summary>
    /// Creates a new text generator.
    /// </summary>
    /// <param name="model">The model to use for forward passes.</param>
    /// <param name="tokenizer">The tokenizer for encoding/decoding text.</param>
    /// <param name="kvCacheFactory">Optional factory for creating a KV-cache. When null, uses <see cref="SimpleKvCache"/>.
    /// Parameters: (config, maxSeqLen).</param>
    /// <param name="prefixCache">Optional prefix cache for reusing KV-cache state across calls.
    /// When provided, the KV-cache is kept alive between calls and only new suffix tokens are prefilled.</param>
    /// <param name="draftModel">Optional draft model for speculative decoding.</param>
    /// <param name="draftKvCacheFactory">Optional factory for creating the draft model's KV-cache.</param>
    /// <param name="speculativeCandidates">Number of draft tokens per speculative step (K). Default 5.</param>
    /// <param name="prefixTrieManager">Optional cross-request prefix trie manager (Step 37).
    /// Takes precedence over <paramref name="prefixCache"/> when supplied — multiple sessions
    /// share KV blocks via the trie.</param>
    /// <param name="hybridStrategy">Optional CPU-prefill / GPU-decode hybrid strategy. When set
    /// and the prompt length is below the strategy's crossover threshold, prefill runs on the
    /// strategy's CPU model and the KV state is handed off to <paramref name="model"/> (the
    /// decode model) before the decode loop. When the prompt exceeds the threshold, or when
    /// the strategy is null, the existing single-backend path runs unchanged.</param>
    public TextGenerator(IModel model, ITokenizer tokenizer,
                          Func<ModelConfig, int, Core.Attention.IKvCache>? kvCacheFactory = null,
                          PrefixCache? prefixCache = null,
                          IModel? draftModel = null,
                          Func<ModelConfig, int, Core.Attention.IKvCache>? draftKvCacheFactory = null,
                          int speculativeCandidates = 5,
                          PrefixTrieManager? prefixTrieManager = null,
                          HybridPrefillDecodeStrategy? hybridStrategy = null)
    {
        _model = model;
        _tokenizer = tokenizer;
        _kvCacheFactory = kvCacheFactory;
        _prefixCache = prefixCache;
        _prefixTrieManager = prefixTrieManager;
        _draftModel = draftModel;
        _draftKvCacheFactory = draftKvCacheFactory;
        _speculativeCandidates = speculativeCandidates;
        _hybridStrategy = hybridStrategy;

        if (hybridStrategy is not null
            && !ReferenceEquals(hybridStrategy.DecodeModel, model))
        {
            throw new ArgumentException(
                "When a HybridPrefillDecodeStrategy is supplied, its DecodeModel must be the same "
                + "instance as the TextGenerator's primary model (which runs the decode loop).",
                nameof(hybridStrategy));
        }
    }


    /// <summary>
    /// Generates text from the given prompt using the specified options.
    /// </summary>
    /// <param name="prompt">Input text prompt.</param>
    /// <param name="options">Inference options controlling sampling and stopping. Null uses defaults.</param>
    /// <param name="onTokenGenerated">Optional callback invoked after each token is generated, receiving the token ID.</param>
    /// <param name="adapter">Optional LoRA adapter to apply during the forward passes (Phase 4c).</param>
    /// <returns>The inference response with generated text, metadata, and timings.</returns>
    /// <remarks>
    /// Synchronous wrapper over <see cref="GenerateCoreAsync"/>. The core never <c>await</c>s a
    /// non-completed task (every <c>MoveNextAsync</c> completes inline because all model / sampler
    /// calls are CPU/GPU-synchronous), so the blocking drain pattern is safe — it never schedules
    /// a continuation on the thread pool. Accumulates ids and logprobs via side-effect parameters
    /// so the core's existing keep/remove semantics on stop conditions are preserved byte-identical
    /// without reconstruction logic in this wrapper.
    /// </remarks>
    public InferenceResponse Generate(string prompt, InferenceOptions? options = null,
        Action<int>? onTokenGenerated = null,
        ILoraAdapter? adapter = null)
    {
        options ??= new InferenceOptions();
        int maxTokens = options.MaxTokens;

        // Guard: MaxTokens=0 — return immediately, no generation. Pre-encode to honour the
        // PromptTokenCount contract (matches the empty-prompt → BOS-seed guard inside the core).
        if (maxTokens <= 0)
        {
            int[] promptIdsEarly = _tokenizer.Encode(prompt);
            int promptLenEarly = promptIdsEarly.Length == 0 ? 1 : promptIdsEarly.Length;
            return new InferenceResponse
            {
                GeneratedTokenIds = [],
                Text = string.Empty,
                FinishReason = FinishReason.Length,
                PromptTokenCount = promptLenEarly,
                GeneratedTokenCount = 0
            };
        }

        bool captureLogprobs = options.Logprobs;
        var generatedIds = new List<int>(maxTokens);
        var logprobsList = captureLogprobs ? new List<TokenLogprobInfo>(maxTokens) : null;

        // Stop-conditions list is replicated here so the wrapper can pass it to
        // BuildResponseFromTimings for character-level suffix trimming on stop-string matches.
        // The core builds its own identical list via the same helper; the two lists are
        // semantically equivalent.
        var stopConditions = BuildStopConditions(options, maxTokens);

        FinishReason finishReason = FinishReason.Length;
        InferenceTimings? terminalTimings = null;

        var enumerator = GenerateCoreAsync(prompt, options, generatedIds, logprobsList,
            CancellationToken.None, adapter).GetAsyncEnumerator();
        try
        {
            while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
                var token = enumerator.Current;

                // onTokenGenerated parity: invoked iff the token survives into the visible output.
                // StopInclude (FinishReason.Length) and natural-end (FinishReason.Length) tokens are
                // kept; Stop tokens are either kept in generatedIds (stop-string match) or removed
                // (EOS-like) by the core itself — but in either case the callback was historically
                // NOT invoked, so we mirror that: skip the callback iff FinishReason==Stop.
                if (token.FinishReason != FinishReason.Stop)
                    onTokenGenerated?.Invoke(token.TokenId);

                if (token.FinishReason is { } fr)
                {
                    finishReason = fr;
                    terminalTimings = token.Timings;
                }
            }
        }
        finally
        {
            enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }

        // PromptTokenCount is carried in the terminal token's timings (set by the core's BuildTimings).
        int promptLen = terminalTimings?.PrefillTokenCount ?? 0;

        return BuildResponseFromTimings(promptLen, generatedIds, finishReason,
            terminalTimings, logprobsList?.ToArray(), stopConditions);
    }

    /// <summary>
    /// Streams generated tokens as an async enumerable, yielding each token with incremental text,
    /// finish reason, and timings on the final token. Thin pass-through over <see cref="GenerateCoreAsync"/>.
    /// </summary>
    /// <param name="prompt">Input text prompt.</param>
    /// <param name="options">Inference options controlling sampling and stopping. Null uses defaults.</param>
    /// <param name="cancellationToken">Token to cancel generation cooperatively between decode steps.</param>
    /// <param name="adapter">Optional LoRA adapter to apply during the forward passes (Phase 4c).</param>
    /// <returns>An async enumerable of <see cref="GenerationToken"/> values.</returns>
    public IAsyncEnumerable<GenerationToken> GenerateStreamingTokensAsync(
        string prompt,
        InferenceOptions? options = null,
        CancellationToken cancellationToken = default,
        ILoraAdapter? adapter = null)
    {
        options ??= new InferenceOptions();
        int maxTokens = options.MaxTokens;
        // Streaming MaxTokens<=0 historically yields nothing (no terminal token).
        // The core handles the same guard internally, so the wrapper just passes through.
        var generatedIds = new List<int>(Math.Max(1, maxTokens));
        // Streaming doesn't accumulate a logprobs list — per-token logprobs travel via
        // GenerationToken.Logprobs. Passing null tells the core to skip list bookkeeping.
        return GenerateCoreAsync(prompt, options, generatedIds, logprobsList: null,
            cancellationToken, adapter);
    }

    /// <summary>
    /// Streams generated text as an async enumerable, yielding incremental text fragments.
    /// This is a convenience wrapper over <see cref="GenerateStreamingTokensAsync"/>.
    /// </summary>
    /// <param name="prompt">Input text prompt.</param>
    /// <param name="options">Inference options controlling sampling and stopping. Null uses defaults.</param>
    /// <param name="cancellationToken">Token to cancel generation cooperatively between decode steps.</param>
    /// <param name="adapter">Optional LoRA adapter to apply during the forward passes (Phase 4c).</param>
    /// <returns>An async enumerable of incremental text strings.</returns>
    public async IAsyncEnumerable<string> GenerateStreamingAsync(
        string prompt,
        InferenceOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default,
        ILoraAdapter? adapter = null)
    {
        await foreach (var token in GenerateStreamingTokensAsync(prompt, options, cancellationToken, adapter))
            yield return token.Text;
    }

    /// <summary>
    /// Single source of truth for the prefill + decode + stop orchestration. Both the synchronous
    /// <see cref="Generate"/> and streaming <see cref="GenerateStreamingTokensAsync"/> consume this.
    /// Yields one <see cref="GenerationToken"/> per visible decoded token; the last yielded token
    /// carries <see cref="GenerationToken.FinishReason"/> and final <see cref="InferenceTimings"/>.
    /// </summary>
    /// <param name="prompt">Input text prompt.</param>
    /// <param name="options">Inference options controlling sampling and stopping (must be non-null).</param>
    /// <param name="generatedIds">List that the core populates with the final visible token ids.
    /// Stop-string matches keep the triggering token; EOS-like matches do not (mirrors today's behaviour).</param>
    /// <param name="logprobsList">Optional list to accumulate per-token logprobs into. Null when the caller
    /// only needs the per-yield <see cref="GenerationToken.Logprobs"/> (streaming).</param>
    /// <param name="cancellationToken">Cooperative cancellation between decode steps.</param>
    /// <param name="adapter">Optional LoRA adapter applied to every forward pass.</param>
    /// <returns>An async enumerable that yields one <see cref="GenerationToken"/> per visible decoded token.</returns>
    /// <remarks>
    /// All terminal cleanup — <c>StoreInPrefixCache</c>, <c>telemetry.Complete</c>, KV-cache disposal,
    /// scratch buffer return — happens exactly once in this method's <c>finally</c> block, regardless of
    /// which terminal branch fired. This eliminates the seven prior <c>StoreInPrefixCache</c> sites in
    /// the streaming code and fixes the telemetry <c>FinishReason</c> being hard-coded to <c>Length</c>.
    /// </remarks>
    [SkipLocalsInit]
    private async IAsyncEnumerable<GenerationToken> GenerateCoreAsync(
        string prompt,
        InferenceOptions options,
        List<int> generatedIds,
        List<TokenLogprobInfo>? logprobsList,
        [EnumeratorCancellation] CancellationToken cancellationToken,
        ILoraAdapter? adapter)
    {
        int[] promptIds = _tokenizer.Encode(prompt);
        int promptLen = promptIds.Length;
        int maxTokens = options.MaxTokens;
        int vocabSize = _model.Config.VocabSize;

        // Guard: empty prompt — use BOS token as seed
        if (promptLen == 0)
        {
            promptIds = [_tokenizer.BosTokenId];
            promptLen = 1;
        }

        // Guard: MaxTokens=0 — yield nothing. The synchronous wrapper short-circuits before
        // ever invoking the core in this case, so this path is exercised only by streaming.
        if (maxTokens <= 0)
            yield break;

        cancellationToken.ThrowIfCancellationRequested();

        var telemetry = new TelemetryRecorder(_model.Config, options);

        // Build sampling pipeline
        var pipeline = new SamplerPipeline(options);

        // Logprobs capture setup
        bool captureLogprobs = options.Logprobs;
        int topLogprobs = Math.Clamp(options.TopLogprobs, 0, 20);

        // Build decoding constraint for structured output
        IDecodingConstraint? constraint = options.ResponseFormat switch
        {
            ResponseFormat.JsonObject => new JsonConstraint(_tokenizer),
            ResponseFormat.JsonSchema js => new JsonSchemaConstraint(_tokenizer, js.Schema),
            ResponseFormat.Regex rx => new RegexConstraint(_tokenizer, rx.Pattern),
            ResponseFormat.Grammar gr => new GrammarConstraint(_tokenizer, gr.GbnfGrammar),
            _ => null
        };

        // Build stop conditions
        var stopConditions = BuildStopConditions(options, maxTokens);

        // Resolve KV-cache: reuse from prefix cache or allocate fresh
        var (kvCache, cachedTokenCount, ownsKvCache) = ResolveKvCache(promptIds, promptLen, maxTokens);
        long kvBytes = GetKvCacheBytes(kvCache);

        // Hybrid mode: enabled when a strategy is wired up, the prompt is short enough, and we
        // have a clean cache (no prefix-cache reuse, no speculative draft model).
        bool useHybrid = _hybridStrategy is not null
            && _hybridStrategy.ShouldRunHybrid(promptLen)
            && cachedTokenCount == 0
            && _draftModel is null;

        // Stop-check scratch buffer: rented up-front and returned in the outer finally. try/finally
        // is preserved across yield points by the async-iterator state machine, so Return runs on
        // normal completion, exception, or consumer-side cancellation (Dispose of the enumerator).
        int stopTailSize = ComputeStopTailSize(stopConditions);
        char[] stopScratch = ArrayPool<char>.Shared.Rent(stopTailSize);
        long prefillTicks = 0;
        long decodeTicks = 0;
        long samplerTicks = 0;
        int specDrafted = 0, specAccepted = 0;

        // Method-scoped finish reason: defaults to Length (natural-end / max-tokens) and is
        // overwritten by every terminal branch before yield. The outer finally passes it to
        // telemetry.Complete — fixing the streaming-side bug where Length was always reported.
        FinishReason finishReason = FinishReason.Length;

        // Incremental detokenizer: O(1) amortized per token for stop-check + streaming delta,
        // instead of decoding the full generated sequence each step. Lifted outside the try so
        // the finally can deterministically return its pooled buffers even on cancellation.
        var detok = new IncrementalDetokenizer(_tokenizer, initialCapacity: Math.Max(64, maxTokens * 4));

        try
        {
            int cacheSize = kvCache.MaxLength;

            // Streaming holdback buffer: keeps the last max-stop-string chars un-emitted so a
            // stop-string match split across multiple tokens can be trimmed character-exactly
            // before any part of it leaks to the SSE consumer (#121 item #8). No-op when no
            // StopStringCondition is registered — preserves zero-latency EOS-only behaviour.
            // For non-streaming callers the holdback simply equals "delta-of-the-moment" plus
            // a flush at the end, so the same machinery serves both paths correctly.
            var streamBuffer = new StreamingStopBuffer(stopConditions);

            // Local helper: snapshot log-softmax before sampling (which modifies logits in-place),
            // sample a token, then build logprob info. Closes over generatedIds / pipeline /
            // logprobsList / topLogprobs / vocabSize.
            (int tokenId, TokenLogprobInfo? logprob) SampleWithLogprobs(Span<float> logitSpan)
            {
                float[]? lsBuf = captureLogprobs ? LogprobsCapture.ComputeLogSoftmax(logitSpan) : null;
                int tokenId = pipeline.Sample(logitSpan, generatedIds);
                TokenLogprobInfo? info = null;
                if (lsBuf != null)
                {
                    info = LogprobsCapture.BuildInfo(lsBuf.AsSpan(0, vocabSize), vocabSize, tokenId, topLogprobs, _tokenizer);
                    ArrayPool<float>.Shared.Return(lsBuf);
                }
                return (tokenId, info);
            }

            // Prefill: run only new suffix tokens through the model
            int prefillStart = cachedTokenCount;
            int prefillLen = promptLen - prefillStart;

            int firstTokenId;
            TokenLogprobInfo? firstLogprobInfo = null;
            long ts0 = Stopwatch.GetTimestamp();

            using (var prefillSpan = telemetry.StartPrefill())
            {
                if (useHybrid)
                {
                    // ── Hybrid prefill: CPU populates a SimpleKvCache, hand off to kvCache.
                    var handoff = _hybridStrategy!.RunPrefill(promptIds.AsSpan(0, promptLen), cacheSize);
                    try
                    {
                        _hybridStrategy.Handoff(handoff.HostCache, kvCache);
                        prefillTicks = handoff.PrefillTicks;

                        using var sampleSpan = telemetry.StartSample();
                        long samplerStart = Stopwatch.GetTimestamp();
                        var logitSpan = handoff.LastLogits.AsSpan(0, vocabSize);
                        if (constraint != null)
                            TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                        (firstTokenId, firstLogprobInfo) = SampleWithLogprobs(logitSpan);
                        samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                    }
                    finally
                    {
                        handoff.HostCache.Dispose();
                    }
                }
                else if (prefillLen > 0)
                {
                    // Span slice avoids array allocation for suffix tokens
                    ReadOnlySpan<int> suffixTokens = promptIds.AsSpan(prefillStart);
                    int[] positionsArray = ArrayPool<int>.Shared.Rent(prefillLen);
                    try
                    {
                        Span<int> positions = positionsArray.AsSpan(0, prefillLen);
                        for (int i = 0; i < prefillLen; i++)
                            positions[i] = prefillStart + i;

                        using (ITensor prefillLogits = _model.Forward(suffixTokens, positions, deviceId: -1, kvCache, adapter))
                        {
                            long ts1 = Stopwatch.GetTimestamp();
                            prefillTicks = ts1 - ts0;

                            unsafe
                            {
                                using var sampleSpan = telemetry.StartSample();
                                long samplerStart = Stopwatch.GetTimestamp();
                                // GPU/hybrid models return [1, vocabSize] (last token only);
                                // CPU model returns [seqLen, vocabSize]. Use actual shape to index.
                                float* logitPtr = (float*)prefillLogits.DataPointer;
                                int logitRows = prefillLogits.Shape[0];
                                var logitSpan = new Span<float>(logitPtr + (long)(logitRows - 1) * vocabSize, vocabSize);
                                if (constraint != null)
                                    TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                                (firstTokenId, firstLogprobInfo) = SampleWithLogprobs(logitSpan);
                                samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<int>.Shared.Return(positionsArray);
                    }
                }
                else if (promptLen > 0)
                {
                    // 100% cache hit — re-forward last prompt token to get logits
                    using (ITensor logits = _model.Forward([promptIds[^1]], [promptLen - 1], deviceId: -1, kvCache, adapter))
                    {
                        long ts1 = Stopwatch.GetTimestamp();
                        prefillTicks = ts1 - ts0;

                        unsafe
                        {
                            using var sampleSpan = telemetry.StartSample();
                            long samplerStart = Stopwatch.GetTimestamp();
                            var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            (firstTokenId, firstLogprobInfo) = SampleWithLogprobs(logitSpan);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }
                    }
                }
                else
                {
                    // Unreachable: empty prompt guard ensures promptLen >= 1
                    throw new InvalidOperationException("Prompt is empty after guard.");
                }

                if (prefillSpan is { IsAllDataRequested: true })
                {
                    prefillSpan.SetTag(TelemetryTags.PrefillTokenCount, prefillLen);
                    prefillSpan.SetTag(TelemetryTags.PrefillDurationMs, prefillTicks * 1000.0 / Stopwatch.Frequency);
                }
            }

            telemetry.RecordFirstToken();
            constraint?.Advance(firstTokenId);

            // First-token logprob mirrors today's behaviour: added to logprobsList BEFORE the
            // stop-check, so even the discarded trigger contributes a logprob entry. Kept here
            // for byte-identical parity with the prior non-streaming path.
            if (firstLogprobInfo.HasValue) logprobsList?.Add(firstLogprobInfo.Value);

            // Check stop conditions for first token
            generatedIds.Add(firstTokenId);
            detok.Append(firstTokenId);

            var stopResult = CheckStopConditions(stopConditions, firstTokenId, generatedIds,
                detok.GetTailView(stopTailSize, stopScratch), out int firstMatchedIdx);
            if (stopResult != StopResult.Continue)
            {
                bool isStopStringMatch = IsStopStringMatch(stopConditions, firstMatchedIdx);
                finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;

                if (stopResult == StopResult.Stop)
                {
                    // Push the just-decoded delta into the holdback buffer first — its
                    // safe-emit return must NOT be discarded (any text past the holdback
                    // window is already trim-immune). Then trim the matched suffix
                    // (stop-string case) or flush as-is (EOS etc.).
                    string newDelta = streamBuffer.Push(detok.TakeDelta());
                    string emit = newDelta + (isStopStringMatch
                        ? streamBuffer.TrimAndFlush(stopConditions)
                        : streamBuffer.FlushAll());
                    if (!isStopStringMatch)
                        generatedIds.RemoveAt(generatedIds.Count - 1);
                    // Stop-string match: keep token in id list so KV-cache length stays in sync
                    // with the stored prompt+generated sequence (mirrors prior non-streaming path).
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                    yield return new GenerationToken(firstTokenId, emit, finishReason, timings, firstLogprobInfo);
                }
                else
                {
                    // StopInclude (e.g. max-tokens consuming the just-yielded token) — flush all
                    // buffered text including the just-decoded delta. No trim.
                    string emit = streamBuffer.Push(detok.TakeDelta()) + streamBuffer.FlushAll();
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                    yield return new GenerationToken(firstTokenId, emit, finishReason, timings, firstLogprobInfo);
                }
                yield break;
            }

            // Yield first token — check if it's also the last (maxTokens == 1)
            {
                bool firstIsLast = maxTokens <= 1;
                string emit = streamBuffer.Push(detok.TakeDelta());
                if (firstIsLast)
                {
                    // Decode loop won't run; drain holdback as the natural-end (Length) emit.
                    emit += streamBuffer.FlushAll();
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                    yield return new GenerationToken(firstTokenId, emit, FinishReason.Length, timings, firstLogprobInfo);
                    yield break;
                }
                yield return new GenerationToken(firstTokenId, emit, null, Logprobs: firstLogprobInfo);
            }

            // Speculative decode disabled when logprobs requested — no per-position logit access.
            if (_draftModel != null && !captureLogprobs)
            {
                // ── Speculative decode loop ──
                var specDecoder = new SpeculativeDecoder(
                    greedy: pipeline.IsGreedy, seed: options.Seed);
                Core.Attention.IKvCache draftKvCache = AllocateDraftKvCache(cacheSize);
                int[] specBuffer = ArrayPool<int>.Shared.Rent(_speculativeCandidates + 1);
                try
                {
                    PrefillDraftModel(promptIds, draftKvCache);

                    int step = 1;
                    while (step < maxTokens)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        int pos = promptLen + step - 1;
                        if (pos >= cacheSize) break;

                        int remaining = maxTokens - step;
                        int kk = Math.Min(_speculativeCandidates, remaining);

                        var result = specDecoder.DraftAndVerify(
                            _model, _draftModel, kvCache, draftKvCache,
                            pipeline, generatedIds, constraint,
                            pos, vocabSize, _draftModel.Config.VocabSize, kk, specBuffer);

                        if (result.AcceptedCount == 0) break;

                        decodeTicks += result.DraftTicks + result.VerifyTicks;
                        specDrafted += result.DraftedCount;

                        // Constraint is already advanced inside DraftAndVerify — do NOT advance again here.
                        // Only count tokens that actually make it into output.
                        bool shouldBreak = false;
                        for (int i = 0; i < result.AcceptedCount; i++)
                        {
                            int tokenId = specBuffer[i];
                            generatedIds.Add(tokenId);
                            detok.Append(tokenId);

                            stopResult = CheckStopConditions(stopConditions, tokenId, generatedIds,
                                detok.GetTailView(stopTailSize, stopScratch), out int specMatchedIdx);
                            if (stopResult != StopResult.Continue)
                            {
                                bool isStopStringMatch = IsStopStringMatch(stopConditions, specMatchedIdx);
                                finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;

                                if (stopResult == StopResult.Stop)
                                {
                                    _ = streamBuffer.Push(detok.TakeDelta());
                                    string emit = isStopStringMatch
                                        ? streamBuffer.TrimAndFlush(stopConditions)
                                        : streamBuffer.FlushAll();
                                    if (!isStopStringMatch)
                                        generatedIds.RemoveAt(generatedIds.Count - 1);
                                    else
                                        specAccepted++;
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    yield return new GenerationToken(tokenId, emit, finishReason, timings);
                                }
                                else
                                {
                                    specAccepted++;
                                    string emit = streamBuffer.Push(detok.TakeDelta()) + streamBuffer.FlushAll();
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    yield return new GenerationToken(tokenId, emit, finishReason, timings);
                                }
                                shouldBreak = true;
                                yield break;
                            }

                            specAccepted++;

                            // Yield each accepted token
                            {
                                bool isLastStep = (step + 1 >= maxTokens) || (promptLen + step >= cacheSize);
                                string emit = streamBuffer.Push(detok.TakeDelta());
                                if (isLastStep && i == result.AcceptedCount - 1)
                                {
                                    // End of decode loop — drain holdback as the natural-end (Length) emit.
                                    emit += streamBuffer.FlushAll();
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    yield return new GenerationToken(tokenId, emit, FinishReason.Length, timings);
                                    shouldBreak = true;
                                    break;
                                }
                                yield return new GenerationToken(tokenId, emit, null);
                            }

                            step++;
                        }

                        if (shouldBreak) yield break;
                    }

                    // Spec loop exited without a stop-condition path (e.g. AcceptedCount==0,
                    // cache full at top of while). Drain any held-back tail as the natural-end
                    // (Length) emit so the holdback never silently truncates output.
                    if (streamBuffer.PendingLength > 0)
                    {
                        var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                        int lastId = generatedIds.Count > 0 ? generatedIds[^1] : 0;
                        yield return new GenerationToken(lastId, streamBuffer.FlushAll(), FinishReason.Length, timings);
                    }
                }
                finally
                {
                    draftKvCache.Dispose();
                    ArrayPool<int>.Shared.Return(specBuffer);
                }
            }
            else
            {
                // ── Standard decode loop: one token at a time ──
                for (int step = 1; step < maxTokens; step++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int pos = promptLen + step - 1;
                    if (pos >= cacheSize)
                        break;

                    Activity? decodeStepSpan = telemetry.StartDecodeStep(step);

                    int lastToken = generatedIds[^1];
                    int nextTokenId;
                    TokenLogprobInfo? tokenLogprob;

                    long fwdStart = Stopwatch.GetTimestamp();
                    using (ITensor logits = _model.Forward([lastToken], [pos], deviceId: -1, kvCache, adapter))
                    {
                        decodeTicks += Stopwatch.GetTimestamp() - fwdStart;

                        unsafe
                        {
                            using var sampleSpan = telemetry.StartSample();
                            long samplerStart = Stopwatch.GetTimestamp();
                            var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            (nextTokenId, tokenLogprob) = SampleWithLogprobs(logitSpan);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }
                    }

                    decodeStepSpan?.Dispose();
                    constraint?.Advance(nextTokenId);

                    // Mirror prior non-streaming behaviour: logprob added before stop check, so the
                    // discarded-trigger logprob is still recorded (parity assertion in tests).
                    if (tokenLogprob.HasValue) logprobsList?.Add(tokenLogprob.Value);

                    generatedIds.Add(nextTokenId);
                    detok.Append(nextTokenId);

                    stopResult = CheckStopConditions(stopConditions, nextTokenId, generatedIds,
                        detok.GetTailView(stopTailSize, stopScratch), out int decMatchedIdx);
                    if (stopResult != StopResult.Continue)
                    {
                        bool isStopStringMatch = IsStopStringMatch(stopConditions, decMatchedIdx);
                        finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;

                        if (stopResult == StopResult.Stop)
                        {
                            string newDelta = streamBuffer.Push(detok.TakeDelta());
                            string emit = newDelta + (isStopStringMatch
                                ? streamBuffer.TrimAndFlush(stopConditions)
                                : streamBuffer.FlushAll());
                            if (!isStopStringMatch)
                                generatedIds.RemoveAt(generatedIds.Count - 1);
                            // Stop-string match: keep token so KV-cache length matches stored ids.
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                            yield return new GenerationToken(nextTokenId, emit, finishReason, timings, tokenLogprob);
                        }
                        else
                        {
                            string emit = streamBuffer.Push(detok.TakeDelta()) + streamBuffer.FlushAll();
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                            yield return new GenerationToken(nextTokenId, emit, finishReason, timings, tokenLogprob);
                        }
                        yield break;
                    }

                    // Yield token — attach finish reason if this is the last iteration
                    {
                        bool isLastStep = (step + 1 >= maxTokens) || (promptLen + step >= cacheSize);
                        string emit = streamBuffer.Push(detok.TakeDelta());
                        if (isLastStep)
                        {
                            // End of decode loop — drain holdback as the natural-end (Length) emit.
                            emit += streamBuffer.FlushAll();
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                            yield return new GenerationToken(nextTokenId, emit, FinishReason.Length, timings, tokenLogprob);
                            yield break;
                        }
                        yield return new GenerationToken(nextTokenId, emit, null, Logprobs: tokenLogprob);
                    }
                }

                // Standard loop exited via pos>=cacheSize at the top without isLastStep firing.
                // The isLastStep predicate at the prior yield already accounts for this normally,
                // but guard against future refactors leaving the buffer with un-emitted tail.
                if (streamBuffer.PendingLength > 0)
                {
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                    int lastId = generatedIds.Count > 0 ? generatedIds[^1] : 0;
                    yield return new GenerationToken(lastId, streamBuffer.FlushAll(), FinishReason.Length, timings);
                }
            }
        }
        finally
        {
            // Single source of truth for terminal bookkeeping. StoreInPrefixCache and
            // telemetry.Complete each happen exactly once per Generate / streaming call,
            // regardless of which branch terminated the decode loop.
            StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
            telemetry.Complete(promptLen, cachedTokenCount, generatedIds.Count,
                prefillTicks * 1000.0 / Stopwatch.Frequency,
                decodeTicks * 1000.0 / Stopwatch.Frequency,
                finishReason);
            ArrayPool<char>.Shared.Return(stopScratch);
            detok.Dispose();
            if (ownsKvCache)
                kvCache.Dispose();
            telemetry.RequestSpan?.Dispose();
        }
    }

    /// <summary>
    /// Builds the stop-conditions list for a generation call from <paramref name="options"/>.
    /// When <c>options.StopConditions</c> is supplied it is copied verbatim; otherwise the
    /// default set is constructed: EOS + MaxTokens + one <see cref="StopStringCondition"/>
    /// per entry in <c>options.StopSequences</c>. Used by both the public <see cref="Generate"/>
    /// wrapper (for character-level suffix trimming) and <see cref="GenerateCoreAsync"/> (for
    /// the decode-loop stop check) — keeping the two views byte-identical.
    /// </summary>
    private List<IStopCondition> BuildStopConditions(InferenceOptions options, int maxTokens)
    {
        if (options.StopConditions is not null)
            return new List<IStopCondition>(options.StopConditions);

        var list = new List<IStopCondition>
        {
            new EosStopCondition(_tokenizer.EosTokenId),
            new MaxTokensStopCondition(maxTokens)
        };
        foreach (string seq in options.StopSequences)
            list.Add(new StopStringCondition(seq));
        return list;
    }

    /// <summary>
    /// Builds the final <see cref="InferenceResponse"/> from timings that have already been
    /// computed by the core's terminal <see cref="GenerationToken"/>. Performs the same
    /// character-level stop-string suffix trim that the inlined non-streaming path used to do
    /// before <see cref="GenerateCoreAsync"/> became the single source of truth.
    /// </summary>
    private InferenceResponse BuildResponseFromTimings(int promptLen, List<int> generatedIds,
        FinishReason finishReason, InferenceTimings? timings,
        TokenLogprobInfo[]? logprobs, List<IStopCondition> stopConditionsForSuffixTrim)
    {
        string text = generatedIds.Count > 0
            ? _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), stripBosSpace: false)
            : string.Empty;

        // Character-level stop-string suffix trim. When generation stopped because a
        // StopStringCondition matched, the last token is kept in `generatedIds` so the user
        // can see how many tokens were actually emitted, but its decoded text may contain a
        // partial overlap with the stop string. Trim at the char boundary so the returned
        // text excludes the matched suffix.
        if (finishReason == FinishReason.Stop)
            text = StopSuffixTrimmer.TrimMatchedSuffix(text, stopConditionsForSuffixTrim);

        return new InferenceResponse
        {
            GeneratedTokenIds = generatedIds.ToArray(),
            Text = text,
            FinishReason = finishReason,
            PromptTokenCount = promptLen,
            GeneratedTokenCount = generatedIds.Count,
            Timings = timings ?? default,
            Logprobs = logprobs,
        };
    }


    /// <summary>
    /// Resolves the KV-cache to use: either from the prefix cache (on hit) or freshly allocated.
    /// Returns the cache, number of cached tokens, and whether the caller owns (should dispose) the cache.
    /// </summary>
    private (Core.Attention.IKvCache KvCache, int CachedTokenCount, bool OwnsKvCache) ResolveKvCache(
        int[] promptIds, int promptLen, int maxTokens)
    {
        // Cross-request prefix trie (Step 37) takes priority — multiple sessions share blocks.
        if (_prefixTrieManager != null)
        {
            int cacheSize = Math.Min(promptLen + maxTokens, _model.Config.MaxSequenceLength);
            var admission = _prefixTrieManager.Admit(promptIds, cacheSize);
            return (admission.Cache, admission.CachedTokens, true);
        }

        if (_prefixCache != null)
        {
            var (entry, matchedTokens) = _prefixCache.FindMatch(promptIds);

            if (entry != null && matchedTokens > 0 && SupportsPrefixReuse(entry.KvCache))
            {
                // Cache hit — reuse existing KV-cache, truncate to matched prefix
                switch (entry.KvCache)
                {
                    case SimpleKvCache simpleCache:
                        simpleCache.SetCurrentLength(matchedTokens);
                        break;
                    case KvCache.PagedKvCache pagedCache:
                        pagedCache.SetCurrentLength(matchedTokens);
                        break;
                }

                // A cache that only fits the prompt would run out mid-decode and silently terminate
                // generation; require room for the full (prompt + maxTokens) request.
                int requiredSize = promptLen + maxTokens;
                if (entry.KvCache.MaxLength >= requiredSize)
                    return (entry.KvCache, matchedTokens, false);

                // Cache too small — fall through to allocate fresh
            }

            // Cache miss or incompatible — allocate with full model context for future reuse.
            // ownsKvCache=true so that an exception/cancellation between allocation and the
            // Store call below disposes the cache instead of leaking it. StoreInPrefixCache
            // flips this to false only on successful store.
            int cacheSize = Math.Min(promptLen + maxTokens, _model.Config.MaxSequenceLength);
            var kvCache = AllocateKvCache(cacheSize);
            return (kvCache, 0, true);
        }

        // No prefix cache — allocate normally, caller owns
        {
            int cacheSize = Math.Min(promptLen + maxTokens, _model.Config.MaxSequenceLength);
            var kvCache = AllocateKvCache(cacheSize);
            return (kvCache, 0, true);
        }
    }

    // Mirror of the switch in ResolveKvCache — gates StoreInPrefixCache too so we never
    // store cache types we wouldn't be able to reuse (they'd pin RAM/VRAM forever).
    internal static bool SupportsPrefixReuse(Core.Attention.IKvCache kvCache) =>
        kvCache is SimpleKvCache or KvCache.PagedKvCache;

    /// <summary>
    /// Allocates a fresh KV-cache using the factory or default SimpleKvCache.
    /// </summary>
    private Core.Attention.IKvCache AllocateKvCache(int cacheSize)
    {
        return _kvCacheFactory != null
            ? _kvCacheFactory(_model.Config, cacheSize)
            : new SimpleKvCache(
                _model.Config.NumLayers,
                _model.Config.NumKvHeads,
                _model.Config.HeadDim,
                cacheSize);
    }

    /// <summary>
    /// Stores the KV-cache in the prefix cache after generation completes.
    /// Transfers ownership so the cache is not disposed by the caller.
    /// </summary>
    private void StoreInPrefixCache(Core.Attention.IKvCache kvCache, int[] promptIds,
        List<int> generatedIds, ref bool ownsKvCache)
    {
        // Cross-request trie (Step 37): record completion so freshly-computed
        // blocks become available to future sequences, then let Dispose run.
        if (_prefixTrieManager != null && kvCache is KvCache.PagedKvCache paged)
        {
            int total = promptIds.Length + generatedIds.Count;
            var full = ArrayPool<int>.Shared.Rent(total);
            try
            {
                Array.Copy(promptIds, full, promptIds.Length);
                CollectionsMarshal.AsSpan(generatedIds).CopyTo(full.AsSpan(promptIds.Length));
                _prefixTrieManager.RecordCompletion(paged, full.AsSpan(0, total));
            }
            finally
            {
                ArrayPool<int>.Shared.Return(full);
            }
            // ownsKvCache stays unchanged — caller disposes the cache, the trie has
            // already promoted the new blocks to "trie-owned".
            return;
        }

        if (_prefixCache == null)
            return;

        // Only store cache types ResolveKvCache can later reuse — anything else just pins memory.
        if (!SupportsPrefixReuse(kvCache))
            return;

        // Build full token sequence: prompt + generated
        var fullSequence = new int[promptIds.Length + generatedIds.Count];
        Array.Copy(promptIds, fullSequence, promptIds.Length);
        CollectionsMarshal.AsSpan(generatedIds).CopyTo(fullSequence.AsSpan(promptIds.Length));

        _prefixCache.Store(fullSequence, kvCache);
        // Ownership transferred to the prefix cache only after Store succeeds.
        ownsKvCache = false;
    }

    private static StopResult CheckStopConditions(
        List<IStopCondition> conditions, int tokenId,
        IReadOnlyList<int> generatedTokens, ReadOnlySpan<char> decodedTail)
    {
        return CheckStopConditions(conditions, tokenId, generatedTokens, decodedTail, out _);
    }

    private static StopResult CheckStopConditions(
        List<IStopCondition> conditions, int tokenId,
        IReadOnlyList<int> generatedTokens, ReadOnlySpan<char> decodedTail,
        out int matchedIndex)
    {
        for (int i = 0; i < conditions.Count; i++)
        {
            var result = conditions[i].ShouldStop(tokenId, generatedTokens, decodedTail);
            if (result != StopResult.Continue)
            {
                matchedIndex = i;
                return result;
            }
        }
        matchedIndex = -1;
        return StopResult.Continue;
    }

    /// <summary>
    /// True when a <see cref="StopResult.Stop"/> result came from a
    /// <see cref="StopStringCondition"/> match. Determines whether the last token
    /// should be kept in <c>generatedIds</c> (true — its text contains a partial
    /// suffix overlap with the stop string and must be character-trimmed later) or
    /// removed (false — EOS / similar single-token termination where the token's
    /// text is conceptually the terminator itself).
    /// </summary>
    private static bool IsStopStringMatch(List<IStopCondition> conditions, int matchedIndex)
        => matchedIndex >= 0 && matchedIndex < conditions.Count && conditions[matchedIndex] is StopStringCondition;

    // Tail window passed to stop conditions. Must cover the longest stop string currently
    // registered; a safety cushion absorbs future stop strings added via custom conditions.
    private static int ComputeStopTailSize(List<IStopCondition> conditions)
    {
        int maxStopLen = 0;
        for (int i = 0; i < conditions.Count; i++)
        {
            if (conditions[i] is StopStringCondition ssc && ssc.StopString.Length > maxStopLen)
                maxStopLen = ssc.StopString.Length;
        }
        return Math.Max(64, maxStopLen + 16);
    }

    /// <summary>
    /// Prefills the draft model with the full prompt.
    /// </summary>
    private void PrefillDraftModel(int[] promptIds, Core.Attention.IKvCache draftKvCache)
    {
        int promptLen = promptIds.Length;
        int[] positions = ArrayPool<int>.Shared.Rent(promptLen);
        try
        {
            for (int i = 0; i < promptLen; i++)
                positions[i] = i;

            using ITensor _ = _draftModel!.Forward(promptIds, positions.AsSpan(0, promptLen),
                deviceId: -1, draftKvCache);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(positions);
        }
    }

    /// <summary>
    /// Allocates a KV-cache for the draft model.
    /// </summary>
    private Core.Attention.IKvCache AllocateDraftKvCache(int cacheSize)
    {
        if (_draftKvCacheFactory != null)
            return _draftKvCacheFactory(_draftModel!.Config, cacheSize);

        return new SimpleKvCache(
            _draftModel!.Config.NumLayers,
            _draftModel.Config.NumKvHeads,
            _draftModel.Config.HeadDim,
            cacheSize);
    }

    private static InferenceTimings BuildTimings(int promptLen, int generatedCount,
        long prefillTicks, long decodeTicks, long samplerTicks, long kvCacheBytes = 0,
        int cachedTokenCount = 0, int specDrafted = 0, int specAccepted = 0)
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
            KvCacheBytes = kvCacheBytes,
            CachedTokenCount = cachedTokenCount,
            SpeculativeDraftTokens = specDrafted,
            SpeculativeAcceptedTokens = specAccepted
        };
    }

    /// <summary>
    /// Extracts allocated bytes from a KV-cache, regardless of concrete type.
    /// </summary>
    internal static long GetKvCacheBytes(Core.Attention.IKvCache kvCache) => kvCache switch
    {
        KvCache.SimpleKvCache simple => simple.AllocatedBytes,
        KvCache.QuantizedKvCache quantized => quantized.AllocatedBytes,
        _ => 0 // GPU caches — AllocatedBytes is on the concrete type, accessed by CLI directly
    };
}
