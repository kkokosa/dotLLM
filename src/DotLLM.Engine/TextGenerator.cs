using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
    private readonly IModel? _draftModel;
    private readonly Func<ModelConfig, int, Core.Attention.IKvCache>? _draftKvCacheFactory;
    private readonly int _speculativeCandidates;

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
    public TextGenerator(IModel model, ITokenizer tokenizer,
                          Func<ModelConfig, int, Core.Attention.IKvCache>? kvCacheFactory = null,
                          PrefixCache? prefixCache = null,
                          IModel? draftModel = null,
                          Func<ModelConfig, int, Core.Attention.IKvCache>? draftKvCacheFactory = null,
                          int speculativeCandidates = 5)
    {
        _model = model;
        _tokenizer = tokenizer;
        _kvCacheFactory = kvCacheFactory;
        _prefixCache = prefixCache;
        _draftModel = draftModel;
        _draftKvCacheFactory = draftKvCacheFactory;
        _speculativeCandidates = speculativeCandidates;
    }

    /// <summary>
    /// Generates text from the given prompt using the specified options.
    /// </summary>
    /// <param name="prompt">Input text prompt.</param>
    /// <param name="options">Inference options controlling sampling and stopping. Null uses defaults.</param>
    /// <param name="onTokenGenerated">Optional callback invoked after each token is generated, receiving the token ID.</param>
    /// <returns>The inference response with generated text, metadata, and timings.</returns>
    public InferenceResponse Generate(string prompt, InferenceOptions? options = null,
        Action<int>? onTokenGenerated = null)
    {
        options ??= new InferenceOptions();

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

        // Guard: MaxTokens=0 — return immediately, no generation
        if (maxTokens <= 0)
        {
            return new InferenceResponse
            {
                GeneratedTokenIds = [],
                Text = string.Empty,
                FinishReason = FinishReason.Length,
                PromptTokenCount = promptLen,
                GeneratedTokenCount = 0
            };
        }

        // Build sampling pipeline
        var pipeline = new SamplerPipeline(options);

        // Logprobs capture setup
        bool captureLogprobs = options.Logprobs;
        int topLogprobs = Math.Clamp(options.TopLogprobs, 0, 20);
        List<TokenLogprobInfo>? logprobsList = captureLogprobs ? new List<TokenLogprobInfo>(maxTokens) : null;

        // Build decoding constraint for structured output
        IDecodingConstraint? constraint = options.ResponseFormat switch
        {
            ResponseFormat.JsonObject => new JsonConstraint(_tokenizer),
            ResponseFormat.JsonSchema js => new JsonSchemaConstraint(_tokenizer, js.Schema),
            ResponseFormat.Regex rx => new RegexConstraint(_tokenizer, rx.Pattern),
            ResponseFormat.Grammar gr => new GrammarConstraint(_tokenizer, gr.GbnfGrammar),
            _ => null
        };

        // Build stop conditions — use explicit list if provided, otherwise default set
        List<IStopCondition> stopConditions;
        if (options.StopConditions is not null)
        {
            stopConditions = new List<IStopCondition>(options.StopConditions);
        }
        else
        {
            stopConditions = new List<IStopCondition>
            {
                new EosStopCondition(_tokenizer.EosTokenId),
                new MaxTokensStopCondition(maxTokens)
            };
            // TODO: Trim matched suffix only, not entire token (see PR #24 review)
            foreach (string seq in options.StopSequences)
                stopConditions.Add(new StopStringCondition(seq));
        }

        // Resolve KV-cache: reuse from prefix cache or allocate fresh
        var (kvCache, cachedTokenCount, ownsKvCache) = ResolveKvCache(promptIds, promptLen, maxTokens);

        try
        {
            var generatedIds = new List<int>(maxTokens);
            var finishReason = FinishReason.Length;
            long prefillTicks = 0;
            long decodeTicks = 0;
            long samplerTicks = 0;
            int cacheSize = kvCache.MaxLength;

            // Local helper: snapshot log-softmax before sampling (which modifies logits in-place),
            // sample a token, then build logprob info.
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
            long ts0 = Stopwatch.GetTimestamp();

            if (prefillLen > 0)
            {
                // Prefill suffix tokens — span slice avoids array allocation
                ReadOnlySpan<int> suffixTokens = promptIds.AsSpan(prefillStart);
                int[] positionsArray = ArrayPool<int>.Shared.Rent(prefillLen);
                try
                {
                    Span<int> positions = positionsArray.AsSpan(0, prefillLen);
                    for (int i = 0; i < prefillLen; i++)
                        positions[i] = prefillStart + i;

                    using (ITensor prefillLogits = _model.Forward(suffixTokens, positions, deviceId: -1, kvCache))
                    {
                        long ts1 = Stopwatch.GetTimestamp();
                        prefillTicks = ts1 - ts0;

                        unsafe
                        {
                            long samplerStart = Stopwatch.GetTimestamp();
                            // GPU/hybrid models return [1, vocabSize] (last token only);
                            // CPU model returns [seqLen, vocabSize]. Use actual shape to index.
                            float* logitPtr = (float*)prefillLogits.DataPointer;
                            int logitRows = prefillLogits.Shape[0];
                            var logitSpan = new Span<float>(logitPtr + (long)(logitRows - 1) * vocabSize, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            var (tid, lp) = SampleWithLogprobs(logitSpan);
                            firstTokenId = tid;
                            if (lp.HasValue) logprobsList!.Add(lp.Value);
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
                using (ITensor logits = _model.Forward([promptIds[^1]], [promptLen - 1], deviceId: -1, kvCache))
                {
                    long ts1 = Stopwatch.GetTimestamp();
                    prefillTicks = ts1 - ts0;

                    unsafe
                    {
                        long samplerStart = Stopwatch.GetTimestamp();
                        var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
                        if (constraint != null)
                            TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                        var (tid, lp) = SampleWithLogprobs(logitSpan);
                        firstTokenId = tid;
                        if (lp.HasValue) logprobsList!.Add(lp.Value);
                        samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                    }
                }
            }
            else
            {
                // Unreachable: empty prompt guard ensures promptLen >= 1
                throw new InvalidOperationException("Prompt is empty after guard.");
            }

            constraint?.Advance(firstTokenId);

            // Check stop conditions for first token
            generatedIds.Add(firstTokenId);
            string decodedText = _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), stripBosSpace: false);

            var stopResult = CheckStopConditions(stopConditions, firstTokenId, generatedIds, decodedText);
            if (stopResult != StopResult.Continue)
            {
                if (stopResult == StopResult.Stop)
                    generatedIds.RemoveAt(generatedIds.Count - 1);
                else
                    onTokenGenerated?.Invoke(firstTokenId);

                finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                return BuildResponse(promptLen, generatedIds, finishReason,
                    prefillTicks, decodeTicks, samplerTicks, GetKvCacheBytes(kvCache), cachedTokenCount,
                    logprobs: logprobsList?.ToArray());
            }

            onTokenGenerated?.Invoke(firstTokenId);

            int specDrafted = 0, specAccepted = 0;

            // Decode loop (speculative decode disabled when logprobs requested — no per-position logit access)
            if (_draftModel != null && !captureLogprobs)
            {
                // ── Speculative decode loop ──
                var specDecoder = new SpeculativeDecoder(
                    greedy: options.Temperature <= 0f, seed: options.Seed);
                Core.Attention.IKvCache draftKvCache = AllocateDraftKvCache(cacheSize);
                int[] specBuffer = ArrayPool<int>.Shared.Rent(_speculativeCandidates + 1);
                try
                {
                    // Prefill draft model with prompt
                    PrefillDraftModel(promptIds, draftKvCache);

                    int step = 1;
                    while (step < maxTokens)
                    {
                        int pos = promptLen + step - 1;
                        if (pos >= cacheSize) break;

                        int remaining = maxTokens - step;
                        int k = Math.Min(_speculativeCandidates, remaining);

                        var result = specDecoder.DraftAndVerify(
                            _model, _draftModel, kvCache, draftKvCache,
                            pipeline, generatedIds, constraint,
                            pos, vocabSize, _draftModel.Config.VocabSize, k, specBuffer);

                        if (result.AcceptedCount == 0) break;

                        decodeTicks += result.DraftTicks + result.VerifyTicks;
                        specDrafted += result.DraftedCount;

                        // Constraint is already advanced inside DraftAndVerify — do NOT advance again here.
                        // Only count tokens that actually make it into output (stop conditions may discard some).
                        bool shouldBreak = false;
                        for (int i = 0; i < result.AcceptedCount; i++)
                        {
                            int tokenId = specBuffer[i];
                            generatedIds.Add(tokenId);
                            decodedText = _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), stripBosSpace: false);

                            stopResult = CheckStopConditions(stopConditions, tokenId, generatedIds, decodedText);
                            if (stopResult != StopResult.Continue)
                            {
                                if (stopResult == StopResult.Stop)
                                    generatedIds.RemoveAt(generatedIds.Count - 1);
                                else
                                {
                                    specAccepted++;
                                    onTokenGenerated?.Invoke(tokenId);
                                }

                                finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                                shouldBreak = true;
                                break;
                            }

                            specAccepted++;
                            onTokenGenerated?.Invoke(tokenId);
                            step++;
                        }

                        if (shouldBreak) break;
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
                    int pos = promptLen + step - 1;
                    if (pos >= cacheSize)
                        break;

                    int lastToken = generatedIds[^1];
                    int nextTokenId;

                    long fwdStart = Stopwatch.GetTimestamp();
                    using (ITensor logits = _model.Forward([lastToken], [pos], deviceId: -1, kvCache))
                    {
                        decodeTicks += Stopwatch.GetTimestamp() - fwdStart;

                        unsafe
                        {
                            long samplerStart = Stopwatch.GetTimestamp();
                            var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            var (tid, lp) = SampleWithLogprobs(logitSpan);
                            nextTokenId = tid;
                            if (lp.HasValue) logprobsList!.Add(lp.Value);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }
                    }

                    constraint?.Advance(nextTokenId);

                    generatedIds.Add(nextTokenId);
                    decodedText = _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), stripBosSpace: false);

                    stopResult = CheckStopConditions(stopConditions, nextTokenId, generatedIds, decodedText);
                    if (stopResult != StopResult.Continue)
                    {
                        if (stopResult == StopResult.Stop)
                            generatedIds.RemoveAt(generatedIds.Count - 1);
                        else
                            onTokenGenerated?.Invoke(nextTokenId);

                        finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                        break;
                    }

                    onTokenGenerated?.Invoke(nextTokenId);
                }
            }

            StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
            return BuildResponse(promptLen, generatedIds, finishReason,
                prefillTicks, decodeTicks, samplerTicks, GetKvCacheBytes(kvCache), cachedTokenCount,
                specDrafted, specAccepted, logprobsList?.ToArray());
        }
        finally
        {
            if (ownsKvCache)
                kvCache.Dispose();
        }
    }

    /// <summary>
    /// Streams generated tokens as an async enumerable, yielding each token with incremental text,
    /// finish reason, and timings on the final token.
    /// </summary>
    /// <param name="prompt">Input text prompt.</param>
    /// <param name="options">Inference options controlling sampling and stopping. Null uses defaults.</param>
    /// <param name="cancellationToken">Token to cancel generation cooperatively between decode steps.</param>
    /// <returns>An async enumerable of <see cref="GenerationToken"/> values.</returns>
    public async IAsyncEnumerable<GenerationToken> GenerateStreamingTokensAsync(
        string prompt,
        InferenceOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new InferenceOptions();

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

        // Guard: MaxTokens=0 — yield nothing
        if (maxTokens <= 0)
            yield break;

        cancellationToken.ThrowIfCancellationRequested();

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
        List<IStopCondition> stopConditions;
        if (options.StopConditions is not null)
        {
            stopConditions = new List<IStopCondition>(options.StopConditions);
        }
        else
        {
            stopConditions = new List<IStopCondition>
            {
                new EosStopCondition(_tokenizer.EosTokenId),
                new MaxTokensStopCondition(maxTokens)
            };
            foreach (string seq in options.StopSequences)
                stopConditions.Add(new StopStringCondition(seq));
        }

        // Resolve KV-cache: reuse from prefix cache or allocate fresh
        var (kvCache, cachedTokenCount, ownsKvCache) = ResolveKvCache(promptIds, promptLen, maxTokens);
        long kvBytes = GetKvCacheBytes(kvCache);

        try
        {
            var generatedIds = new List<int>(maxTokens);
            long prefillTicks = 0;
            long decodeTicks = 0;
            long samplerTicks = 0;
            int previousDecodeLength = 0;
            int cacheSize = kvCache.MaxLength;

            // Local helper: snapshot log-softmax before sampling (which modifies logits in-place),
            // sample a token, then build logprob info.
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

            if (prefillLen > 0)
            {
                // Span slice avoids array allocation for suffix tokens
                ReadOnlySpan<int> suffixTokens = promptIds.AsSpan(prefillStart);
                int[] positionsArray = ArrayPool<int>.Shared.Rent(prefillLen);
                try
                {
                    Span<int> positions = positionsArray.AsSpan(0, prefillLen);
                    for (int i = 0; i < prefillLen; i++)
                        positions[i] = prefillStart + i;

                    using (ITensor prefillLogits = _model.Forward(suffixTokens, positions, deviceId: -1, kvCache))
                    {
                        long ts1 = Stopwatch.GetTimestamp();
                        prefillTicks = ts1 - ts0;

                        unsafe
                        {
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
                using (ITensor logits = _model.Forward([promptIds[^1]], [promptLen - 1], deviceId: -1, kvCache))
                {
                    long ts1 = Stopwatch.GetTimestamp();
                    prefillTicks = ts1 - ts0;

                    unsafe
                    {
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

            constraint?.Advance(firstTokenId);

            // Check stop conditions for first token
            generatedIds.Add(firstTokenId);
            string decodedText = _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), stripBosSpace: false);

            var stopResult = CheckStopConditions(stopConditions, firstTokenId, generatedIds, decodedText);
            if (stopResult != StopResult.Continue)
            {
                var fr = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;

                if (stopResult == StopResult.Stop)
                {
                    generatedIds.RemoveAt(generatedIds.Count - 1);
                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                    yield return new GenerationToken(firstTokenId, string.Empty, fr, timings, firstLogprobInfo);
                }
                else
                {
                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                    string text = decodedText[previousDecodeLength..];
                    yield return new GenerationToken(firstTokenId, text, fr, timings, firstLogprobInfo);
                }
                yield break;
            }

            // Yield first token — check if it's also the last (maxTokens == 1)
            {
                bool firstIsLast = maxTokens <= 1;
                string text = decodedText[previousDecodeLength..];
                if (firstIsLast)
                {
                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                    yield return new GenerationToken(firstTokenId, text, FinishReason.Length, timings, firstLogprobInfo);
                    yield break;
                }
                previousDecodeLength = decodedText.Length;
                yield return new GenerationToken(firstTokenId, text, null, Logprobs: firstLogprobInfo);
            }

            int specDrafted = 0, specAccepted = 0;

            // Speculative decode disabled when logprobs requested — no per-position logit access
            if (_draftModel != null && !captureLogprobs)
            {
                // ── Speculative decode loop ──
                var specDecoder = new SpeculativeDecoder(
                    greedy: options.Temperature <= 0f, seed: options.Seed);
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
                            decodedText = _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), stripBosSpace: false);

                            stopResult = CheckStopConditions(stopConditions, tokenId, generatedIds, decodedText);
                            if (stopResult != StopResult.Continue)
                            {
                                var fr = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                                if (stopResult == StopResult.Stop)
                                {
                                    generatedIds.RemoveAt(generatedIds.Count - 1);
                                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    yield return new GenerationToken(tokenId, string.Empty, fr, timings);
                                }
                                else
                                {
                                    specAccepted++;
                                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    string text = decodedText[previousDecodeLength..];
                                    yield return new GenerationToken(tokenId, text, fr, timings);
                                }
                                shouldBreak = true;
                                yield break;
                            }

                            specAccepted++;

                            // Yield each accepted token
                            {
                                bool isLastStep = (step + 1 >= maxTokens) || (promptLen + step >= cacheSize);
                                string text = decodedText[previousDecodeLength..];
                                if (isLastStep && i == result.AcceptedCount - 1)
                                {
                                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    yield return new GenerationToken(tokenId, text, FinishReason.Length, timings);
                                    shouldBreak = true;
                                    break;
                                }
                                previousDecodeLength = decodedText.Length;
                                yield return new GenerationToken(tokenId, text, null);
                            }

                            step++;
                        }

                        if (shouldBreak) yield break;
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

                    int lastToken = generatedIds[^1];
                    int nextTokenId;
                    TokenLogprobInfo? tokenLogprob;

                    long fwdStart = Stopwatch.GetTimestamp();
                    using (ITensor logits = _model.Forward([lastToken], [pos], deviceId: -1, kvCache))
                    {
                        decodeTicks += Stopwatch.GetTimestamp() - fwdStart;

                        unsafe
                        {
                            long samplerStart = Stopwatch.GetTimestamp();
                            var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            (nextTokenId, tokenLogprob) = SampleWithLogprobs(logitSpan);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }
                    }

                    constraint?.Advance(nextTokenId);

                    generatedIds.Add(nextTokenId);
                    decodedText = _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), stripBosSpace: false);

                    stopResult = CheckStopConditions(stopConditions, nextTokenId, generatedIds, decodedText);
                    if (stopResult != StopResult.Continue)
                    {
                        var fr = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;

                        if (stopResult == StopResult.Stop)
                        {
                            generatedIds.RemoveAt(generatedIds.Count - 1);
                            StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                            yield return new GenerationToken(nextTokenId, string.Empty, fr, timings, tokenLogprob);
                        }
                        else
                        {
                            StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                            string text = decodedText[previousDecodeLength..];
                            yield return new GenerationToken(nextTokenId, text, fr, timings, tokenLogprob);
                        }
                        yield break;
                    }

                    // Yield token — attach finish reason if this is the last iteration
                    {
                        bool isLastStep = (step + 1 >= maxTokens) || (promptLen + step >= cacheSize);
                        string text = decodedText[previousDecodeLength..];
                        if (isLastStep)
                        {
                            StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                            yield return new GenerationToken(nextTokenId, text, FinishReason.Length, timings, tokenLogprob);
                            yield break;
                        }
                        previousDecodeLength = decodedText.Length;
                        yield return new GenerationToken(nextTokenId, text, null, Logprobs: tokenLogprob);
                    }
                }
            }
        }
        finally
        {
            if (ownsKvCache)
                kvCache.Dispose();
        }
    }

    /// <summary>
    /// Streams generated text as an async enumerable, yielding incremental text fragments.
    /// This is a convenience wrapper over <see cref="GenerateStreamingTokensAsync"/>.
    /// </summary>
    /// <param name="prompt">Input text prompt.</param>
    /// <param name="options">Inference options controlling sampling and stopping. Null uses defaults.</param>
    /// <param name="cancellationToken">Token to cancel generation cooperatively between decode steps.</param>
    /// <returns>An async enumerable of incremental text strings.</returns>
    public async IAsyncEnumerable<string> GenerateStreamingAsync(
        string prompt,
        InferenceOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var token in GenerateStreamingTokensAsync(prompt, options, cancellationToken))
            yield return token.Text;
    }

    /// <summary>
    /// Resolves the KV-cache to use: either from the prefix cache (on hit) or freshly allocated.
    /// Returns the cache, number of cached tokens, and whether the caller owns (should dispose) the cache.
    /// </summary>
    private (Core.Attention.IKvCache KvCache, int CachedTokenCount, bool OwnsKvCache) ResolveKvCache(
        int[] promptIds, int promptLen, int maxTokens)
    {
        if (_prefixCache != null)
        {
            var (entry, matchedTokens) = _prefixCache.FindMatch(promptIds);

            if (entry != null && matchedTokens > 0)
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
                    default:
                        // Unsupported cache type for prefix reuse — fall through to allocate fresh
                        goto cacheMiss;
                }

                // Verify the cache is large enough for the new prompt + generation
                int requiredSize = promptLen + maxTokens;
                if (entry.KvCache.MaxLength >= requiredSize || entry.KvCache.MaxLength >= promptLen)
                    return (entry.KvCache, matchedTokens, false);

                // Cache too small — fall through to allocate fresh
            }
            cacheMiss:

            // Cache miss or incompatible — allocate with full model context for future reuse
            int cacheSize = Math.Min(promptLen + maxTokens, _model.Config.MaxSequenceLength);
            var kvCache = AllocateKvCache(cacheSize);
            return (kvCache, 0, false); // ownsKvCache=false: will be transferred to prefix cache
        }

        // No prefix cache — allocate normally, caller owns
        {
            int cacheSize = Math.Min(promptLen + maxTokens, _model.Config.MaxSequenceLength);
            var kvCache = AllocateKvCache(cacheSize);
            return (kvCache, 0, true);
        }
    }

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
        if (_prefixCache == null)
            return;

        // Build full token sequence: prompt + generated
        var fullSequence = new int[promptIds.Length + generatedIds.Count];
        Array.Copy(promptIds, fullSequence, promptIds.Length);
        CollectionsMarshal.AsSpan(generatedIds).CopyTo(fullSequence.AsSpan(promptIds.Length));

        _prefixCache.Store(fullSequence, kvCache);
        ownsKvCache = false;
    }

    private static StopResult CheckStopConditions(
        List<IStopCondition> conditions, int tokenId,
        IReadOnlyList<int> generatedTokens, string decodedText)
    {
        for (int i = 0; i < conditions.Count; i++)
        {
            var result = conditions[i].ShouldStop(tokenId, generatedTokens, decodedText);
            if (result != StopResult.Continue)
                return result;
        }
        return StopResult.Continue;
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

    private InferenceResponse BuildResponse(int promptLen, List<int> generatedIds,
        FinishReason finishReason, long prefillTicks, long decodeTicks, long samplerTicks,
        long kvCacheBytes = 0, int cachedTokenCount = 0,
        int specDrafted = 0, int specAccepted = 0,
        TokenLogprobInfo[]? logprobs = null)
    {
        string text = generatedIds.Count > 0
            ? _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), stripBosSpace: false)
            : string.Empty;

        return new InferenceResponse
        {
            GeneratedTokenIds = generatedIds.ToArray(),
            Text = text,
            FinishReason = finishReason,
            PromptTokenCount = promptLen,
            GeneratedTokenCount = generatedIds.Count,
            Timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvCacheBytes, cachedTokenCount, specDrafted, specAccepted),
            Logprobs = logprobs,
        };
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
