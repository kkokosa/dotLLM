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

    /// <summary>
    /// Creates a new text generator.
    /// </summary>
    /// <param name="model">The model to use for forward passes.</param>
    /// <param name="tokenizer">The tokenizer for encoding/decoding text.</param>
    /// <param name="kvCacheFactory">Optional factory for creating a KV-cache. When null, uses <see cref="SimpleKvCache"/>.
    /// Parameters: (config, maxSeqLen).</param>
    public TextGenerator(IModel model, ITokenizer tokenizer,
                          Func<ModelConfig, int, Core.Attention.IKvCache>? kvCacheFactory = null)
    {
        _model = model;
        _tokenizer = tokenizer;
        _kvCacheFactory = kvCacheFactory;
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

        // Build decoding constraint for structured output
        IDecodingConstraint? constraint = options.ResponseFormat switch
        {
            ResponseFormat.JsonObject => new JsonConstraint(_tokenizer),
            ResponseFormat.JsonSchema js => new JsonSchemaConstraint(_tokenizer, js.Schema),
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

        // Allocate KV-cache
        int cacheSize = Math.Min(promptLen + maxTokens, _model.Config.MaxSequenceLength);
        using var kvCache = _kvCacheFactory != null
            ? _kvCacheFactory(_model.Config, cacheSize)
            : new SimpleKvCache(
                _model.Config.NumLayers,
                _model.Config.NumKvHeads,
                _model.Config.HeadDim,
                cacheSize);

        var generatedIds = new List<int>(maxTokens);
        var finishReason = FinishReason.Length;
        long prefillTicks = 0;
        long decodeTicks = 0;
        long samplerTicks = 0;

        // Prefill: run full prompt through the model
        int[] positions = new int[promptLen];
        for (int i = 0; i < promptLen; i++)
            positions[i] = i;

        int firstTokenId;
        long ts0 = Stopwatch.GetTimestamp();
        using (ITensor prefillLogits = _model.Forward(promptIds, positions, deviceId: -1, kvCache))
        {
            long ts1 = Stopwatch.GetTimestamp();
            prefillTicks = ts1 - ts0;

            unsafe
            {
                long samplerStart = Stopwatch.GetTimestamp();
                var logitSpan = new Span<float>((void*)prefillLogits.DataPointer, vocabSize);
                if (constraint != null)
                    TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                firstTokenId = pipeline.Sample(logitSpan, generatedIds);
                samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
            }
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
            return BuildResponse(promptLen, generatedIds, finishReason,
                prefillTicks, decodeTicks, samplerTicks, GetKvCacheBytes(kvCache));
        }

        onTokenGenerated?.Invoke(firstTokenId);

        // Decode loop: one token at a time
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
                    nextTokenId = pipeline.Sample(logitSpan, generatedIds);
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

        return BuildResponse(promptLen, generatedIds, finishReason,
            prefillTicks, decodeTicks, samplerTicks, GetKvCacheBytes(kvCache));
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

        // Build decoding constraint for structured output
        IDecodingConstraint? constraint = options.ResponseFormat switch
        {
            ResponseFormat.JsonObject => new JsonConstraint(_tokenizer),
            ResponseFormat.JsonSchema js => new JsonSchemaConstraint(_tokenizer, js.Schema),
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

        // Allocate KV-cache — disposed when the enumerator completes/disposes
        int cacheSize = Math.Min(promptLen + maxTokens, _model.Config.MaxSequenceLength);
        using var kvCache = _kvCacheFactory != null
            ? _kvCacheFactory(_model.Config, cacheSize)
            : new SimpleKvCache(
                _model.Config.NumLayers,
                _model.Config.NumKvHeads,
                _model.Config.HeadDim,
                cacheSize);
        long kvBytes = GetKvCacheBytes(kvCache);

        var generatedIds = new List<int>(maxTokens);
        long prefillTicks = 0;
        long decodeTicks = 0;
        long samplerTicks = 0;
        int previousDecodeLength = 0;

        // Prefill: run full prompt through the model
        int[] positions = new int[promptLen];
        for (int i = 0; i < promptLen; i++)
            positions[i] = i;

        int firstTokenId;
        long ts0 = Stopwatch.GetTimestamp();
        using (ITensor prefillLogits = _model.Forward(promptIds, positions, deviceId: -1, kvCache))
        {
            long ts1 = Stopwatch.GetTimestamp();
            prefillTicks = ts1 - ts0;

            unsafe
            {
                long samplerStart = Stopwatch.GetTimestamp();
                var logitSpan = new Span<float>((void*)prefillLogits.DataPointer, vocabSize);
                if (constraint != null)
                    TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                firstTokenId = pipeline.Sample(logitSpan, generatedIds);
                samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
            }
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
                // Token excluded — match sync path which removes it before computing timings
                generatedIds.RemoveAt(generatedIds.Count - 1);
                var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes);
                yield return new GenerationToken(firstTokenId, string.Empty, fr, timings);
            }
            else
            {
                // Token included
                var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes);
                string text = decodedText[previousDecodeLength..];
                yield return new GenerationToken(firstTokenId, text, fr, timings);
            }
            yield break;
        }

        // Yield first token — check if it's also the last (maxTokens == 1)
        {
            bool firstIsLast = maxTokens <= 1;
            string text = decodedText[previousDecodeLength..];
            if (firstIsLast)
            {
                var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes);
                yield return new GenerationToken(firstTokenId, text, FinishReason.Length, timings);
                yield break;
            }
            previousDecodeLength = decodedText.Length;
            yield return new GenerationToken(firstTokenId, text, null);
        }

        // Decode loop: one token at a time
        for (int step = 1; step < maxTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

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
                    nextTokenId = pipeline.Sample(logitSpan, generatedIds);
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
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes);
                    yield return new GenerationToken(nextTokenId, string.Empty, fr, timings);
                }
                else
                {
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes);
                    string text = decodedText[previousDecodeLength..];
                    yield return new GenerationToken(nextTokenId, text, fr, timings);
                }
                yield break;
            }

            // Yield token — attach finish reason if this is the last iteration
            {
                bool isLastStep = (step + 1 >= maxTokens) || (promptLen + step >= cacheSize);
                string text = decodedText[previousDecodeLength..];
                if (isLastStep)
                {
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes);
                    yield return new GenerationToken(nextTokenId, text, FinishReason.Length, timings);
                    yield break;
                }
                previousDecodeLength = decodedText.Length;
                yield return new GenerationToken(nextTokenId, text, null);
            }
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

    private InferenceResponse BuildResponse(int promptLen, List<int> generatedIds,
        FinishReason finishReason, long prefillTicks, long decodeTicks, long samplerTicks,
        long kvCacheBytes = 0)
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
            Timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvCacheBytes)
        };
    }

    private static InferenceTimings BuildTimings(int promptLen, int generatedCount,
        long prefillTicks, long decodeTicks, long samplerTicks, long kvCacheBytes = 0)
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
            KvCacheBytes = kvCacheBytes
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
