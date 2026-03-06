using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Models.Architectures;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Autoregressive text generator: encodes a prompt, runs prefill + decode loop
/// with sampling and stop conditions, and returns the generated text.
/// </summary>
public sealed class TextGenerator
{
    private readonly LlamaModel _model;
    private readonly ITokenizer _tokenizer;

    /// <summary>
    /// Creates a new text generator.
    /// </summary>
    /// <param name="model">The model to use for forward passes.</param>
    /// <param name="tokenizer">The tokenizer for encoding/decoding text.</param>
    public TextGenerator(LlamaModel model, ITokenizer tokenizer)
    {
        _model = model;
        _tokenizer = tokenizer;
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
        using var kvCache = new SimpleKvCache(
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
                firstTokenId = pipeline.Sample(logitSpan, generatedIds);
                samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
            }
        }

        // Check stop conditions for first token
        generatedIds.Add(firstTokenId);
        string decodedText = _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds));

        var stopResult = CheckStopConditions(stopConditions, firstTokenId, generatedIds, decodedText);
        if (stopResult != StopResult.Continue)
        {
            if (stopResult == StopResult.Stop)
                generatedIds.RemoveAt(generatedIds.Count - 1);
            else
                onTokenGenerated?.Invoke(firstTokenId);

            finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
            return BuildResponse(promptLen, generatedIds, finishReason,
                prefillTicks, decodeTicks, samplerTicks);
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
                    nextTokenId = pipeline.Sample(logitSpan, generatedIds);
                    samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                }
            }

            generatedIds.Add(nextTokenId);
            decodedText = _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds));

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
            prefillTicks, decodeTicks, samplerTicks);
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
        FinishReason finishReason, long prefillTicks, long decodeTicks, long samplerTicks)
    {
        string text = generatedIds.Count > 0
            ? _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds))
            : string.Empty;

        double tickFreq = Stopwatch.Frequency;
        int decodeSteps = generatedIds.Count > 1 ? generatedIds.Count - 1 : 0;

        return new InferenceResponse
        {
            GeneratedTokenIds = generatedIds.ToArray(),
            Text = text,
            FinishReason = finishReason,
            PromptTokenCount = promptLen,
            GeneratedTokenCount = generatedIds.Count,
            Timings = new InferenceTimings
            {
                PrefillTimeMs = prefillTicks / tickFreq * 1000.0,
                DecodeTimeMs = decodeTicks / tickFreq * 1000.0,
                SamplingTimeMs = samplerTicks / tickFreq * 1000.0,
                PrefillTokenCount = promptLen,
                DecodeTokenCount = decodeSteps
            }
        };
    }
}
