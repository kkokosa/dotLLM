using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Regression tests for the <c>Generate</c> / <c>GenerateStreamingTokensAsync</c>
/// consolidation (upstream issue #121 item #10). The consolidation extracted a single
/// internal <c>GenerateCoreAsync</c> source-of-truth that both public entry points consume.
/// These tests pin the invariants that the duplicated orchestration was at risk of
/// drifting on — namely, parity between the synchronous and streaming paths, the subtle
/// stop-discarded-token semantics (EOS-discard vs stop-string-keep), the once-per-call
/// terminal-cleanup contract for <c>StoreInPrefixCache</c>, the <c>onTokenGenerated</c>
/// callback invocation contract, and the incidental telemetry-FinishReason bug-fix.
/// </summary>
public sealed class TextGeneratorConsolidationTests
{
    private const int VocabSize = 16;
    private const int NumLayers = 1;
    private const int NumKvHeads = 1;
    private const int HeadDim = 4;

    /// <summary>
    /// Same prompt + same temperature-0 seed through both <see cref="TextGenerator.Generate"/>
    /// and <see cref="TextGenerator.GenerateStreamingTokensAsync"/>: ids must be byte-identical,
    /// concatenated streaming text must equal non-streaming text, finish reasons must match,
    /// counts must match. This is the headline parity guarantee of the consolidation.
    /// </summary>
    /// <remarks>
    /// The text-equality check works here because EOS decodes to <see cref="string.Empty"/> in
    /// this fixture — the standard real-tokenizer case. For an EOS that decodes to non-empty
    /// text the two paths diverge by pre-existing design; see
    /// <see cref="EosWithNonEmptyDecode_GenerateExcludesText_StreamingFlushesIt_PreservedByteIdentical"/>.
    /// </remarks>
    [Fact]
    public async Task GenerateAndStreaming_ProduceByteIdenticalOutputs()
    {
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = string.Empty,
            [2] = "Hello",
            [3] = ", ",
            [4] = "world",
            [5] = "!",
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);
        var argmaxSequence = new[] { 2, 3, 4, 5, 1 }; // ends on EOS

        var optionsTemplate = new InferenceOptions
        {
            MaxTokens = 16,
            Temperature = 0f,
        };

        // ── Non-streaming ──
        var modelA = new ScriptedSequenceModel(argmaxSequence);
        var generatorA = new TextGenerator(modelA, tokenizer);
        var response = generatorA.Generate("hi", optionsTemplate);

        // ── Streaming ──
        var modelB = new ScriptedSequenceModel(argmaxSequence);
        var generatorB = new TextGenerator(modelB, tokenizer);
        var streamedIds = new List<int>();
        var streamedText = new System.Text.StringBuilder();
        FinishReason? streamedReason = null;
        int streamedTokenCount = 0;
        InferenceTimings? streamedTimings = null;
        await foreach (var tok in generatorB.GenerateStreamingTokensAsync("hi", optionsTemplate))
        {
            // Mirror the wrapper's add-to-list semantics so the assertion compares apples to apples.
            if (tok.FinishReason != FinishReason.Stop)
                streamedIds.Add(tok.TokenId);
            streamedText.Append(tok.Text);
            if (tok.FinishReason.HasValue)
            {
                streamedReason = tok.FinishReason.Value;
                streamedTimings = tok.Timings;
            }
            streamedTokenCount++;
        }

        Assert.Equal(response.GeneratedTokenIds, streamedIds.ToArray());
        Assert.Equal(response.FinishReason, streamedReason);
        Assert.Equal(response.Text, streamedText.ToString());
        Assert.Equal(response.GeneratedTokenCount, streamedIds.Count);
        Assert.Equal(response.PromptTokenCount, streamedTimings!.Value.PrefillTokenCount);
    }

    /// <summary>
    /// EOS-discard case. When the EOS token triggers <see cref="StopResult.Stop"/>, the
    /// triggering token id is NOT added to the visible output id list of either path.
    /// Documents the pre-existing streaming-vs-Generate text divergence on EOS-with-non-empty-decode
    /// so future readers don't mistake the consolidation for full text-parity unification.
    /// </summary>
    /// <remarks>
    /// Pre-consolidation: non-streaming removed the EOS id from <c>generatedIds</c> AND rebuilt text
    /// from the trimmed ids, so the EOS-decoded text never reached the caller. Streaming also removed
    /// the id from <c>generatedIds</c> but still emitted the EOS-decoded text in the terminal yield's
    /// delta — a long-standing streaming quirk that is OUT OF SCOPE for this consolidation and is
    /// intentionally preserved byte-identical (SSE consumers handle it by ignoring text on a
    /// Stop-finish-reason chunk).
    /// </remarks>
    [Fact]
    public async Task StopOnEos_DiscardsTriggeringTokenIdFromBothPaths()
    {
        // BOS=0, EOS=1; sequence: 2, 3, then EOS=1 fires.
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = "<eos-marker>",
            [2] = "Hi",
            [3] = "!",
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);
        var argmaxSequence = new[] { 2, 3, 1 };
        var options = new InferenceOptions { MaxTokens = 16, Temperature = 0f };

        // ── Non-streaming: text rebuilt from generatedIds (which has EOS removed) ──
        var modelA = new ScriptedSequenceModel(argmaxSequence);
        var response = new TextGenerator(modelA, tokenizer).Generate("hi", options);

        Assert.Equal(FinishReason.Stop, response.FinishReason);
        Assert.Equal(new[] { 2, 3 }, response.GeneratedTokenIds); // EOS id 1 must NOT be present.
        Assert.Equal("Hi!", response.Text);
        Assert.DoesNotContain("<eos-marker>", response.Text);

        // ── Streaming: same id-list invariant, but the terminal yield's text intentionally
        //    flushes the just-decoded EOS delta (existing behaviour preserved byte-identical). ──
        var modelB = new ScriptedSequenceModel(argmaxSequence);
        var streamedIds = new List<int>();
        FinishReason? streamedReason = null;
        int terminalTokenId = -1;
        await foreach (var tok in new TextGenerator(modelB, tokenizer).GenerateStreamingTokensAsync("hi", options))
        {
            if (tok.FinishReason != FinishReason.Stop)
                streamedIds.Add(tok.TokenId);
            if (tok.FinishReason.HasValue)
            {
                streamedReason = tok.FinishReason.Value;
                terminalTokenId = tok.TokenId;
            }
        }

        Assert.Equal(FinishReason.Stop, streamedReason);
        Assert.Equal(new[] { 2, 3 }, streamedIds.ToArray());
        // The terminal token carries the EOS id (as the trigger), even though it's excluded
        // from streamedIds by the wrapper rule above. This is the discriminator scope §3
        // names — and the test that distinguishes the "kept" vs "discarded" semantics.
        Assert.Equal(1, terminalTokenId);
    }

    /// <summary>
    /// Pins the deliberate text-handling divergence between <see cref="TextGenerator.Generate"/>
    /// and <see cref="TextGenerator.GenerateStreamingTokensAsync"/> when an EOS token has a
    /// non-empty decoded text. Generate rebuilds text from trimmed ids → EOS text excluded.
    /// Streaming flushes the just-decoded delta in the terminal yield → EOS text appears.
    /// The consolidation preserves this byte-identical because unifying it is outside scope
    /// §10 (it would touch BuildResponse/streaming-emit semantics that the maintainer didn't ask for).
    /// </summary>
    [Fact]
    public async Task EosWithNonEmptyDecode_GenerateExcludesText_StreamingFlushesIt_PreservedByteIdentical()
    {
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = "[EOS]",
            [2] = "Hi",
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);
        var argmaxSequence = new[] { 2, 1 };
        var options = new InferenceOptions { MaxTokens = 16, Temperature = 0f };

        var modelA = new ScriptedSequenceModel(argmaxSequence);
        var response = new TextGenerator(modelA, tokenizer).Generate("hi", options);
        Assert.Equal("Hi", response.Text);
        Assert.DoesNotContain("[EOS]", response.Text);

        var modelB = new ScriptedSequenceModel(argmaxSequence);
        var streamed = new System.Text.StringBuilder();
        await foreach (var tok in new TextGenerator(modelB, tokenizer).GenerateStreamingTokensAsync("hi", options))
            streamed.Append(tok.Text);
        // Pre-existing streaming behaviour: terminal yield flushes the EOS delta.
        Assert.Contains("[EOS]", streamed.ToString());
        Assert.StartsWith("Hi", streamed.ToString());
    }

    /// <summary>
    /// Stop-string case. The triggering token IS KEPT in <c>generatedIds</c> (so the
    /// KV-cache length stays in sync), but its decoded text is character-trimmed by
    /// <see cref="StopSuffixTrimmer"/>. This distinguishes the consolidation from a
    /// naïve "Stop → remove" treatment that would mishandle stop strings.
    /// </summary>
    [Fact]
    public void StopOnStopString_KeepsTriggeringTokenInIdList()
    {
        // Sequence decodes to: "Hello, world<|im_end|>" where the last token contains the
        // stop-string suffix. Generation must stop, but the last id stays in generatedIds.
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = string.Empty,
            [2] = "Hello, world",
            [3] = "<|im_end|>", // single token equals the entire stop string
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);
        var argmaxSequence = new[] { 2, 3 };
        var options = new InferenceOptions
        {
            MaxTokens = 16,
            Temperature = 0f,
            StopSequences = new List<string> { "<|im_end|>" },
        };

        var model = new ScriptedSequenceModel(argmaxSequence);
        var response = new TextGenerator(model, tokenizer).Generate("hi", options);

        Assert.Equal(FinishReason.Stop, response.FinishReason);
        // Stop-string case: token 3 IS kept in the id list.
        Assert.Equal(new[] { 2, 3 }, response.GeneratedTokenIds);
        // But its decoded suffix is trimmed from the returned text.
        Assert.Equal("Hello, world", response.Text);
        Assert.DoesNotContain("<|im_end|>", response.Text);
    }

    /// <summary>
    /// Telemetry-FinishReason regression. Pre-consolidation the streaming finally hard-coded
    /// <c>FinishReason.Length</c> on <c>telemetry.Complete</c>, so a stop-string termination
    /// reported the wrong reason. The consolidated finally passes the captured reason.
    /// The observable proxy here is that the streaming terminal token's <c>FinishReason</c>
    /// matches the non-streaming response — and the in-flight test below
    /// (<see cref="GenerateAndStreaming_ProduceByteIdenticalOutputs"/>) already exercises this
    /// for natural-end. This test exercises it specifically for the Stop path.
    /// </summary>
    [Fact]
    public async Task TelemetryFinishReason_PassesActualReason_OnStopStringStreamingTermination()
    {
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = string.Empty,
            [2] = "Hello",
            [3] = "<|im_end|>",
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);
        var argmaxSequence = new[] { 2, 3 };
        var options = new InferenceOptions
        {
            MaxTokens = 16,
            Temperature = 0f,
            StopSequences = new List<string> { "<|im_end|>" },
        };

        var model = new ScriptedSequenceModel(argmaxSequence);
        FinishReason? terminalReason = null;
        await foreach (var tok in new TextGenerator(model, tokenizer).GenerateStreamingTokensAsync("hi", options))
        {
            if (tok.FinishReason.HasValue) terminalReason = tok.FinishReason.Value;
        }

        // Pre-fix: terminal would have carried Length even though the loop stopped on a stop-string.
        // Post-fix: terminal carries the real reason — Stop — and telemetry.Complete sees the same.
        Assert.Equal(FinishReason.Stop, terminalReason);
    }

    /// <summary>
    /// Terminal <c>StoreInPrefixCache</c> happens for both paths. <see cref="PrefixCache"/> is
    /// sealed and <c>Store</c> is internal, so a precise call-count probe isn't possible from
    /// unit tests — instead this test asserts the observable outcome: after a single Generate
    /// call the prefix cache holds exactly one entry whose token sequence matches the full
    /// prompt + generated stream. Pre-consolidation the streaming path had ~7 candidate store
    /// sites all writing the SAME kvCache (the prefix-cache dedupes by reference, so call-count
    /// is invisible to this kind of assertion). The byte-identical parity assertion in
    /// <see cref="GenerateAndStreaming_ProduceByteIdenticalOutputs"/> guards the orchestration;
    /// this test guards that the terminal store fires at all on both paths.
    /// </summary>
    [Fact]
    public void TerminalStoreInPrefixCache_HappensOnGenerate()
    {
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = string.Empty,
            [2] = "Hello",
            [3] = ", ",
            [4] = "world",
            [5] = "!",
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);
        var argmaxSequence = new[] { 2, 3, 4, 5, 1 };

        var prefixCache = new PrefixCache(maxEntries: 4);
        var model = new ScriptedSequenceModel(argmaxSequence);
        var generator = new TextGenerator(model, tokenizer, prefixCache: prefixCache);

        var options = new InferenceOptions { MaxTokens = 16, Temperature = 0f };
        _ = generator.Generate("hi", options);

        Assert.Equal(1, prefixCache.EntryCount);
    }

    /// <summary>
    /// Streaming variant of <see cref="TerminalStoreInPrefixCache_HappensOnGenerate"/>.
    /// Pre-consolidation streaming hit one of ~7 terminal yield sites — and the dedupe in
    /// PrefixCache.Store hid any regression in call count. The observable assertion is that
    /// the terminal store fires at least once and produces exactly one entry.
    /// </summary>
    [Fact]
    public async Task TerminalStoreInPrefixCache_HappensOnStreaming()
    {
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = string.Empty,
            [2] = "Hi",
            [3] = "!",
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);
        var argmaxSequence = new[] { 2, 3, 1 };

        var prefixCache = new PrefixCache(maxEntries: 4);
        var model = new ScriptedSequenceModel(argmaxSequence);
        var generator = new TextGenerator(model, tokenizer, prefixCache: prefixCache);

        var options = new InferenceOptions { MaxTokens = 16, Temperature = 0f };
        await foreach (var _ in generator.GenerateStreamingTokensAsync("hi", options)) { }

        Assert.Equal(1, prefixCache.EntryCount);
    }

    /// <summary>
    /// <c>onTokenGenerated</c> must NOT fire for a Stop-discarded token. Today's behaviour:
    /// callback fires for natural-end / max-tokens (Length) terminations but is skipped when
    /// the loop exits via <see cref="StopResult.Stop"/>. The wrapper preserves this contract.
    /// </summary>
    [Fact]
    public void OnTokenGenerated_NotInvokedForStopDiscardedToken()
    {
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = "<eos-should-not-show>",
            [2] = "Hi",
            [3] = "!",
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);
        var argmaxSequence = new[] { 2, 3, 1 };

        var observed = new List<int>();
        var model = new ScriptedSequenceModel(argmaxSequence);
        var generator = new TextGenerator(model, tokenizer);

        var options = new InferenceOptions { MaxTokens = 16, Temperature = 0f };
        _ = generator.Generate("hi", options, onTokenGenerated: id => observed.Add(id));

        Assert.Equal(new[] { 2, 3 }, observed.ToArray()); // EOS id 1 must not appear.
    }

    /// <summary>
    /// MaxTokens=1 edge case: the first token is also the last. Both paths must handle this
    /// — terminal yield (streaming) and immediate response build (Generate) — without losing
    /// the token or double-counting.
    /// </summary>
    [Fact]
    public async Task MaxTokensOne_FirstTokenIsAlsoLast_BothPathsHandleCorrectly()
    {
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = string.Empty,
            [2] = "OnlyOne",
            [3] = "NEVER",
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);
        var argmaxSequence = new[] { 2, 3 };
        var options = new InferenceOptions { MaxTokens = 1, Temperature = 0f };

        var modelA = new ScriptedSequenceModel(argmaxSequence);
        var response = new TextGenerator(modelA, tokenizer).Generate("hi", options);
        Assert.Equal(new[] { 2 }, response.GeneratedTokenIds);
        Assert.Equal("OnlyOne", response.Text);
        Assert.Equal(FinishReason.Length, response.FinishReason);
        Assert.Equal(1, response.GeneratedTokenCount);

        var modelB = new ScriptedSequenceModel(argmaxSequence);
        var ids = new List<int>();
        var text = new System.Text.StringBuilder();
        FinishReason? streamedReason = null;
        await foreach (var tok in new TextGenerator(modelB, tokenizer).GenerateStreamingTokensAsync("hi", options))
        {
            if (tok.FinishReason != FinishReason.Stop)
                ids.Add(tok.TokenId);
            text.Append(tok.Text);
            if (tok.FinishReason.HasValue) streamedReason = tok.FinishReason.Value;
        }
        Assert.Equal(new[] { 2 }, ids.ToArray());
        Assert.Equal("OnlyOne", text.ToString());
        Assert.Equal(FinishReason.Length, streamedReason);
    }

    /// <summary>
    /// MaxTokens=0 edge case: the wrapper short-circuits BEFORE invoking the core, returning
    /// a response with the correct <c>PromptTokenCount</c>. Asserts the early-return contract
    /// is preserved across the consolidation.
    /// </summary>
    [Fact]
    public void MaxTokensZero_GenerateShortCircuitsWithPromptTokenCount()
    {
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = string.Empty,
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);
        var model = new ScriptedSequenceModel(new[] { 2, 3 });
        var generator = new TextGenerator(model, tokenizer);

        var response = generator.Generate("hi", new InferenceOptions { MaxTokens = 0 });

        Assert.Equal(0, response.GeneratedTokenCount);
        Assert.Empty(response.GeneratedTokenIds);
        Assert.Equal(string.Empty, response.Text);
        Assert.Equal(FinishReason.Length, response.FinishReason);
        // ScriptedTokenizer.Encode returns the BOS-seed prompt (length 1) for any input.
        Assert.Equal(1, response.PromptTokenCount);
    }

    // ── Helpers — duplicated from TextGeneratorStopStringStreamingTests so this file stands alone ──

    private sealed class ScriptedTokenizer : ITokenizer
    {
        private readonly Dictionary<int, string> _idToText;

        public ScriptedTokenizer(Dictionary<int, string> idToText, int bosId, int eosId)
        {
            _idToText = idToText;
            BosTokenId = bosId;
            EosTokenId = eosId;
        }

        public int VocabSize => 16;
        public int BosTokenId { get; }
        public int EosTokenId { get; }

        public int[] Encode(string text) => new[] { BosTokenId };

        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new System.Text.StringBuilder();
            foreach (var id in tokenIds)
                if (_idToText.TryGetValue(id, out var s))
                    sb.Append(s);
            return sb.ToString();
        }

        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);

        public string DecodeToken(int tokenId) => _idToText.TryGetValue(tokenId, out var s) ? s : string.Empty;

        public int CountTokens(string text) => 1;
    }

    private sealed class ScriptedSequenceModel : IModel
    {
        private readonly int[] _sequence;
        private int _callIndex;

        public ScriptedSequenceModel(int[] sequence) => _sequence = sequence;

        public ModelConfig Config => new()
        {
            VocabSize = VocabSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumKvHeads,
            NumKvHeads = NumKvHeads,
            HiddenSize = HeadDim * NumKvHeads,
            IntermediateSize = HeadDim * 4,
            HeadDim = HeadDim,
            MaxSequenceLength = 64,
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Forward(tokenIds, positions, deviceId, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
            int deviceId, IKvCache? kvCache)
        {
            int batchSize = tokenIds.Length;
            long totalFloats = (long)batchSize * VocabSize;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(totalFloats * sizeof(float)), 64);

            float* dst = (float*)ptr;
            for (int b = 0; b < batchSize; b++)
            {
                int nextId = _callIndex < _sequence.Length ? _sequence[_callIndex] : 1;
                float* row = dst + (long)b * VocabSize;
                for (int v = 0; v < VocabSize; v++)
                    row[v] = -1000f;
                row[nextId] = 1000f;
            }
            _callIndex++;

            if (kvCache != null)
            {
                int kvStride = NumKvHeads * HeadDim;
                int rows = batchSize;
                nint kPtr = (nint)NativeMemory.AlignedAlloc((nuint)(rows * kvStride * sizeof(float)), 64);
                nint vPtr = (nint)NativeMemory.AlignedAlloc((nuint)(rows * kvStride * sizeof(float)), 64);
                NativeMemory.Clear((void*)kPtr, (nuint)(rows * kvStride * sizeof(float)));
                NativeMemory.Clear((void*)vPtr, (nuint)(rows * kvStride * sizeof(float)));
                for (int layer = 0; layer < NumLayers; layer++)
                {
                    var kRef = new TensorRef(rows, kvStride, DType.Float32, -1, kPtr);
                    var vRef = new TensorRef(rows, kvStride, DType.Float32, -1, vPtr);
                    kvCache.Update(kRef, vRef, positions, layer);
                }
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }

            return new UnmanagedTensor(new TensorShape(batchSize, VocabSize), DType.Float32, deviceId, ptr);
        }

        public void Dispose() { }
    }
}
