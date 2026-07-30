using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// End-to-end regression test for the streaming stop-string suffix trim
/// (upstream issue #121 item 8). Wires a scripted model + scripted tokenizer
/// through <see cref="TextGenerator.GenerateStreamingTokensAsync"/> and
/// asserts that no fragment of the matched stop string ever leaks out as a
/// streamed delta — proving the fix end-to-end rather than only at the
/// <see cref="StreamingStopBuffer"/> unit level.
/// </summary>
public sealed class TextGeneratorStopStringStreamingTests
{
    private const int VocabSize = 16;
    private const int NumLayers = 1;
    private const int NumKvHeads = 1;
    private const int HeadDim = 4;

    /// <summary>
    /// Discrimination test for the headline bug. With a scripted tokenizer where the
    /// stop string <c>"&lt;|im_end|&gt;"</c> is decoded across three consecutive
    /// generated tokens (<c>"&lt;|"</c>, <c>"im_"</c>, <c>"end|&gt;"</c>), the streaming
    /// emit path used to send <c>"&lt;|"</c> and <c>"im_"</c> through as
    /// <c>delta.content</c> before the third token completed the match. The fixed
    /// path must buffer them and emit nothing of the stop string.
    /// </summary>
    [Fact]
    public async Task GenerateStreamingTokensAsync_StopStringStraddlesThreeTokens_NoFragmentLeaks()
    {
        // Scripted vocabulary:
        //   id 0 = BOS, id 1 = EOS
        //   id 2 = "Hello, world", id 3 = "<|", id 4 = "im_", id 5 = "end|>"
        var idToText = new Dictionary<int, string>
        {
            [0] = string.Empty,
            [1] = string.Empty,
            [2] = "Hello, world",
            [3] = "<|",
            [4] = "im_",
            [5] = "end|>",
        };
        var tokenizer = new ScriptedTokenizer(idToText, bosId: 0, eosId: 1);

        // Scripted decode sequence (deterministic argmax walk): 2, 3, 4, 5.
        var argmaxSequence = new[] { 2, 3, 4, 5 };
        var model = new ScriptedSequenceModel(argmaxSequence);

        var options = new InferenceOptions
        {
            MaxTokens = 16,
            Temperature = 0f,
            StopSequences = new List<string> { "<|im_end|>" },
        };

        var generator = new TextGenerator(model, tokenizer);

        var emittedChunks = new List<string>();
        FinishReason? finalReason = null;
        await foreach (var tok in generator.GenerateStreamingTokensAsync("hi", options))
        {
            if (!string.IsNullOrEmpty(tok.Text))
                emittedChunks.Add(tok.Text);
            if (tok.FinishReason.HasValue)
                finalReason = tok.FinishReason.Value;
        }

        string emittedTotal = string.Concat(emittedChunks);

        // The visible stream must reconstruct exactly the pre-stop content.
        Assert.Equal("Hello, world", emittedTotal);
        // The decisive assertion — no chunk may contain any fragment of "<|im_end|>".
        foreach (var chunk in emittedChunks)
        {
            Assert.DoesNotContain("<|", chunk);
            Assert.DoesNotContain("im_", chunk);
            Assert.DoesNotContain("end|>", chunk);
        }
        Assert.Equal(FinishReason.Stop, finalReason);
    }

    /// <summary>
    /// Sanity test: when the registered stop string never matches and generation ends
    /// on EOS / max-tokens, the full decoded text must be streamed — the holdback must
    /// not silently truncate the tail (advisor: tail-not-lost-on-natural-end trap).
    /// </summary>
    [Fact]
    public async Task GenerateStreamingTokensAsync_StopStringNeverMatches_FullTextStreamed()
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

        // Hits EOS after 4 content tokens.
        var argmaxSequence = new[] { 2, 3, 4, 5, 1 };
        var model = new ScriptedSequenceModel(argmaxSequence);

        var options = new InferenceOptions
        {
            MaxTokens = 16,
            Temperature = 0f,
            StopSequences = new List<string> { "<|im_end|>" }, // never matches
        };

        var generator = new TextGenerator(model, tokenizer);

        var emittedChunks = new List<string>();
        await foreach (var tok in generator.GenerateStreamingTokensAsync("hi", options))
        {
            if (!string.IsNullOrEmpty(tok.Text))
                emittedChunks.Add(tok.Text);
        }

        Assert.Equal("Hello, world!", string.Concat(emittedChunks));
    }

    // ── Helpers ──

    /// <summary>
    /// A scripted tokenizer with a fixed id→text map. <c>Decode(span)</c> concatenates the
    /// per-id strings. <c>Encode</c> always returns a single-id seed prompt — the test only
    /// cares about generation behavior, not prompt tokenization.
    /// </summary>
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

    /// <summary>
    /// Model that walks a fixed argmax sequence at temperature=0: position 0 → seq[0],
    /// position 1 → seq[1], etc. Logits give the selected id a huge positive value and
    /// all others a very negative value, so any temperature-0 sampler picks the scripted id.
    /// </summary>
    private sealed class ScriptedSequenceModel : IModel
    {
        private readonly int[] _sequence;
        private int _callIndex; // position in the scripted sequence

        public ScriptedSequenceModel(int[] sequence)
        {
            _sequence = sequence;
        }

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
                int nextId = _callIndex < _sequence.Length ? _sequence[_callIndex] : 1; // EOS fallback
                float* row = dst + (long)b * VocabSize;
                for (int v = 0; v < VocabSize; v++)
                    row[v] = -1000f;
                row[nextId] = 1000f;
            }
            // Only the last position's logits are sampled by the engine — advance by one.
            _callIndex++;

            // Touch the KV-cache so its position counter advances (the engine guards on
            // pos < cacheSize). A zero-filled write at the supplied positions is sufficient.
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
