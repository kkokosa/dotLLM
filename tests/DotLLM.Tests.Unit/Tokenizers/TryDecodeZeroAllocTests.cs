using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers;

/// <summary>
/// Tests for the zero-allocation <see cref="ITokenizer.TryDecode"/> overload.
/// Verifies bit-exact parity with the allocating <see cref="ITokenizer.Decode(System.ReadOnlySpan{int}, bool)"/>
/// path across the SentencePiece and tiktoken (GPT-2) BPE encodings, and confirms that the
/// hot decode path produces zero managed allocation per call when the destination buffer is
/// pre-sized adequately.
/// </summary>
public sealed class TryDecodeZeroAllocTests
{
    // -------------------------------------------------------------------------
    // Vocabulary factories
    // -------------------------------------------------------------------------

    private static BpeTokenizer BuildSpaceMarkerVocab(bool addBosSpace)
    {
        string[] tokens =
        [
            "<unk>",        // 0
            "▁",       // 1  ▁
            "h",            // 2
            "e",            // 3
            "l",            // 4
            "o",            // 5
            "▁h",      // 6  ▁h
            "▁he",     // 7  ▁he
            "▁hel",    // 8  ▁hel
            "▁hell",   // 9  ▁hell
            "▁hello",  // 10 ▁hello
            "▁world",  // 11 ▁world
            "▁w",      // 12 ▁w
            "w",            // 13
            "r",            // 14
            "d",            // 15
        ];
        float[] scores = new float[tokens.Length];
        return BpeTokenizer.CreateSentencePiece(tokens, scores, tokenTypes: null,
            bosId: 0, eosId: 0, addBosSpace: addBosSpace);
    }

    private static BpeTokenizer BuildByteFallbackVocab()
    {
        var tokens = new List<string> { "<unk>", "a" };
        for (int i = 0; i < 256; i++)
            tokens.Add($"<0x{i:X2}>");
        float[] scores = new float[tokens.Count];
        return BpeTokenizer.CreateSentencePiece(tokens.ToArray(), scores, tokenTypes: null,
            bosId: 0, eosId: 0, addBosSpace: false);
    }

    private static BpeTokenizer BuildTiktokenVocab()
    {
        // Minimal GPT-2 / tiktoken vocab: ASCII chars + a couple of merges.
        // Each char in a token string is interpreted as one byte via the GPT-2 byte-to-unicode mapping.
        string[] tokens =
        [
            "<unk>",   // 0
            "h",       // 1
            "e",       // 2
            "l",       // 3
            "o",       // 4
            " ",       // 5  (raw space is one byte)
            "w",       // 6
            "r",       // 7
            "d",       // 8
            "he",      // 9
            "lo",      // 10
            "hello",   // 11
            " w",      // 12
            "wo",      // 13
            "rl",      // 14
            "world",   // 15
        ];
        string[] merges =
        [
            "h e",
            "l o",
            "he llo",
            "w o",
            "r l",
            "wo rld",
        ];
        return BpeTokenizer.CreateTiktoken(tokens, merges, tokenTypes: null,
            bosId: 0, eosId: 0, preTokenizerType: null);
    }

    // -------------------------------------------------------------------------
    // Parity tests — TryDecode output must equal Decode output
    // -------------------------------------------------------------------------

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Parity_SentencePiece_ShortSequence(bool stripBosSpace)
    {
        BpeTokenizer tok = BuildSpaceMarkerVocab(addBosSpace: true);
        int[] ids = [10, 1, 11]; // "▁hello" + "▁" + "▁world"

        string expected = tok.Decode(ids, stripBosSpace);

        Span<char> buffer = stackalloc char[64];
        bool ok = tok.TryDecode(ids, stripBosSpace, buffer, out int written);

        Assert.True(ok);
        Assert.Equal(expected, buffer[..written].ToString());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Parity_SentencePiece_NoBosSpace(bool stripBosSpace)
    {
        // addBosSpace=false means the encoding never prepends ▁ — stripBosSpace should be a no-op.
        BpeTokenizer tok = BuildSpaceMarkerVocab(addBosSpace: false);
        int[] ids = [2, 3, 4, 4, 5, 1, 13, 5, 14, 4, 15]; // "hello" + "▁" + "world"

        string expected = tok.Decode(ids, stripBosSpace);

        Span<char> buffer = stackalloc char[64];
        bool ok = tok.TryDecode(ids, stripBosSpace, buffer, out int written);

        Assert.True(ok);
        Assert.Equal(expected, buffer[..written].ToString());
    }

    [Fact]
    public void Parity_SentencePiece_ByteFallbackRun()
    {
        BpeTokenizer tok = BuildByteFallbackVocab();
        const int aId = 1;
        int byteC3 = 2 + 0xC3;
        int byteA9 = 2 + 0xA9;
        int[] ids = [aId, byteC3, byteA9, aId]; // "aéa"

        string expected = tok.Decode(ids, stripBosSpace: false);

        Span<char> buffer = stackalloc char[32];
        bool ok = tok.TryDecode(ids, stripBosSpace: false, buffer, out int written);

        Assert.True(ok);
        Assert.Equal(expected, buffer[..written].ToString());
    }

    [Fact]
    public void Parity_SentencePiece_RandomSequences()
    {
        BpeTokenizer tok = BuildSpaceMarkerVocab(addBosSpace: true);
        var rng = new Random(42);
        Span<char> buffer = stackalloc char[256];

        for (int trial = 0; trial < 50; trial++)
        {
            int len = rng.Next(1, 24);
            int[] ids = new int[len];
            for (int i = 0; i < len; i++) ids[i] = rng.Next(1, 16);

            foreach (bool stripBosSpace in new[] { true, false })
            {
                string expected = tok.Decode(ids, stripBosSpace);
                bool ok = tok.TryDecode(ids, stripBosSpace, buffer, out int written);
                Assert.True(ok, $"TryDecode unexpectedly returned false on trial {trial} (stripBosSpace={stripBosSpace})");
                Assert.Equal(expected, buffer[..written].ToString());
            }
        }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Parity_Tiktoken_ShortSequence(bool stripBosSpace)
    {
        BpeTokenizer tok = BuildTiktokenVocab();
        int[] ids = [11, 5, 15]; // "hello" + " " + "world"

        string expected = tok.Decode(ids, stripBosSpace);

        Span<char> buffer = stackalloc char[64];
        bool ok = tok.TryDecode(ids, stripBosSpace, buffer, out int written);

        Assert.True(ok);
        Assert.Equal(expected, buffer[..written].ToString());
    }

    [Fact]
    public void Parity_Tiktoken_RandomSequences()
    {
        BpeTokenizer tok = BuildTiktokenVocab();
        var rng = new Random(123);
        Span<char> buffer = stackalloc char[256];

        for (int trial = 0; trial < 50; trial++)
        {
            int len = rng.Next(1, 20);
            int[] ids = new int[len];
            for (int i = 0; i < len; i++) ids[i] = rng.Next(1, 16);

            foreach (bool stripBosSpace in new[] { true, false })
            {
                string expected = tok.Decode(ids, stripBosSpace);
                bool ok = tok.TryDecode(ids, stripBosSpace, buffer, out int written);
                Assert.True(ok, $"TryDecode unexpectedly returned false on trial {trial}");
                Assert.Equal(expected, buffer[..written].ToString());
            }
        }
    }

    // -------------------------------------------------------------------------
    // Buffer-too-small contract
    // -------------------------------------------------------------------------

    [Fact]
    public void BufferTooSmall_SentencePiece_ReturnsFalse_WrittenIsZero()
    {
        BpeTokenizer tok = BuildSpaceMarkerVocab(addBosSpace: false);
        int[] ids = [10, 1, 11]; // expands to >2 chars

        Span<char> tiny = stackalloc char[2];
        bool ok = tok.TryDecode(ids, stripBosSpace: false, tiny, out int written);

        Assert.False(ok);
        Assert.Equal(0, written);
    }

    [Fact]
    public void BufferTooSmall_Tiktoken_ReturnsFalse_WrittenIsZero_AndAtomic()
    {
        BpeTokenizer tok = BuildTiktokenVocab();
        int[] ids = [11, 5, 15]; // "hello world"

        Span<char> tiny = stackalloc char[3];
        // Pre-fill the buffer with a sentinel to verify atomicity (tiktoken pre-computes char count).
        tiny.Fill('X');
        bool ok = tok.TryDecode(ids, stripBosSpace: false, tiny, out int written);

        Assert.False(ok);
        Assert.Equal(0, written);
        Assert.Equal("XXX", tiny.ToString());
    }

    [Fact]
    public void EmptyInput_ReturnsTrue_WrittenIsZero()
    {
        BpeTokenizer tok = BuildSpaceMarkerVocab(addBosSpace: true);

        Span<char> buffer = stackalloc char[8];
        bool ok = tok.TryDecode([], stripBosSpace: true, buffer, out int written);

        Assert.True(ok);
        Assert.Equal(0, written);
    }

    // -------------------------------------------------------------------------
    // Zero-allocation assertions (steady-state)
    // -------------------------------------------------------------------------

    [Fact]
    public void TryDecode_SentencePiece_ZeroAllocationPerCall()
    {
        BpeTokenizer tok = BuildSpaceMarkerVocab(addBosSpace: false);
        int[] ids = [10, 1, 11];
        Span<char> buffer = stackalloc char[128];

        // Warm-up: pay JIT + first-call ArrayPool rent costs outside the measurement window.
        for (int i = 0; i < 32; i++)
        {
            bool ok = tok.TryDecode(ids, stripBosSpace: false, buffer, out _);
            Assert.True(ok);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
        {
            bool ok = tok.TryDecode(ids, stripBosSpace: false, buffer, out _);
            Assert.True(ok);
        }
        long after = GC.GetAllocatedBytesForCurrentThread();

        long delta = after - before;
        // Expect bit-exact zero managed allocation on the steady-state hot path.
        Assert.True(delta == 0,
            $"Expected 0 managed bytes allocated across 1000 TryDecode calls, got {delta} bytes");
    }

    [Fact]
    public void TryDecode_Tiktoken_ZeroAllocationPerCall()
    {
        BpeTokenizer tok = BuildTiktokenVocab();
        int[] ids = [11, 5, 15];
        Span<char> buffer = stackalloc char[128];

        for (int i = 0; i < 32; i++)
        {
            bool ok = tok.TryDecode(ids, stripBosSpace: false, buffer, out _);
            Assert.True(ok);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
        {
            bool ok = tok.TryDecode(ids, stripBosSpace: false, buffer, out _);
            Assert.True(ok);
        }
        long after = GC.GetAllocatedBytesForCurrentThread();

        long delta = after - before;
        Assert.True(delta == 0,
            $"Expected 0 managed bytes allocated across 1000 TryDecode calls, got {delta} bytes");
    }

    [Fact]
    public void Decode_SentencePiece_AllocatesAtLeastOneStringPerCall_Baseline()
    {
        // Sanity check: the existing allocating Decode path should allocate per call.
        // This is the baseline the TryDecode improvement is measured against — without it,
        // a passing zero-alloc test for TryDecode would be meaningless.
        BpeTokenizer tok = BuildSpaceMarkerVocab(addBosSpace: false);
        int[] ids = [10, 1, 11];

        for (int i = 0; i < 32; i++) _ = tok.Decode(ids, stripBosSpace: false);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++) _ = tok.Decode(ids, stripBosSpace: false);
        long after = GC.GetAllocatedBytesForCurrentThread();

        long delta = after - before;
        Assert.True(delta > 0,
            $"Sanity check failed: allocating Decode reported {delta} bytes — expected > 0");
    }
}
