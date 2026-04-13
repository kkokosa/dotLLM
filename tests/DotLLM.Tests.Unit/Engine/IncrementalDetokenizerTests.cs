using DotLLM.Engine;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Tests for <see cref="IncrementalDetokenizer"/>: verifies incremental decoding produces
/// the same full text as bulk <c>ITokenizer.Decode(all)</c>, across BPE/SentencePiece vocabs
/// with ASCII, space markers, and byte-fallback tokens (which accumulate across tokens).
/// </summary>
public sealed class IncrementalDetokenizerTests
{
    private static BpeTokenizer BuildSpaceMarkerVocab()
    {
        string[] tokens =
        [
            "<unk>",        // 0
            "\u2581",       // 1  ▁
            "h",            // 2
            "e",            // 3
            "l",            // 4
            "o",            // 5
            "\u2581h",      // 6  ▁h
            "\u2581he",     // 7  ▁he
            "\u2581hel",    // 8  ▁hel
            "\u2581hell",   // 9  ▁hell
            "\u2581hello",  // 10 ▁hello
            "\u2581world",  // 11 ▁world
            "\u2581w",      // 12 ▁w
            "w",            // 13
            "r",            // 14
            "d",            // 15
        ];
        float[] scores = new float[tokens.Length];
        return BpeTokenizer.CreateSentencePiece(tokens, scores, tokenTypes: null,
            bosId: 0, eosId: 0, addBosSpace: false);
    }

    private static BpeTokenizer BuildByteFallbackVocab()
    {
        // <unk>, 'a', 256 byte tokens <0x00>..<0xFF>
        var tokens = new List<string> { "<unk>", "a" };
        for (int i = 0; i < 256; i++)
            tokens.Add($"<0x{i:X2}>");
        float[] scores = new float[tokens.Count];
        return BpeTokenizer.CreateSentencePiece(tokens.ToArray(), scores, tokenTypes: null,
            bosId: 0, eosId: 0, addBosSpace: false);
    }

    [Fact]
    public void Append_EmptySequence_LengthZeroAndEmptyString()
    {
        var tok = BuildSpaceMarkerVocab();
        var detok = new IncrementalDetokenizer(tok);
        Assert.Equal(0, detok.Length);
        Assert.Equal(string.Empty, detok.ToString());
        Assert.Equal(string.Empty, detok.TakeDelta());
    }

    [Fact]
    public void Append_MatchesBulkDecode_ShortSequence()
    {
        var tok = BuildSpaceMarkerVocab();
        int[] ids = [10, 1, 13, 5, 14, 4, 15]; // "▁hello" + " " + "world" (space-markered)
        var detok = new IncrementalDetokenizer(tok);
        foreach (int id in ids) detok.Append(id);
        Assert.Equal(tok.Decode(ids, stripBosSpace: false), detok.ToString());
    }

    [Fact]
    public void Append_MatchesBulkDecode_LongSequenceBeyondSoftWindow()
    {
        var tok = BuildSpaceMarkerVocab();
        // 40 tokens, alternating h/e/l/l/o with separators — forces eviction many times.
        var ids = new List<int>();
        for (int i = 0; i < 40; i++) ids.Add(2 + (i % 4));
        var detok = new IncrementalDetokenizer(tok);
        foreach (int id in ids) detok.Append(id);
        Assert.Equal(tok.Decode(ids.ToArray(), stripBosSpace: false), detok.ToString());
    }

    [Fact]
    public void TakeDelta_ConcatenationEqualsFullText()
    {
        var tok = BuildSpaceMarkerVocab();
        int[] ids = [10, 1, 13, 5, 14, 4, 15, 2, 3, 4, 4, 5];
        var detok = new IncrementalDetokenizer(tok);
        var accumulated = new System.Text.StringBuilder();
        foreach (int id in ids)
        {
            detok.Append(id);
            accumulated.Append(detok.TakeDelta());
        }
        Assert.Equal(tok.Decode(ids, stripBosSpace: false), accumulated.ToString());
        Assert.Equal(string.Empty, detok.TakeDelta()); // no pending delta after draining
    }

    [Fact]
    public void TakeDelta_LeavesFullToStringConsistent()
    {
        var tok = BuildSpaceMarkerVocab();
        int[] ids = [10, 1, 11];
        var detok = new IncrementalDetokenizer(tok);
        foreach (int id in ids) detok.Append(id);
        string full = detok.ToString();
        string delta = detok.TakeDelta();
        // First drain returns everything; ToString is unchanged; a second drain is empty.
        Assert.Equal(full, delta);
        Assert.Equal(full, detok.ToString());
        Assert.Equal(string.Empty, detok.TakeDelta());
    }

    [Fact]
    public void GetTailView_ReturnsLastChars_FromWindowOnly()
    {
        var tok = BuildSpaceMarkerVocab();
        int[] ids = [10]; // "▁hello" → " hello" (with addBosSpace=false, leading space preserved? check)
        var detok = new IncrementalDetokenizer(tok);
        foreach (int id in ids) detok.Append(id);
        string full = detok.ToString();

        Span<char> scratch = stackalloc char[32];
        var tail = detok.GetTailView(3, scratch);
        Assert.Equal(full.AsSpan(full.Length - 3).ToString(), tail.ToString());
    }

    [Fact]
    public void GetTailView_SpansCommittedAndWindowBoundary()
    {
        var tok = BuildSpaceMarkerVocab();
        // Generate enough tokens to force commits into the committed buffer.
        int[] ids = [2, 3, 4, 4, 5, 2, 3, 4, 4, 5, 2, 3, 4, 4, 5]; // "hellohellohello"
        var detok = new IncrementalDetokenizer(tok);
        foreach (int id in ids) detok.Append(id);
        string full = detok.ToString();

        Span<char> scratch = stackalloc char[64];
        // Request a tail larger than the current window, forcing the committed+window stitch path.
        var tail = detok.GetTailView(full.Length, scratch);
        Assert.Equal(full, tail.ToString());
    }

    [Fact]
    public void ByteFallbackTokens_AccumulateAcrossWindow()
    {
        // "é" is UTF-8 0xC3 0xA9 — two byte-fallback tokens that must decode together.
        var tok = BuildByteFallbackVocab();
        const int AId = 1;               // 'a'
        int byteC3 = 2 + 0xC3;           // <0xC3>
        int byteA9 = 2 + 0xA9;           // <0xA9>
        int[] ids = [AId, byteC3, byteA9, AId]; // "aéa"

        string bulk = tok.Decode(ids, stripBosSpace: false);

        var detok = new IncrementalDetokenizer(tok);
        foreach (int id in ids) detok.Append(id);

        Assert.Equal(bulk, detok.ToString());
    }

    [Fact]
    public void ByteFallbackTokens_LongRun_MatchesBulkDecode()
    {
        var tok = BuildByteFallbackVocab();
        int byteC3 = 2 + 0xC3;
        int byteA9 = 2 + 0xA9;
        const int AId = 1;

        // 100 repetitions of "aé" — exercises eviction with interleaved byte-token runs.
        var ids = new List<int>();
        for (int i = 0; i < 100; i++)
        {
            ids.Add(AId);
            ids.Add(byteC3);
            ids.Add(byteA9);
        }

        string bulk = tok.Decode(ids.ToArray(), stripBosSpace: false);

        var detok = new IncrementalDetokenizer(tok);
        foreach (int id in ids) detok.Append(id);

        Assert.Equal(bulk, detok.ToString());
    }
}
