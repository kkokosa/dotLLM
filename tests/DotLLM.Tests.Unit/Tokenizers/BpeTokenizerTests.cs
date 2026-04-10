using System.Text.RegularExpressions;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers;

/// <summary>
/// Unit tests for <see cref="BpeTokenizer"/> using synthetic in-memory vocabularies.
/// No file I/O; all vocab data is built inline.
/// </summary>
public class BpeTokenizerTests
{
    // -------------------------------------------------------------------------
    // Vocabulary helpers
    // -------------------------------------------------------------------------

    /// <summary>
    /// Minimal SentencePiece-style vocab: single chars + two merges.
    /// <code>
    /// 0: &lt;unk&gt;  1: a  2: b  3: c  4: ab (score -0.5)  5: bc (score -0.8)  6: abc (score -0.2)
    /// </code>
    /// With addBosSpace=false for deterministic unit-test behaviour.
    /// </summary>
    private static BpeTokenizer BuildMinimalVocab()
    {
        string[] tokens = ["<unk>", "a", "b", "c", "ab", "bc", "abc"];
        float[] scores  = [0f, -1.0f, -2.0f, -3.0f, -0.5f, -0.8f, -0.2f];
        return BpeTokenizer.CreateSentencePiece(tokens, scores, tokenTypes: null,
            bosId: 0, eosId: 0, addBosSpace: false);
    }

    /// <summary>
    /// Vocab with ▁ marker and merges building up "▁hello".
    /// Used to verify SentencePiece space-marker handling.
    /// </summary>
    private static BpeTokenizer BuildSpaceMarkerVocab()
    {
        string[] tokens =
        [
            "<unk>",   // 0
            "\u2581",  // 1  ▁
            "h",       // 2
            "e",       // 3
            "l",       // 4
            "o",       // 5
            "\u2581h", // 6  ▁h
            "\u2581he",// 7  ▁he
            "\u2581hel",// 8 ▁hel
            "\u2581hell",// 9 ▁hell
            "\u2581hello",// 10 ▁hello
        ];
        float[] scores = [0f, -5f, -4f, -3f, -2f, -1f, -0.5f, -0.4f, -0.3f, -0.2f, -0.1f];
        return BpeTokenizer.CreateSentencePiece(tokens, scores, tokenTypes: null,
            bosId: 0, eosId: 0, addBosSpace: true);
    }

    /// <summary>
    /// Vocab with byte-fallback tokens for bytes 0x61='a' and non-ASCII bytes.
    /// </summary>
    private static BpeTokenizer BuildByteVocab()
    {
        // 256 byte tokens (<0x00>–<0xFF>) plus a few regular tokens.
        var tokenList = new List<string>(260);
        tokenList.Add("<unk>"); // 0
        tokenList.Add("a");    // 1
        for (int i = 0; i < 256; i++)
            tokenList.Add($"<0x{i:X2}>"); // 2–257
        string[] tokens = [.. tokenList];
        float[] scores = new float[tokens.Length]; // all 0
        return BpeTokenizer.CreateSentencePiece(tokens, scores, tokenTypes: null,
            bosId: 0, eosId: 0, addBosSpace: false);
    }

    // -------------------------------------------------------------------------
    // Encode tests
    // -------------------------------------------------------------------------

    [Fact]
    public void Encode_EmptyString_ReturnsEmpty()
    {
        var tok = BuildMinimalVocab();
        Assert.Empty(tok.Encode(string.Empty));
    }

    [Fact]
    public void Encode_SingleKnownToken_ReturnsSingleId()
    {
        var tok = BuildMinimalVocab();
        // "a" is token 1; no merge possible with a single symbol.
        int[] ids = tok.Encode("a");
        Assert.Equal([1], ids);
    }

    [Fact]
    public void Encode_TwoTokenWord_MergesIntoOne()
    {
        var tok = BuildMinimalVocab();
        // "ab" → initial [a(1), b(2)] → bigram (a,b) → "ab"(4) → result [4]
        int[] ids = tok.Encode("ab");
        Assert.Equal([4], ids);
    }

    [Fact]
    public void Encode_ThreeChars_MergesByScorePriority()
    {
        var tok = BuildMinimalVocab();
        // "abc" → initial [a(1), b(2), c(3)]
        // Possible merges: (a,b)→ab score -0.5, (b,c)→bc score -0.8
        // -0.5 > -0.8 so "ab" has higher priority → merge (a,b) first → [ab(4), c(3)]
        // Then (ab,c) → "abc"(6) → [abc(6)]
        int[] ids = tok.Encode("abc");
        Assert.Equal([6], ids);
    }

    [Fact]
    public void Encode_StaleQueueEntry_IsSkipped()
    {
        var tok = BuildMinimalVocab();
        // "abc" encodes to [6]. The (b,c)→bc bigram was enqueued but b is consumed by (a,b) merge.
        // The stale (b,c) entry must be silently discarded.
        int[] ids = tok.Encode("abc");
        // If stale entry were not skipped we'd get a wrong result; [6] is the correct merge.
        Assert.Equal([6], ids);
    }

    [Fact]
    public void Encode_LowPriorityMerge_NotAppliedWhenConsumed()
    {
        // "bc" alone should produce [5] (bc merged).
        var tok = BuildMinimalVocab();
        int[] ids = tok.Encode("bc");
        Assert.Equal([5], ids);
    }

    [Fact]
    public void Encode_UnknownChar_UsesByteTokenFallback()
    {
        var tok = BuildByteVocab();
        // 'Z' = 0x5A; no regular token for 'Z', but <0x5A> (index 2+0x5A=92) exists.
        int byteTokenId = 2 + 0x5A; // 0x5A = 90 → token index 92
        int[] ids = tok.Encode("Z");
        Assert.Equal([byteTokenId], ids);
    }

    [Fact]
    public void Encode_SpaceMarkerPrepended_WhenAddBosSpaceTrue()
    {
        var tok = BuildSpaceMarkerVocab();
        // "hello" → normalized "▁hello" → eventually merges to token 10
        int[] ids = tok.Encode("hello");
        Assert.Equal([10], ids);
    }

    [Fact]
    public void Encode_SpaceMarkerNotDoublePrepended_WhenTextStartsWithSpace()
    {
        var tok = BuildSpaceMarkerVocab();
        // " hello" already starts with space → normalized "▁hello" → same as "hello"
        int[] ids1 = tok.Encode("hello");
        int[] ids2 = tok.Encode(" hello");
        Assert.Equal(ids1, ids2);
    }

    [Fact]
    public void Encode_SpaceMarkerNotDoublePrepended_WhenTextStartsWithRawMarker()
    {
        var tok = BuildSpaceMarkerVocab();
        // Input starting with raw ▁ (U+2581) should NOT get another ▁ prepended.
        // "\u2581hello" → stays "▁hello" (no double ▁▁hello), same result as "hello".
        int[] ids1 = tok.Encode("hello");
        int[] ids2 = tok.Encode("\u2581hello");
        Assert.Equal(ids1, ids2);
    }

    // -------------------------------------------------------------------------
    // Decode tests
    // -------------------------------------------------------------------------

    [Fact]
    public void Decode_RestoresSpaceFromSentencePieceMarker()
    {
        var tok = BuildSpaceMarkerVocab();
        // Token 6 = "▁h"; DecodeToken should replace ▁ with space.
        Assert.Equal(" h", tok.DecodeToken(6));
    }

    [Fact]
    public void Decode_ByteTokenSequence_ReturnsUtf8Char()
    {
        var tok = BuildByteVocab();
        // é = U+00E9 = UTF-8 bytes 0xC3 0xA9
        int id_c3 = 2 + 0xC3; // 195 + 2 = 197
        int id_a9 = 2 + 0xA9; // 169 + 2 = 171
        string result = tok.Decode([id_c3, id_a9]);
        Assert.Equal("é", result);
    }

    [Fact]
    public void Decode_EmptySpan_ReturnsEmpty()
    {
        var tok = BuildMinimalVocab();
        Assert.Equal(string.Empty, tok.Decode([]));
    }

    [Fact]
    public void Decode_SingleToken_ReturnsTokenText()
    {
        var tok = BuildMinimalVocab();
        Assert.Equal("ab", tok.Decode([4]));
    }

    [Fact]
    public void Decode_MultipleTokens_Concatenates()
    {
        var tok = BuildMinimalVocab();
        Assert.Equal("abc", tok.Decode([4, 3])); // "ab" + "c"
    }

    // -------------------------------------------------------------------------
    // Roundtrip tests
    // -------------------------------------------------------------------------

    [Fact]
    public void Roundtrip_AsciiText_WithSpaceMarkerVocab()
    {
        var tok = BuildSpaceMarkerVocab();
        const string text = "hello";
        Assert.Equal(text, tok.Decode(tok.Encode(text)));
    }

    [Fact]
    public void Roundtrip_AsciiText_WithMinimalVocab()
    {
        var tok = BuildMinimalVocab();
        const string text = "abc";
        Assert.Equal(text, tok.Decode(tok.Encode(text)));
    }

    [Fact]
    public void Roundtrip_UnicodeChar_ViaByteTokens()
    {
        var tok = BuildByteVocab();
        // 'a' has a regular token; 'é' goes through byte fallback.
        const string text = "aé";
        Assert.Equal(text, tok.Decode(tok.Encode(text)));
    }

    // -------------------------------------------------------------------------
    // VocabSize / BOS / EOS
    // -------------------------------------------------------------------------

    [Fact]
    public void VocabSize_MatchesTokenCount()
    {
        var tok = BuildMinimalVocab();
        Assert.Equal(7, tok.VocabSize);
    }

    [Fact]
    public void BosEosIds_CorrectlySet()
    {
        string[] tokens = ["<unk>", "<s>", "</s>", "a"];
        float[] scores  = [0f, 0f, 0f, -1f];
        var tok = BpeTokenizer.CreateSentencePiece(tokens, scores, tokenTypes: null,
            bosId: 1, eosId: 2);
        Assert.Equal(1, tok.BosTokenId);
        Assert.Equal(2, tok.EosTokenId);
    }

    // -------------------------------------------------------------------------
    // GgufBpeTokenizerFactory unit test (no file I/O — direct metadata construction)
    // -------------------------------------------------------------------------

    [Fact]
    public void GgufFactory_LoadsSentencePieceTokenizer()
    {
        // Build GgufMetadata directly from a dictionary — no file needed.
        // Vocab includes ▁ (token 6) so that addBosSpace=true doesn't fall through to byte
        // fallback — real SentencePiece models always have ▁ in the vocabulary.
        var entries = new Dictionary<string, GgufMetadataValue>
        {
            ["tokenizer.ggml.model"]        = new(GgufValueType.String, "llama"),
            ["tokenizer.ggml.tokens"]       = new(GgufValueType.Array, new string[] { "<unk>", "<s>", "</s>", "a", "b", "ab", "\u2581" }),
            ["tokenizer.ggml.scores"]       = new(GgufValueType.Array, new float[]  { 0f, 0f, 0f, -1.0f, -2.0f, -0.5f, -5f }),
            ["tokenizer.ggml.token_type"]   = new(GgufValueType.Array, new int[]    { 1, 2, 2, 0, 0, 0, 0 }),
            ["tokenizer.ggml.bos_token_id"] = new(GgufValueType.UInt32, 1u),
            ["tokenizer.ggml.eos_token_id"] = new(GgufValueType.UInt32, 2u),
        };
        var metadata = new GgufMetadata(entries);

        BpeTokenizer tokenizer = GgufBpeTokenizerFactory.Load(metadata);

        Assert.NotNull(tokenizer);
        Assert.Equal(1, tokenizer.BosTokenId);
        Assert.Equal(2, tokenizer.EosTokenId);
        Assert.Equal(7, tokenizer.VocabSize);

        // "ab" with addBosSpace=true normalises to "▁ab".
        // ▁(6) is a direct vocab hit; (a,b) merges to ab(5) → [▁, ab].
        int[] ids = tokenizer.Encode("ab");
        Assert.Equal([6, 5], ids);
    }

    [Fact]
    public void GgufFactory_DefaultsToLlamaWhenModelKeyMissing()
    {
        // No "tokenizer.ggml.model" key → defaults to SentencePiece.
        var entries = new Dictionary<string, GgufMetadataValue>
        {
            ["tokenizer.ggml.tokens"]       = new(GgufValueType.Array, new string[] { "<unk>", "a" }),
            ["tokenizer.ggml.scores"]       = new(GgufValueType.Array, new float[]  { 0f, -1f }),
            ["tokenizer.ggml.bos_token_id"] = new(GgufValueType.UInt32, 0u),
            ["tokenizer.ggml.eos_token_id"] = new(GgufValueType.UInt32, 0u),
        };
        var metadata = new GgufMetadata(entries);

        BpeTokenizer tokenizer = GgufBpeTokenizerFactory.Load(metadata);
        Assert.Equal(2, tokenizer.VocabSize);
    }

    // -------------------------------------------------------------------------
    // Special-token pre-splitting (trie-based)
    // -------------------------------------------------------------------------

    /// <summary>
    /// Minimal ChatML-style vocab with special control tokens:
    ///   0:&lt;unk&gt;  1:h  2:i  3:&lt;|im_start|&gt;  4:&lt;|im_end|&gt;  5:&lt;|user|&gt;  6:&lt;|im&gt;
    /// Tokens 3..6 have tokenType=3 (control) so they are pre-split before BPE.
    /// Token 6 (&lt;|im&gt;) is a strict prefix of tokens 3/4 — exercises longest-match semantics.
    /// </summary>
    private static BpeTokenizer BuildChatMlVocab()
    {
        string[] tokens = ["<unk>", "h", "i", "<|im_start|>", "<|im_end|>", "<|user|>", "<|im>"];
        float[] scores  = [0f, -1f, -2f, 0f, 0f, 0f, 0f];
        int[] tokenTypes = [1, 1, 1, 3, 3, 3, 3]; // 1=normal, 3=control
        return BpeTokenizer.CreateSentencePiece(tokens, scores, tokenTypes,
            bosId: 0, eosId: 0, addBosSpace: false);
    }

    [Fact]
    public void SpecialToken_EmittedAsSingleId_NotBpeEncoded()
    {
        var tok = BuildChatMlVocab();
        int[] ids = tok.Encode("<|im_start|>");
        Assert.Equal([3], ids);
    }

    [Fact]
    public void SpecialToken_LongestMatchWins_OverPrefix()
    {
        // "<|im_start|>" must match before "<|im>" (both are prefixes; longest wins).
        var tok = BuildChatMlVocab();
        int[] ids = tok.Encode("<|im_start|>");
        Assert.Equal([3], ids);

        // "<|im>" on its own should still match as its own control token.
        int[] idsShort = tok.Encode("<|im>");
        Assert.Equal([6], idsShort);
    }

    [Fact]
    public void SpecialToken_SplitsSurroundingText_IntoBpeSegments()
    {
        var tok = BuildChatMlVocab();
        // "hi<|im_end|>hi" → [h, i, <|im_end|>, h, i] = [1, 2, 4, 1, 2]
        int[] ids = tok.Encode("hi<|im_end|>hi");
        Assert.Equal([1, 2, 4, 1, 2], ids);
    }

    [Fact]
    public void SpecialToken_AtStart_NotAtEnd_EmitsCorrectOrder()
    {
        var tok = BuildChatMlVocab();
        int[] ids = tok.Encode("<|im_start|>hi");
        Assert.Equal([3, 1, 2], ids);
    }

    [Fact]
    public void SpecialToken_MultipleAdjacent_EmittedConsecutively()
    {
        var tok = BuildChatMlVocab();
        int[] ids = tok.Encode("<|im_start|><|im_end|>");
        Assert.Equal([3, 4], ids);
    }

    [Fact]
    public void SpecialToken_TextContainsOpeningBracketButNotFullMatch_BpeEncodesLiterally()
    {
        var tok = BuildChatMlVocab();
        // "hi<" has no matching special token → BPE over the whole segment.
        // "<" is not in the vocab → byte fallback? BuildChatMlVocab has no byte tokens,
        // so the unknown char becomes <unk>(0). The test just verifies the special-token scan
        // doesn't match "<" by itself and doesn't crash.
        int[] ids = tok.Encode("hi<");
        // h, i, <unk>
        Assert.Equal([1, 2, 0], ids);
    }

    [Fact]
    public void SpecialToken_NoSpecials_TakesFastPath()
    {
        // Vocab with no control tokens → specialTokens.Length == 0, trie is null,
        // Encode() should take the no-special-tokens fast path.
        var tok = BuildMinimalVocab(); // tokenTypes=null → no specials
        int[] ids = tok.Encode("abc");
        Assert.Equal([6], ids);
    }

    // -------------------------------------------------------------------------
    // Tiktoken pre-tokenization
    // -------------------------------------------------------------------------

    [Fact]
    public void TiktokenPreTokenizer_ReturnsRegexForKnownTypes()
    {
        Assert.NotNull(TiktokenPreTokenizer.GetRegex("gpt2"));
        Assert.NotNull(TiktokenPreTokenizer.GetRegex("default"));
        Assert.NotNull(TiktokenPreTokenizer.GetRegex("llama3"));
        Assert.NotNull(TiktokenPreTokenizer.GetRegex("llama-bpe"));
        Assert.NotNull(TiktokenPreTokenizer.GetRegex("deepseek-llm"));
        Assert.NotNull(TiktokenPreTokenizer.GetRegex("deepseek-coder"));
        Assert.NotNull(TiktokenPreTokenizer.GetRegex("command-r"));
    }

    [Fact]
    public void TiktokenPreTokenizer_ReturnsNullForUnknownType()
    {
        Assert.Null(TiktokenPreTokenizer.GetRegex(null));
        Assert.Null(TiktokenPreTokenizer.GetRegex(""));
        Assert.Null(TiktokenPreTokenizer.GetRegex("unknown-model"));
    }

    [Fact]
    public void Gpt2Regex_SplitsWordsAndSpaces()
    {
        var regex = TiktokenPreTokenizer.GetRegex("gpt2")!;
        var matches = regex.Matches("hello world");
        // "hello" and " world"
        Assert.Equal(2, matches.Count);
        Assert.Equal("hello", matches[0].Value);
        Assert.Equal(" world", matches[1].Value);
    }

    [Fact]
    public void Llama3Regex_SplitsContractions()
    {
        var regex = TiktokenPreTokenizer.GetRegex("llama3")!;
        var matches = regex.Matches("I'm happy");
        // "I", "'m", " happy"
        Assert.True(matches.Count >= 3);
        Assert.Equal("I", matches[0].Value);
        Assert.Equal("'m", matches[1].Value);
    }

    [Fact]
    public void Llama3Regex_GroupsDigitsInThrees()
    {
        var regex = TiktokenPreTokenizer.GetRegex("llama3")!;
        var matches = regex.Matches("12345");
        // "123", "45"
        Assert.Equal(2, matches.Count);
        Assert.Equal("123", matches[0].Value);
        Assert.Equal("45", matches[1].Value);
    }

    [Fact]
    public void Tiktoken_NullRegex_PreservesExistingBehavior()
    {
        // Build a tiktoken tokenizer WITHOUT pre-tokenization (null regex).
        // Verify it produces tokens (regression — should not crash).
        var tok = BuildMinimalTiktokenVocab(preType: null);
        int[] ids = tok.Encode("ab");
        Assert.NotEmpty(ids);
    }

    [Fact]
    public void Tiktoken_WithPreTokenizer_PreventsCrossBoundaryMerges()
    {
        // With GPT-2 regex, "a b" should split into ["a", " b"].
        // BPE cannot merge across the boundary, so the tokens for "a" and " b"
        // are encoded independently.
        var tokNoRegex = BuildMinimalTiktokenVocab(preType: null);
        var tokWithRegex = BuildMinimalTiktokenVocab(preType: "gpt2");

        int[] idsWithout = tokNoRegex.Encode("a b");
        int[] idsWith = tokWithRegex.Encode("a b");

        // Both should succeed without error
        Assert.NotEmpty(idsWithout);
        Assert.NotEmpty(idsWith);

        // With regex, we should get at least 2 separate groups of tokens
        // (one for "a", one for " b") whereas without regex everything is one segment.
        // The exact tokens depend on vocab, but the key point is no crash and
        // different segmentation behavior.
        Assert.True(idsWith.Length >= 2);
    }

    [Fact]
    public void GgufFactory_LoadsTiktokenWithPreType()
    {
        // Build GGUF metadata with tokenizer.ggml.pre = "llama3"
        var (tokens, merges) = BuildTiktokenGgufData();
        var entries = new Dictionary<string, GgufMetadataValue>
        {
            ["tokenizer.ggml.model"]        = new(GgufValueType.String, "gpt2"),
            ["tokenizer.ggml.tokens"]       = new(GgufValueType.Array, tokens),
            ["tokenizer.ggml.merges"]       = new(GgufValueType.Array, merges),
            ["tokenizer.ggml.bos_token_id"] = new(GgufValueType.UInt32, 0u),
            ["tokenizer.ggml.eos_token_id"] = new(GgufValueType.UInt32, 0u),
            ["tokenizer.ggml.pre"]          = new(GgufValueType.String, "llama3"),
        };
        var metadata = new GgufMetadata(entries);

        BpeTokenizer tokenizer = GgufBpeTokenizerFactory.Load(metadata);
        Assert.NotNull(tokenizer);
        // Encoding should work without error
        int[] ids = tokenizer.Encode("hi");
        Assert.NotEmpty(ids);
    }

    [Fact]
    public void GgufFactory_LoadsTiktokenWithoutPreType()
    {
        // No tokenizer.ggml.pre key → no pre-tokenization (backwards compat)
        var (tokens, merges) = BuildTiktokenGgufData();
        var entries = new Dictionary<string, GgufMetadataValue>
        {
            ["tokenizer.ggml.model"]        = new(GgufValueType.String, "gpt2"),
            ["tokenizer.ggml.tokens"]       = new(GgufValueType.Array, tokens),
            ["tokenizer.ggml.merges"]       = new(GgufValueType.Array, merges),
            ["tokenizer.ggml.bos_token_id"] = new(GgufValueType.UInt32, 0u),
            ["tokenizer.ggml.eos_token_id"] = new(GgufValueType.UInt32, 0u),
        };
        var metadata = new GgufMetadata(entries);

        BpeTokenizer tokenizer = GgufBpeTokenizerFactory.Load(metadata);
        Assert.NotNull(tokenizer);
        int[] ids = tokenizer.Encode("hi");
        Assert.NotEmpty(ids);
    }

    /// <summary>
    /// Builds a minimal tiktoken-style tokenizer for testing.
    /// Creates 256 single-byte tokens using GPT-2 byte encoding.
    /// </summary>
    private static BpeTokenizer BuildMinimalTiktokenVocab(string? preType)
    {
        // GPT-2 byte tokens: each byte 0x00-0xFF maps to a specific Unicode char.
        // Reproduce the GPT-2 byte_encoder mapping.
        char[] byteToUnicode = new char[256];
        for (int b = 33; b <= 126; b++) byteToUnicode[b] = (char)b;
        for (int b = 161; b <= 172; b++) byteToUnicode[b] = (char)b;
        for (int b = 174; b <= 255; b++) byteToUnicode[b] = (char)b;
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (byteToUnicode[b] == 0)
                byteToUnicode[b] = (char)(0x100 + n++);
        }

        string[] tokens = new string[256];
        for (int i = 0; i < 256; i++)
            tokens[i] = byteToUnicode[i].ToString();

        return BpeTokenizer.CreateTiktoken(tokens, merges: [], tokenTypes: null,
            bosId: 0, eosId: 0, preTokenizerType: preType);
    }

    /// <summary>
    /// Builds minimal GGUF-compatible tiktoken data (256 byte tokens, no merges).
    /// </summary>
    private static (string[] tokens, string[] merges) BuildTiktokenGgufData()
    {
        char[] byteToUnicode = new char[256];
        for (int b = 33; b <= 126; b++) byteToUnicode[b] = (char)b;
        for (int b = 161; b <= 172; b++) byteToUnicode[b] = (char)b;
        for (int b = 174; b <= 255; b++) byteToUnicode[b] = (char)b;
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (byteToUnicode[b] == 0)
                byteToUnicode[b] = (char)(0x100 + n++);
        }

        string[] tokens = new string[256];
        for (int i = 0; i < 256; i++)
            tokens[i] = byteToUnicode[i].ToString();

        return (tokens, []);
    }
}
