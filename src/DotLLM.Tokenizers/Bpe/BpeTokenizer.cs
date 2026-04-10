namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// Byte-pair encoding tokenizer supporting SentencePiece and tiktoken variants.
/// Vocabulary data is loaded at construction; all encoding and decoding is delegated
/// to the variant-specific <see cref="IBpeEncoding"/> implementation.
/// Special tokens (control/user-defined) are pre-split from the input and emitted
/// as single token IDs, bypassing BPE encoding.
/// </summary>
public sealed class BpeTokenizer : ITokenizer
{
    private readonly IBpeEncoding _encoding;

    /// <summary>
    /// Special tokens sorted by descending length so longer tokens match first.
    /// Kept for <see cref="BuildSpecialTokenTable"/>'s output contract; the hot-path
    /// encode loop uses <see cref="_specialTokenTrie"/> instead.
    /// </summary>
    private readonly (string Text, int Id)[] _specialTokens;

    /// <summary>
    /// Trie built from <see cref="_specialTokens"/> for O(L) longest-prefix matching
    /// at each text position. <see langword="null"/> when there are no special tokens
    /// (the tokenizer takes the fast path in <see cref="Encode"/>).
    /// </summary>
    private readonly Trie? _specialTokenTrie;

    /// <inheritdoc/>
    public int BosTokenId { get; }

    /// <inheritdoc/>
    public int EosTokenId { get; }

    /// <inheritdoc/>
    public int VocabSize { get; }

    private BpeTokenizer(IBpeEncoding encoding, (string Text, int Id)[] specialTokens,
        int bosId, int eosId, int vocabSize)
    {
        _encoding = encoding;
        _specialTokens = specialTokens;
        _specialTokenTrie = BuildSpecialTokenTrie(specialTokens);
        BosTokenId = bosId;
        EosTokenId = eosId;
        VocabSize = vocabSize;
    }

    private static Trie? BuildSpecialTokenTrie((string Text, int Id)[] specialTokens)
    {
        if (specialTokens.Length == 0)
            return null;
        var trie = new Trie();
        foreach (var (text, id) in specialTokens)
            trie.Add(text, id, score: 0f);
        return trie;
    }

    /// <summary>
    /// Creates a SentencePiece BPE tokenizer (Llama 1/2, Mistral, TinyLlama, SmolLM).
    /// </summary>
    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="scores">Unigram log-probability scores (higher = preferred merge).</param>
    /// <param name="tokenTypes">Per-token type flags (1=normal, 2=unknown, 3=control, 4=user-defined, 5=unused, 6=byte). Null = all normal.</param>
    /// <param name="bosId">Beginning-of-sequence token ID.</param>
    /// <param name="eosId">End-of-sequence token ID.</param>
    /// <param name="addBosSpace">Prepend ▁ to text that doesn't start with a space (matches SentencePiece default).</param>
    public static BpeTokenizer CreateSentencePiece(
        string[] tokens, float[] scores, int[]? tokenTypes,
        int bosId, int eosId, bool addBosSpace = true)
    {
        float[] safeScores = scores.Length == tokens.Length ? scores : new float[tokens.Length];
        var specialTokens = BuildSpecialTokenTable(tokens, tokenTypes);
        return new BpeTokenizer(
            new SentencePieceEncoding(tokens, safeScores, tokenTypes, addBosSpace),
            specialTokens, bosId, eosId, tokens.Length);
    }

    /// <summary>
    /// Creates a tiktoken BPE tokenizer (Llama 3, GPT-4).
    /// </summary>
    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="merges">Merge table entries in "A B" format; index = rank (lower = applied first).</param>
    /// <param name="tokenTypes">Per-token type flags. Null = all normal.</param>
    /// <param name="bosId">Beginning-of-sequence token ID.</param>
    /// <param name="eosId">End-of-sequence token ID.</param>
    /// <param name="preTokenizerType">GGUF <c>tokenizer.ggml.pre</c> value (e.g., "llama3", "gpt2"). Null = no pre-tokenization.</param>
    public static BpeTokenizer CreateTiktoken(
        string[] tokens, string[] merges, int[]? tokenTypes,
        int bosId, int eosId, string? preTokenizerType = null)
    {
        var specialTokens = BuildSpecialTokenTable(tokens, tokenTypes);
        var preRegex = TiktokenPreTokenizer.GetRegex(preTokenizerType);
        return new BpeTokenizer(
            new Gpt2TiktokenEncoding(tokens, merges, tokenTypes, preRegex),
            specialTokens, bosId, eosId, tokens.Length);
    }

    /// <inheritdoc/>
    public int[] Encode(string text)
    {
        if (text.Length == 0)
            return [];

        // Fast path: no special tokens to split on
        if (_specialTokens.Length == 0)
            return _encoding.Encode(text);

        return EncodeWithSpecialTokens(text);
    }

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> tokenIds) =>
        tokenIds.IsEmpty ? string.Empty : _encoding.Decode(tokenIds);

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) =>
        tokenIds.IsEmpty ? string.Empty : _encoding.Decode(tokenIds, stripBosSpace);

    /// <inheritdoc/>
    public string DecodeToken(int tokenId) => _encoding.DecodeToken(tokenId);

    /// <inheritdoc/>
    public int CountTokens(string text) => Encode(text).Length;

    /// <summary>
    /// Builds the special token table from vocabulary and token types.
    /// Control tokens (type 3) and user-defined tokens (type 4) that are non-empty
    /// and not single-byte are treated as special tokens for pre-splitting.
    /// Sorted by descending length for longest-match-first semantics.
    /// </summary>
    private static (string Text, int Id)[] BuildSpecialTokenTable(string[] tokens, int[]? tokenTypes)
    {
        if (tokenTypes is null)
            return [];

        var special = new List<(string Text, int Id)>();
        for (int i = 0; i < tokens.Length && i < tokenTypes.Length; i++)
        {
            // Type 3 = control, Type 4 = user-defined (added tokens)
            // Skip single chars and empty strings — they're not useful for pre-splitting
            if ((tokenTypes[i] == 3 || tokenTypes[i] == 4) &&
                tokens[i].Length > 1 &&
                !string.IsNullOrEmpty(tokens[i]))
            {
                special.Add((tokens[i], i));
            }
        }

        // Sort by descending length so longer tokens match first
        special.Sort((a, b) => b.Text.Length.CompareTo(a.Text.Length));
        return special.ToArray();
    }

    /// <summary>
    /// Encodes text with special token pre-splitting. Scans for special tokens via the
    /// pre-built trie (O(L) per position where L is the match length), emits their IDs
    /// directly, and BPE-encodes the text segments between them.
    /// </summary>
    private int[] EncodeWithSpecialTokens(string text)
    {
        var trie = _specialTokenTrie!; // Encode() fast-paths when _specialTokens.Length == 0
        var result = new List<int>();
        int pos = 0;
        bool isFirstSegment = true;

        while (pos < text.Length)
        {
            if (trie.TryMatchLongest(text.AsSpan(pos), out int matchedId, out _, out int matchedLen))
            {
                // Emit the special token ID directly
                result.Add(matchedId);
                pos += matchedLen;
                isFirstSegment = false;
            }
            else
            {
                // Find the next special token (or end of string)
                int nextSpecialPos = FindNextSpecialToken(text, pos + 1);
                string segment = text[pos..nextSpecialPos];

                // BPE-encode the segment.
                // First segment gets normal encoding (with BOS space prepend for SentencePiece).
                // Subsequent segments use EncodeSegment (no BOS space) to avoid spurious ▁ markers.
                if (segment.Length > 0)
                {
                    int[] segmentIds = isFirstSegment
                        ? _encoding.Encode(segment)
                        : _encoding.EncodeSegment(segment);
                    result.AddRange(segmentIds);
                }

                pos = nextSpecialPos;
                isFirstSegment = false;
            }
        }

        return result.ToArray();
    }

    /// <summary>
    /// Finds the position of the next special token starting from <paramref name="startPos"/>.
    /// Returns <c>text.Length</c> if no special token is found.
    /// </summary>
    private int FindNextSpecialToken(string text, int startPos)
    {
        var trie = _specialTokenTrie!;
        for (int pos = startPos; pos < text.Length; pos++)
        {
            if (trie.TryMatchLongest(text.AsSpan(pos), out _, out _, out _))
                return pos;
        }

        return text.Length;
    }
}
