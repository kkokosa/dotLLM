using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;

namespace DotLLM.Tokenizers.Bpe;

/// <summary>BPE tokenizer variant.</summary>
public enum BpeVariant
{
    /// <summary>SentencePiece BPE (Llama 1/2, Mistral, TinyLlama, SmolLM).</summary>
    SentencePiece,
    /// <summary>tiktoken BPE (Llama 3, GPT-4). Regex pre-tokenization is not yet implemented.</summary>
    Tiktoken,
}

/// <summary>
/// Byte-pair encoding tokenizer supporting SentencePiece and tiktoken variants.
/// Vocabulary data is loaded at construction; no allocations occur on the encode hot path
/// beyond the rented symbol array and result array.
/// </summary>
public sealed class BpeTokenizer : ITokenizer
{
    private const char SpaceMarker = '\u2581'; // ▁ (SentencePiece word-boundary marker)

    // -------------------------------------------------------------------------
    // GPT-2 byte-to-unicode tables (used by tiktoken/gpt2 variant)
    // -------------------------------------------------------------------------

    /// <summary>
    /// Maps a raw byte value (0–255) to its GPT-2 Unicode character representation.
    /// GPT-2's byte_encoder maps printable ASCII (33–126) and Latin-1 (161–255, minus 173)
    /// to the same code-point; remaining bytes map to U+0100+n to avoid control characters.
    /// </summary>
    private static readonly char[] Gpt2ByteToUnicode = BuildGpt2ByteToUnicode();

    /// <summary>
    /// Reverse of <see cref="Gpt2ByteToUnicode"/>. Index = Unicode char (up to 0x0144).
    /// Value = byte value (0–255), or -1 if the char is not a GPT-2-encoded byte.
    /// </summary>
    private static readonly short[] Gpt2UnicodeToByteTable = BuildGpt2UnicodeToByteTable();

    private static char[] BuildGpt2ByteToUnicode()
    {
        char[] table = new char[256];
        // Printable ASCII 33..126 → same code point.
        for (int b = 33; b <= 126; b++) table[b] = (char)b;
        // Latin-1 supplement 161..172 → same code point.
        for (int b = 161; b <= 172; b++) table[b] = (char)b;
        // Latin-1 supplement 174..255 → same code point.
        for (int b = 174; b <= 255; b++) table[b] = (char)b;
        // Remaining bytes (0..32, 127..160, 173) → U+0100+n.
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (table[b] == 0) // not yet assigned
                table[b] = (char)(0x100 + n++);
        }
        return table;
    }

    private static short[] BuildGpt2UnicodeToByteTable()
    {
        char[] byteToChar = BuildGpt2ByteToUnicode();
        int maxChar = 0;
        foreach (char c in byteToChar) if (c > maxChar) maxChar = c;
        short[] table = new short[maxChar + 1];
        for (int i = 0; i < table.Length; i++) table[i] = -1;
        for (int b = 0; b < 256; b++) table[(int)byteToChar[b]] = (short)b;
        return table;
    }

    private readonly string[] _idToToken;
    private readonly float[] _scores;
    private readonly int[] _tokenTypes;
    private readonly Dictionary<string, int>? _mergeRanks; // tiktoken only
    private readonly int[] _byteToTokenId;                 // 256 entries; -1 = no token for that byte
    private readonly Trie _vocabTrie;
    private readonly BpeVariant _variant;
    private readonly bool _addBosSpace;

    /// <inheritdoc/>
    public int BosTokenId { get; }

    /// <inheritdoc/>
    public int EosTokenId { get; }

    /// <inheritdoc/>
    public int VocabSize => _idToToken.Length;

    private BpeTokenizer(
        string[] tokens, float[] scores, int[] tokenTypes,
        Dictionary<string, int>? mergeRanks,
        int bosId, int eosId,
        BpeVariant variant, bool addBosSpace)
    {
        _idToToken = tokens;
        _scores = scores;
        _tokenTypes = tokenTypes;
        _mergeRanks = mergeRanks;
        BosTokenId = bosId;
        EosTokenId = eosId;
        _variant = variant;
        _addBosSpace = addBosSpace;

        // Build byte-value → token ID mapping from entries like "<0xNN>".
        _byteToTokenId = new int[256];
        _byteToTokenId.AsSpan().Fill(-1);
        for (int i = 0; i < tokens.Length; i++)
        {
            if (TryParseByteLiteral(tokens[i], out byte b))
                _byteToTokenId[b] = i;
        }

        // Build trie for BPE merge loop and initial segmentation lookups.
        _vocabTrie = new Trie();
        for (int i = 0; i < tokens.Length; i++)
        {
            if (!string.IsNullOrEmpty(tokens[i]))
                _vocabTrie.Add(tokens[i].AsSpan(), i, scores[i]);
        }
    }

    /// <summary>
    /// Creates a SentencePiece BPE tokenizer (Llama 1/2, Mistral, TinyLlama, SmolLM).
    /// </summary>
    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="scores">Unigram log-probability scores (higher = preferred merge).</param>
    /// <param name="tokenTypes">Per-token type flags (0=normal, 1=unknown, 2=control, 3=byte, 5=user-defined). Null = all normal.</param>
    /// <param name="bosId">Beginning-of-sequence token ID.</param>
    /// <param name="eosId">End-of-sequence token ID.</param>
    /// <param name="addBosSpace">Prepend ▁ to text that doesn't start with a space (matches SentencePiece default).</param>
    public static BpeTokenizer CreateSentencePiece(
        string[] tokens, float[] scores, int[]? tokenTypes,
        int bosId, int eosId, bool addBosSpace = true)
    {
        int[] types = tokenTypes ?? new int[tokens.Length];
        float[] safeScores = scores.Length == tokens.Length ? scores : new float[tokens.Length];
        return new BpeTokenizer(tokens, safeScores, types, null, bosId, eosId,
            BpeVariant.SentencePiece, addBosSpace);
    }

    /// <summary>
    /// Creates a tiktoken BPE tokenizer (Llama 3, GPT-4).
    /// </summary>
    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="merges">Merge table entries in "A B" format; index = rank (lower = applied first).</param>
    /// <param name="tokenTypes">Per-token type flags. Null = all normal.</param>
    /// <param name="bosId">Beginning-of-sequence token ID.</param>
    /// <param name="eosId">End-of-sequence token ID.</param>
    public static BpeTokenizer CreateTiktoken(
        string[] tokens, string[] merges, int[]? tokenTypes, int bosId, int eosId)
    {
        var mergeRanks = new Dictionary<string, int>(merges.Length, StringComparer.Ordinal);
        for (int i = 0; i < merges.Length; i++)
            mergeRanks[merges[i]] = i;

        int[] types = tokenTypes ?? new int[tokens.Length];
        float[] scores = new float[tokens.Length]; // not used for tiktoken
        return new BpeTokenizer(tokens, scores, types, mergeRanks, bosId, eosId,
            BpeVariant.Tiktoken, addBosSpace: false);
    }

    /// <inheritdoc/>
    public int[] Encode(string text)
    {
        if (text.Length == 0) return [];
        return _variant switch
        {
            BpeVariant.SentencePiece => EncodeSentencePiece(text),
            BpeVariant.Tiktoken => EncodeTiktoken(text),
            _ => throw new NotSupportedException($"Unknown BPE variant: {_variant}"),
        };
    }

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        if (tokenIds.IsEmpty) return string.Empty;

        if (_variant == BpeVariant.Tiktoken)
            return DecodeGpt2(tokenIds);

        // SentencePiece path.
        var sb = new StringBuilder(tokenIds.Length * 4);
        byte[]? byteBuffer = null;
        int byteCount = 0;

        foreach (int id in tokenIds)
        {
            if ((uint)id >= (uint)_idToToken.Length) continue;
            string token = _idToToken[id];
            if (IsByteToken(token, out byte b))
            {
                // Accumulate bytes; flush when a non-byte token appears.
                byteBuffer ??= ArrayPool<byte>.Shared.Rent(16);
                if (byteCount >= byteBuffer.Length)
                {
                    byte[] larger = ArrayPool<byte>.Shared.Rent(byteBuffer.Length * 2);
                    byteBuffer.AsSpan(0, byteCount).CopyTo(larger);
                    ArrayPool<byte>.Shared.Return(byteBuffer);
                    byteBuffer = larger;
                }
                byteBuffer[byteCount++] = b;
            }
            else
            {
                FlushByteBuffer(sb, byteBuffer, ref byteCount);
                sb.Append(token.Replace(SpaceMarker, ' '));
            }
        }
        FlushByteBuffer(sb, byteBuffer, ref byteCount);

        if (byteBuffer != null)
            ArrayPool<byte>.Shared.Return(byteBuffer);

        string result = sb.ToString();

        // Strip the single leading space introduced by ▁ prepending (matches HF tokenizer behaviour).
        if (_addBosSpace && result.Length > 0 && result[0] == ' ')
            result = result[1..];

        return result;
    }

    /// <summary>
    /// GPT-2 decode: every char in a token string is a GPT-2-encoded byte.
    /// Map each char back to its byte, then UTF-8 decode the combined byte stream.
    /// </summary>
    private string DecodeGpt2(ReadOnlySpan<int> tokenIds)
    {
        // Upper-bound: each token can contribute at most 6 UTF-8 bytes per char.
        int maxBytes = tokenIds.Length * 8;
        byte[] buf = ArrayPool<byte>.Shared.Rent(maxBytes);
        int count = 0;

        foreach (int id in tokenIds)
        {
            if ((uint)id >= (uint)_idToToken.Length) continue;
            string token = _idToToken[id];
            foreach (char c in token)
            {
                if (count >= buf.Length)
                {
                    byte[] larger = ArrayPool<byte>.Shared.Rent(buf.Length * 2);
                    buf.AsSpan(0, count).CopyTo(larger);
                    ArrayPool<byte>.Shared.Return(buf);
                    buf = larger;
                }
                // Look up the byte value for this GPT-2 Unicode char.
                int idx = (int)c;
                if ((uint)idx < (uint)Gpt2UnicodeToByteTable.Length)
                {
                    short b = Gpt2UnicodeToByteTable[idx];
                    if (b >= 0) buf[count++] = (byte)b;
                }
            }
        }

        string result = Encoding.UTF8.GetString(buf, 0, count);
        ArrayPool<byte>.Shared.Return(buf);
        return result;
    }

    /// <inheritdoc/>
    public string DecodeToken(int tokenId)
    {
        if ((uint)tokenId >= (uint)_idToToken.Length) return string.Empty;
        string token = _idToToken[tokenId];

        if (_variant == BpeVariant.Tiktoken)
        {
            // GPT-2: each token char encodes one byte.
            byte[] bytes = new byte[token.Length];
            for (int i = 0; i < token.Length; i++)
            {
                int idx = (int)token[i];
                short bval = (uint)idx < (uint)Gpt2UnicodeToByteTable.Length
                    ? Gpt2UnicodeToByteTable[idx] : (short)-1;
                bytes[i] = bval >= 0 ? (byte)bval : (byte)0;
            }
            return Encoding.UTF8.GetString(bytes);
        }

        if (IsByteToken(token, out byte b))
        {
            // Return single-byte interpretation; caller should use Decode for multi-byte sequences.
            return Encoding.Latin1.GetString([b]);
        }
        return token.Replace(SpaceMarker, ' ');
    }

    /// <inheritdoc/>
    public int CountTokens(string text) => Encode(text).Length;

    // -------------------------------------------------------------------------
    // SentencePiece encode path
    // -------------------------------------------------------------------------

    private int[] EncodeSentencePiece(string text)
    {
        // 1. Normalize: replace ' ' with ▁ throughout; optionally prepend ▁.
        string normalized = text.Replace(' ', SpaceMarker);
        if (_addBosSpace && (normalized.Length == 0 || normalized[0] != SpaceMarker))
            normalized = SpaceMarker + normalized;

        // 2. Build initial symbol list: one symbol per Unicode code point.
        //    Rent a buffer that fits the worst case (each char is a separate symbol).
        Symbol[] symbols = ArrayPool<Symbol>.Shared.Rent(normalized.Length);
        int symbolCount;
        try
        {
            symbolCount = BuildInitialSymbols(normalized, symbols);

            // 3. Run BPE merge loop using a min-heap with (-score, leftIdx) as priority.
            var queue = new PriorityQueue<BgramEntry, (float, int)>();
            for (int i = 0; i < symbolCount - 1; i++)
                TryEnqueueBigramSp(symbols, i, i + 1, queue);

            RunMergeLoopSp(symbols, queue);

            // 4. Collect surviving symbols.
            return CollectTokenIds(symbols, symbolCount);
        }
        finally
        {
            ArrayPool<Symbol>.Shared.Return(symbols, clearArray: false);
        }
    }

    private int BuildInitialSymbols(string text, Symbol[] symbols)
    {
        int count = 0;
        int i = 0;
        // Pre-allocate outside the loop to satisfy CA2014.
        Span<byte> utf8 = stackalloc byte[4];
        while (i < text.Length)
        {
            // Consume one Unicode code point (1 or 2 chars for a surrogate pair).
            int charLen = char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])
                ? 2 : 1;
            ReadOnlySpan<char> cpSpan = text.AsSpan(i, charLen);
            i += charLen;

            // Try exact vocab match for this code point.
            if (_vocabTrie.TryMatchLongest(cpSpan, out int tokenId, out _, out int ml) && ml == charLen)
            {
                symbols[count] = new Symbol { Prev = count - 1, Next = count + 1, TokenId = tokenId };
                count++;
            }
            else
            {
                // Byte fallback: emit one symbol per UTF-8 byte.
                int byteLen = Encoding.UTF8.GetBytes(cpSpan, utf8);
                for (int b = 0; b < byteLen; b++)
                {
                    int byteId = _byteToTokenId[utf8[b]];
                    if (byteId < 0) continue; // no token for this byte; skip
                    symbols[count] = new Symbol { Prev = count - 1, Next = count + 1, TokenId = byteId };
                    count++;
                }
            }
        }

        if (count > 0)
            symbols[count - 1].Next = -1;

        return count;
    }

    private void TryEnqueueBigramSp(
        Symbol[] symbols, int leftIdx, int rightIdx,
        PriorityQueue<BgramEntry, (float, int)> queue)
    {
        if (leftIdx < 0 || rightIdx < 0) return;
        string leftText = _idToToken[symbols[leftIdx].TokenId];
        string rightText = _idToToken[symbols[rightIdx].TokenId];
        int totalLen = leftText.Length + rightText.Length;

        // Build concatenation on the stack to avoid string allocation.
        Span<char> buf = totalLen <= 256 ? stackalloc char[256] : new char[totalLen];
        Span<char> concat = buf[..totalLen];
        leftText.AsSpan().CopyTo(concat);
        rightText.AsSpan().CopyTo(concat[leftText.Length..]);

        if (_vocabTrie.TryMatchLongest(concat, out int mergedId, out float score, out int ml)
            && ml == totalLen)
        {
            int leftToken = symbols[leftIdx].TokenId;
            int rightToken = symbols[rightIdx].TokenId;
            // Negate score so the min-heap behaves as a max-heap by score.
            // Use leftIdx as secondary key: lower index = higher priority on ties.
            queue.Enqueue(new BgramEntry(leftIdx, rightIdx, mergedId, leftToken, rightToken),
                (-score, leftIdx));
        }
    }

    private void RunMergeLoopSp(Symbol[] symbols, PriorityQueue<BgramEntry, (float, int)> queue)
    {
        while (queue.Count > 0)
        {
            BgramEntry entry = queue.Dequeue();
            ref Symbol left = ref symbols[entry.Left];
            ref Symbol right = ref symbols[entry.Right];

            // Discard stale entries: symbol deleted, no longer adjacent, or token changed since enqueue.
            // The token-ID check catches the case where a symbol was merged into something else
            // (changing its TokenId) without being marked as deleted — e.g. an old (P, L) bigram
            // where L was later the left side of a merge and its TokenId changed.
            if (left.Deleted || right.Deleted
                || left.Next != entry.Right
                || left.TokenId != entry.ExpectedLeft
                || right.TokenId != entry.ExpectedRight)
                continue;

            // Merge: replace left with merged token, delete right.
            left.TokenId = entry.MergedId;
            right.Deleted = true;
            int nextIdx = right.Next;
            left.Next = nextIdx;
            if (nextIdx >= 0) symbols[nextIdx].Prev = entry.Left;

            // Enqueue new bigrams formed by the merged symbol with its neighbours.
            TryEnqueueBigramSp(symbols, left.Prev, entry.Left, queue);
            TryEnqueueBigramSp(symbols, entry.Left, nextIdx, queue);
        }
    }

    // -------------------------------------------------------------------------
    // tiktoken encode path
    // -------------------------------------------------------------------------

    private int[] EncodeTiktoken(string text)
    {
        // Convert text to GPT-2 byte-level Unicode encoding:
        // each UTF-8 byte of the input maps to a specific Unicode char (byte_encoder in GPT-2).
        // This is required because GPT-2/BBPE vocab tokens are stored in this encoding
        // (e.g., space 0x20 → 'Ġ' U+0120 rather than ' ' U+0020).
        byte[] utf8Bytes = Encoding.UTF8.GetBytes(text);
        char[] gpt2Chars = ArrayPool<char>.Shared.Rent(utf8Bytes.Length);
        for (int i = 0; i < utf8Bytes.Length; i++)
            gpt2Chars[i] = Gpt2ByteToUnicode[utf8Bytes[i]];
        string gpt2Text = new string(gpt2Chars, 0, utf8Bytes.Length);
        ArrayPool<char>.Shared.Return(gpt2Chars);

        // TODO: implement regex pre-tokenization using tokenizer.ggml.pre pattern.
        // Without it, this path treats the whole GPT-2-encoded text as one segment, which is
        // incorrect for most tiktoken models (splits should happen at word boundaries first).
        Symbol[] symbols = ArrayPool<Symbol>.Shared.Rent(gpt2Text.Length * 2);
        int symbolCount;
        try
        {
            symbolCount = BuildInitialSymbolsTiktoken(gpt2Text, symbols);

            var queue = new PriorityQueue<BgramEntry, (int, int)>();
            for (int i = 0; i < symbolCount - 1; i++)
                TryEnqueueBigramTt(symbols, i, i + 1, queue);

            RunMergeLoopTt(symbols, queue);
            return CollectTokenIds(symbols, symbolCount);
        }
        finally
        {
            ArrayPool<Symbol>.Shared.Return(symbols, clearArray: false);
        }
    }

    private int BuildInitialSymbolsTiktoken(string text, Symbol[] symbols)
    {
        int count = 0;
        int i = 0;
        Span<byte> utf8 = stackalloc byte[4]; // pre-allocate outside loop (CA2014)
        while (i < text.Length)
        {
            int charLen = char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])
                ? 2 : 1;
            ReadOnlySpan<char> cpSpan = text.AsSpan(i, charLen);
            i += charLen;

            if (_vocabTrie.TryMatchLongest(cpSpan, out int tokenId, out _, out int ml) && ml == charLen)
            {
                symbols[count] = new Symbol { Prev = count - 1, Next = count + 1, TokenId = tokenId };
                count++;
            }
            else
            {
                int byteLen = Encoding.UTF8.GetBytes(cpSpan, utf8);
                for (int b = 0; b < byteLen; b++)
                {
                    int byteId = _byteToTokenId[utf8[b]];
                    if (byteId < 0) continue;
                    symbols[count] = new Symbol { Prev = count - 1, Next = count + 1, TokenId = byteId };
                    count++;
                }
            }
        }
        if (count > 0) symbols[count - 1].Next = -1;
        return count;
    }

    private void TryEnqueueBigramTt(
        Symbol[] symbols, int leftIdx, int rightIdx,
        PriorityQueue<BgramEntry, (int, int)> queue)
    {
        if (leftIdx < 0 || rightIdx < 0 || _mergeRanks == null) return;
        string leftText = _idToToken[symbols[leftIdx].TokenId];
        string rightText = _idToToken[symbols[rightIdx].TokenId];

        // tiktoken merge format: "A B" (space-separated).
        string mergeKey = leftText + " " + rightText;
        if (!_mergeRanks.TryGetValue(mergeKey, out int rank)) return;

        // Resolve merged token ID via trie.
        int totalLen = leftText.Length + rightText.Length;
        Span<char> buf = totalLen <= 256 ? stackalloc char[256] : new char[totalLen];
        Span<char> concat = buf[..totalLen];
        leftText.AsSpan().CopyTo(concat);
        rightText.AsSpan().CopyTo(concat[leftText.Length..]);

        if (_vocabTrie.TryMatchLongest(concat, out int mergedId, out _, out int ml) && ml == totalLen)
        {
            int leftToken = symbols[leftIdx].TokenId;
            int rightToken = symbols[rightIdx].TokenId;
            queue.Enqueue(new BgramEntry(leftIdx, rightIdx, mergedId, leftToken, rightToken),
                (rank, leftIdx));
        }
    }

    private void RunMergeLoopTt(Symbol[] symbols, PriorityQueue<BgramEntry, (int, int)> queue)
    {
        while (queue.Count > 0)
        {
            BgramEntry entry = queue.Dequeue();
            ref Symbol left = ref symbols[entry.Left];
            ref Symbol right = ref symbols[entry.Right];

            if (left.Deleted || right.Deleted
                || left.Next != entry.Right
                || left.TokenId != entry.ExpectedLeft
                || right.TokenId != entry.ExpectedRight)
                continue;

            left.TokenId = entry.MergedId;
            right.Deleted = true;
            int nextIdx = right.Next;
            left.Next = nextIdx;
            if (nextIdx >= 0) symbols[nextIdx].Prev = entry.Left;

            TryEnqueueBigramTt(symbols, left.Prev, entry.Left, queue);
            TryEnqueueBigramTt(symbols, entry.Left, nextIdx, queue);
        }
    }

    // -------------------------------------------------------------------------
    // Shared helpers
    // -------------------------------------------------------------------------

    private static int[] CollectTokenIds(Symbol[] symbols, int symbolCount)
    {
        int count = 0;
        for (int i = 0; i < symbolCount; i++)
            if (!symbols[i].Deleted) count++;

        int[] result = new int[count];
        int ri = 0;
        for (int i = 0; i < symbolCount; i++)
            if (!symbols[i].Deleted) result[ri++] = symbols[i].TokenId;
        return result;
    }

    private static void FlushByteBuffer(StringBuilder sb, byte[]? buffer, ref int count)
    {
        if (count == 0) return;
        sb.Append(Encoding.UTF8.GetString(buffer!, 0, count));
        count = 0;
    }

    /// <summary>Returns true if <paramref name="token"/> is in the <c>&lt;0xNN&gt;</c> byte-literal format.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool TryParseByteLiteral(string token, out byte value)
    {
        if (token.Length == 6
            && token[0] == '<' && token[1] == '0' && token[2] == 'x'
            && token[5] == '>'
            && IsHexDigit(token[3]) && IsHexDigit(token[4]))
        {
            value = (byte)(HexValue(token[3]) << 4 | HexValue(token[4]));
            return true;
        }
        value = 0;
        return false;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsByteToken(string token, out byte value) =>
        TryParseByteLiteral(token, out value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsHexDigit(char c) =>
        (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int HexValue(char c) =>
        c >= 'a' ? c - 'a' + 10 : c >= 'A' ? c - 'A' + 10 : c - '0';

    // -------------------------------------------------------------------------
    // Private data structures
    // -------------------------------------------------------------------------

    private struct Symbol
    {
        public int Prev;     // index of previous live symbol; -1 = head
        public int Next;     // index of next live symbol; -1 = tail
        public int TokenId;
        public bool Deleted;
    }

    private readonly struct BgramEntry(int left, int right, int mergedId, int expectedLeft, int expectedRight)
    {
        public int Left { get; } = left;
        public int Right { get; } = right;
        public int MergedId { get; } = mergedId;
        /// <summary>TokenId the left symbol must still have for this entry to be valid.</summary>
        public int ExpectedLeft { get; } = expectedLeft;
        /// <summary>TokenId the right symbol must still have for this entry to be valid.</summary>
        public int ExpectedRight { get; } = expectedRight;
    }
}
