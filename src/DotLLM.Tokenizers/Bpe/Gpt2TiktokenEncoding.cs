using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;

namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// GPT-2 / tiktoken BPE encoding (Llama 3, GPT-4).
/// Each character in a token string represents one byte via the GPT-2 byte-to-Unicode mapping.
/// Merge priority is determined by rank (lower rank = applied first).
/// </summary>
/// <remarks>
/// The merge rank dictionary uses <c>(leftTokenId, rightTokenId)</c> tuple keys to avoid
/// the string allocation that a <c>"leftText rightText"</c> key lookup would incur on every
/// bigram check during the hot encode path.
/// </remarks>
internal sealed class Gpt2TiktokenEncoding : IBpeEncoding
{
    // -------------------------------------------------------------------------
    // GPT-2 byte-to-unicode tables (static — shared across all instances)
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

    // -------------------------------------------------------------------------
    // Instance state
    // -------------------------------------------------------------------------

    private readonly string[] _idToToken;
    private readonly int[] _byteToTokenId;
    private readonly Trie _vocabTrie;

    /// <summary>
    /// Merge rank table keyed by (leftTokenId, rightTokenId) value-type tuple.
    /// Zero allocation per bigram lookup — eliminates the <c>string mergeKey</c> hot-path alloc.
    /// </summary>
    private readonly Dictionary<(int, int), int> _mergeRanks;

    private readonly int _unkId;

    /// <summary>
    /// Ordered pre-tokenization pipeline, applied to RAW text before byte-encoding: each expression
    /// further splits the spans the previous one produced, so merges cannot cross a boundary the
    /// model was trained to respect.
    /// </summary>
    /// <remarks>
    /// <para><see langword="null"/> or empty means <b>no pre-tokenization at all</b> — the whole
    /// input becomes one segment and merges run across it. That is rarely what an unrecognized
    /// pre-type should mean: the resulting stream mostly matches the reference and diverges at a
    /// handful of sites, which is the failure mode that motivated this change (see #237).</para>
    /// <para>An array rather than a single expression because a pre-type maps to a <i>pipeline</i>:
    /// the StarCoder/SmolLM family isolates every digit before applying its main pattern, and
    /// collapsing that to one expression silently mis-tokenizes.</para>
    /// </remarks>
    private readonly Regex[]? _preRegexes;

    internal Gpt2TiktokenEncoding(string[] tokens, string[] merges, int[]? tokenTypes, Regex[]? preRegexes = null)
    {
        _idToToken = tokens;
        _byteToTokenId = BpeCore.BuildByteToTokenId(tokens);

        _unkId = Array.FindIndex(tokens, t => t is "<unk>" or "<UNK>");
        if (_unkId < 0) _unkId = 0;

        _vocabTrie = new Trie();
        for (int i = 0; i < tokens.Length; i++)
        {
            if (!string.IsNullOrEmpty(tokens[i]))
                _vocabTrie.Add(tokens[i].AsSpan(), i, 0f);
        }

        // Build token string → ID reverse lookup (one pass at init time).
        var tokenToId = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
            tokenToId[tokens[i]] = i;

        // Parse "A B" merge entries → (idA, idB) tuple keys.
        var mergeRanks = new Dictionary<(int, int), int>(merges.Length);
        for (int rank = 0; rank < merges.Length; rank++)
        {
            int sep = merges[rank].IndexOf(' ');
            if (sep < 0) continue;
            string a = merges[rank][..sep], b = merges[rank][(sep + 1)..];
            if (tokenToId.TryGetValue(a, out int idA) && tokenToId.TryGetValue(b, out int idB))
                mergeRanks[(idA, idB)] = rank;
        }
        _mergeRanks = mergeRanks;
        _preRegexes = preRegexes;
    }

    /// <summary>
    /// Pre-tokenizes the <b>raw</b> text, then byte-encodes and BPEs each segment independently
    /// so merges cannot cross segment boundaries.
    /// </summary>
    /// <remarks>
    /// <para><b>Order matters: split first, byte-encode second.</b> The GPT-2 byte-level mapping
    /// sends byte 0x20 to U+0120, not to a literal space, so a pre-tokenization regex applied to
    /// already-encoded text can never match <c>\s</c>, a leading <c>' ?'</c>, or
    /// <c>\s+(?!\S)</c> — every whitespace-dependent alternative in every pattern silently dies.
    /// llama.cpp splits the raw text and byte-encodes each segment afterwards; this mirrors that.
    /// </para>
    /// </remarks>
    public int[] Encode(string text)
    {
        if (_preRegexes is null || _preRegexes.Length == 0)
            return EncodeRawSegment(text.AsSpan());

        var spans = PreTokenize(text.AsSpan(), _preRegexes);
        var result = new List<int>(text.Length);
        foreach ((int start, int length) in spans)
            EncodeRawSegmentInto(text.AsSpan(start, length), result);
        return result.ToArray();
    }

    /// <summary>
    /// Maps a raw text segment through the GPT-2 byte-level encoding (each UTF-8 byte becomes a
    /// specific Unicode char) and BPEs it. Uses <see cref="ArrayPool{T}"/> for both buffers.
    /// </summary>
    private int[] EncodeRawSegment(ReadOnlySpan<char> raw)
    {
        if (raw.IsEmpty) return [];

        int utf8Len = Encoding.UTF8.GetByteCount(raw);
        byte[] rentedUtf8 = ArrayPool<byte>.Shared.Rent(utf8Len);
        try
        {
            Encoding.UTF8.GetBytes(raw, rentedUtf8);
            char[] rentedGpt2 = ArrayPool<char>.Shared.Rent(utf8Len);
            try
            {
                for (int i = 0; i < utf8Len; i++)
                    rentedGpt2[i] = Gpt2ByteToUnicode[rentedUtf8[i]];
                return EncodeSegment(rentedGpt2.AsSpan(0, utf8Len));
            }
            finally { ArrayPool<char>.Shared.Return(rentedGpt2); }
        }
        finally { ArrayPool<byte>.Shared.Return(rentedUtf8); }
    }

    /// <summary>
    /// <see cref="EncodeRawSegment"/>, appending directly to <paramref name="dest"/> to avoid an
    /// intermediate <c>int[]</c> per segment.
    /// </summary>
    private void EncodeRawSegmentInto(ReadOnlySpan<char> raw, List<int> dest)
    {
        if (raw.IsEmpty) return;

        int utf8Len = Encoding.UTF8.GetByteCount(raw);
        byte[] rentedUtf8 = ArrayPool<byte>.Shared.Rent(utf8Len);
        try
        {
            Encoding.UTF8.GetBytes(raw, rentedUtf8);
            char[] rentedGpt2 = ArrayPool<char>.Shared.Rent(utf8Len);
            try
            {
                for (int i = 0; i < utf8Len; i++)
                    rentedGpt2[i] = Gpt2ByteToUnicode[rentedUtf8[i]];
                EncodeSegmentInto(rentedGpt2.AsSpan(0, utf8Len), dest);
            }
            finally { ArrayPool<char>.Shared.Return(rentedGpt2); }
        }
        finally { ArrayPool<byte>.Shared.Return(rentedUtf8); }
    }

    /// <summary>
    /// Splits <paramref name="text"/> into pre-token spans by applying each regex in
    /// <paramref name="pipeline"/> in order, every stage further splitting the previous stage's
    /// spans.
    /// </summary>
    /// <remarks>
    /// <para><b>Unmatched text is preserved as its own span.</b> Matching only, and encoding just
    /// the matches, silently drops any input a pattern does not cover. That is safe for the GPT-2
    /// expression, which ends in <c>|\s+</c> and therefore matches everything — but not for the
    /// StarCoder/SmolLM pattern, which deliberately omits that alternative. Dropping characters
    /// there would corrupt the token stream rather than merely re-split it.</para>
    /// <para>Mirrors llama.cpp's <c>unicode_regex_split</c> over its <c>regex_exprs</c> list.</para>
    /// </remarks>
    private static List<(int Start, int Length)> PreTokenize(ReadOnlySpan<char> text, Regex[] pipeline)
    {
        var spans = new List<(int Start, int Length)>(text.Length) { (0, text.Length) };

        foreach (Regex regex in pipeline)
        {
            var next = new List<(int Start, int Length)>(spans.Count * 2);
            foreach ((int start, int length) in spans)
            {
                if (length == 0) continue;

                int cursor = 0;
                foreach (ValueMatch match in regex.EnumerateMatches(text.Slice(start, length)))
                {
                    if (match.Length == 0) continue;
                    if (match.Index > cursor)
                        next.Add((start + cursor, match.Index - cursor));   // unmatched gap
                    next.Add((start + match.Index, match.Length));
                    cursor = match.Index + match.Length;
                }

                if (cursor < length)
                    next.Add((start + cursor, length - cursor));            // unmatched tail
            }
            spans = next;
        }

        return spans;
    }

    /// <summary>
    /// Encodes a single pre-tokenized segment using BPE merges.
    /// </summary>
    private int[] EncodeSegment(ReadOnlySpan<char> segment)
    {
        Symbol[] symbols = ArrayPool<Symbol>.Shared.Rent(segment.Length * 2);
        int symbolCount;
        try
        {
            symbolCount = BuildInitialSymbols(segment, symbols);

            var queue = new PriorityQueue<BgramEntry, (int, int)>(symbolCount);
            for (int i = 0; i < symbolCount - 1; i++)
                TryEnqueueBigram(symbols, i, i + 1, queue);

            RunMergeLoop(symbols, queue);
            return BpeCore.CollectTokenIds(symbols, symbolCount);
        }
        finally
        {
            ArrayPool<Symbol>.Shared.Return(symbols, clearArray: false);
        }
    }

    /// <summary>
    /// Encodes a segment and appends token IDs directly to <paramref name="dest"/>,
    /// avoiding intermediate <c>int[]</c> allocation per segment.
    /// </summary>
    private void EncodeSegmentInto(ReadOnlySpan<char> segment, List<int> dest)
    {
        Symbol[] symbols = ArrayPool<Symbol>.Shared.Rent(segment.Length * 2);
        try
        {
            int symbolCount = BuildInitialSymbols(segment, symbols);

            var queue = new PriorityQueue<BgramEntry, (int, int)>(symbolCount);
            for (int i = 0; i < symbolCount - 1; i++)
                TryEnqueueBigram(symbols, i, i + 1, queue);

            RunMergeLoop(symbols, queue);
            BpeCore.CollectTokenIds(symbols, symbolCount, dest);
        }
        finally
        {
            ArrayPool<Symbol>.Shared.Return(symbols, clearArray: false);
        }
    }

    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        // GPT-2 decode: every char in a token string is a GPT-2-encoded byte.
        // Map each char back to its byte, then UTF-8 decode the combined byte stream.
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

    public string DecodeToken(int tokenId)
    {
        if ((uint)tokenId >= (uint)_idToToken.Length) return string.Empty;
        string token = _idToToken[tokenId];
        // GPT-2: each token char encodes one byte.
        // stackalloc for typical tokens (≤256 chars), ArrayPool fallback for safety.
        byte[]? rented = null;
        try
        {
            Span<byte> bytes = token.Length <= 256
                ? stackalloc byte[256]
                : (rented = ArrayPool<byte>.Shared.Rent(token.Length));
            bytes = bytes[..token.Length];
            for (int i = 0; i < token.Length; i++)
            {
                int idx = (int)token[i];
                short bval = (uint)idx < (uint)Gpt2UnicodeToByteTable.Length
                    ? Gpt2UnicodeToByteTable[idx] : (short)-1;
                bytes[i] = bval >= 0 ? (byte)bval : (byte)0;
            }
            return Encoding.UTF8.GetString(bytes);
        }
        finally
        {
            if (rented is not null) ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private int BuildInitialSymbols(ReadOnlySpan<char> text, Symbol[] symbols)
    {
        int count = 0;
        int i = 0;
        Span<byte> utf8 = stackalloc byte[4]; // pre-allocate outside loop (CA2014)
        while (i < text.Length)
        {
            int charLen = char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])
                ? 2 : 1;
            ReadOnlySpan<char> cpSpan = text.Slice(i, charLen);
            i += charLen;

            if (_vocabTrie.TryMatchLongest(cpSpan, out int tokenId, out _, out int ml) && ml == charLen)
            {
                symbols[count] = new Symbol { Prev = count - 1, Next = count + 1, TokenId = tokenId };
                count++;
            }
            else
            {
                // Byte fallback: emit one symbol per UTF-8 byte.
                // If the byte has no <0xNN> token, emit <unk> rather than silently dropping it.
                int byteLen = Encoding.UTF8.GetBytes(cpSpan, utf8);
                for (int b = 0; b < byteLen; b++)
                {
                    int byteId = _byteToTokenId[utf8[b]];
                    int effectiveId = byteId >= 0 ? byteId : _unkId;
                    symbols[count] = new Symbol { Prev = count - 1, Next = count + 1, TokenId = effectiveId };
                    count++;
                }
            }
        }
        if (count > 0) symbols[count - 1].Next = -1;
        return count;
    }

    private void TryEnqueueBigram(
        Symbol[] symbols, int leftIdx, int rightIdx,
        PriorityQueue<BgramEntry, (int, int)> queue)
    {
        if (leftIdx < 0 || rightIdx < 0) return;

        // Zero allocation: tuple key is a value type — no string concat needed.
        if (!_mergeRanks.TryGetValue((symbols[leftIdx].TokenId, symbols[rightIdx].TokenId), out int rank)) return;

        // Resolve merged token ID via trie (stack-allocated concat — no heap alloc).
        // ArrayPool fallback for the rare case where combined token length exceeds 256.
        string leftText = _idToToken[symbols[leftIdx].TokenId];
        string rightText = _idToToken[symbols[rightIdx].TokenId];
        int totalLen = leftText.Length + rightText.Length;
        char[]? rented = null;
        try
        {
            Span<char> buf = totalLen <= 256
                ? stackalloc char[256]
                : (rented = ArrayPool<char>.Shared.Rent(totalLen));
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
        finally
        {
            if (rented is not null) ArrayPool<char>.Shared.Return(rented);
        }
    }

    private void RunMergeLoop(Symbol[] symbols, PriorityQueue<BgramEntry, (int, int)> queue)
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

            TryEnqueueBigram(symbols, left.Prev, entry.Left, queue);
            TryEnqueueBigram(symbols, entry.Left, nextIdx, queue);
        }
    }
}
