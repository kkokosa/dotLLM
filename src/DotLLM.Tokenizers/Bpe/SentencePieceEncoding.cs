using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;

namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// SentencePiece BPE encoding (Llama 1/2, Mistral, TinyLlama, SmolLM).
/// Uses score-based merge priority: higher log-prob score = applied first.
/// Byte fallback for unmapped code-points uses &lt;0xNN&gt; tokens where available,
/// or emits <c>_unkId</c> (the &lt;unk&gt; token) when a byte has no vocab entry.
/// </summary>
internal sealed class SentencePieceEncoding : IBpeEncoding
{
    private const char SpaceMarker = '\u2581'; // ▁ (SentencePiece word-boundary marker)

    private readonly string[] _idToToken;
    private readonly int[] _byteToTokenId;
    private readonly Trie _vocabTrie;
    private readonly bool _addBosSpace;
    private readonly int _unkId;

    internal SentencePieceEncoding(string[] tokens, float[] scores, int[]? tokenTypes, bool addBosSpace)
    {
        _idToToken = tokens;
        _addBosSpace = addBosSpace;
        _byteToTokenId = BpeCore.BuildByteToTokenId(tokens);

        _unkId = Array.FindIndex(tokens, t => t is "<unk>" or "<UNK>");
        if (_unkId < 0) _unkId = 0;

        _vocabTrie = new Trie();
        for (int i = 0; i < tokens.Length; i++)
        {
            if (!string.IsNullOrEmpty(tokens[i]))
                _vocabTrie.Add(tokens[i].AsSpan(), i, scores[i]);
        }
    }

    public int[] Encode(string text)
    {
        // 1. Normalize: replace ' ' with ▁ throughout; optionally prepend ▁.
        //    Uses ArrayPool to avoid string allocations on the hot path.
        bool needPrepend = _addBosSpace && (text.Length == 0 || (text[0] != ' ' && text[0] != SpaceMarker));
        int normalizedLen = text.Length + (needPrepend ? 1 : 0);
        char[] rentedNorm = ArrayPool<char>.Shared.Rent(normalizedLen);
        try
        {
            int offset = needPrepend ? 1 : 0;
            if (needPrepend) rentedNorm[0] = SpaceMarker;
            text.AsSpan().CopyTo(rentedNorm.AsSpan(offset));
            MemoryExtensions.Replace(rentedNorm.AsSpan(offset, text.Length), ' ', SpaceMarker);
            ReadOnlySpan<char> normalized = rentedNorm.AsSpan(0, normalizedLen);

            // 2. Build initial symbol list: one symbol per Unicode code point.
            Symbol[] symbols = ArrayPool<Symbol>.Shared.Rent(normalizedLen);
            int symbolCount;
            try
            {
                symbolCount = BuildInitialSymbols(normalized, symbols);

                // 3. Run BPE merge loop using a min-heap with (-score, leftIdx) as priority.
                var queue = new PriorityQueue<BgramEntry, (float, int)>(symbolCount);
                for (int i = 0; i < symbolCount - 1; i++)
                    TryEnqueueBigram(symbols, i, i + 1, queue);

                RunMergeLoop(symbols, queue);

                // 4. Collect surviving symbols.
                return BpeCore.CollectTokenIds(symbols, symbolCount);
            }
            finally
            {
                ArrayPool<Symbol>.Shared.Return(symbols, clearArray: false);
            }
        }
        finally
        {
            ArrayPool<char>.Shared.Return(rentedNorm);
        }
    }

    public string Decode(ReadOnlySpan<int> tokenIds)
    {
        var sb = new StringBuilder(tokenIds.Length * 4);
        byte[]? byteBuffer = null;
        int byteCount = 0;

        foreach (int id in tokenIds)
        {
            if ((uint)id >= (uint)_idToToken.Length) continue;
            string token = _idToToken[id];
            if (BpeCore.IsByteToken(token, out byte b))
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
                BpeCore.FlushByteBuffer(sb, byteBuffer, ref byteCount);
                int startLen = sb.Length;
                sb.Append(token);
                sb.Replace(SpaceMarker, ' ', startLen, token.Length);
            }
        }
        BpeCore.FlushByteBuffer(sb, byteBuffer, ref byteCount);

        if (byteBuffer != null)
            ArrayPool<byte>.Shared.Return(byteBuffer);

        // Strip the single leading space introduced by ▁ prepending (matches HF tokeniser behaviour).
        bool stripLeading = _addBosSpace && sb.Length > 0 && sb[0] == ' ';
        return stripLeading ? sb.ToString(1, sb.Length - 1) : sb.ToString();
    }

    public string DecodeToken(int tokenId)
    {
        if ((uint)tokenId >= (uint)_idToToken.Length) return string.Empty;
        string token = _idToToken[tokenId];
        if (BpeCore.IsByteToken(token, out byte b))
        {
            // Return single-byte interpretation; caller should use Decode for multi-byte sequences.
            Span<byte> single = stackalloc byte[] { b };
            return Encoding.Latin1.GetString(single);
        }
        return token.Contains(SpaceMarker) ? token.Replace(SpaceMarker, ' ') : token;
    }

    private int BuildInitialSymbols(ReadOnlySpan<char> text, Symbol[] symbols)
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
            ReadOnlySpan<char> cpSpan = text.Slice(i, charLen);
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

        if (count > 0)
            symbols[count - 1].Next = -1;

        return count;
    }

    private void TryEnqueueBigram(
        Symbol[] symbols, int leftIdx, int rightIdx,
        PriorityQueue<BgramEntry, (float, int)> queue)
    {
        if (leftIdx < 0 || rightIdx < 0) return;
        string leftText = _idToToken[symbols[leftIdx].TokenId];
        string rightText = _idToToken[symbols[rightIdx].TokenId];
        int totalLen = leftText.Length + rightText.Length;

        // Build concatenation on the stack to avoid string allocation.
        // ArrayPool fallback for the rare case where combined token length exceeds 256.
        char[]? rented = null;
        try
        {
            Span<char> buf = totalLen <= 256
                ? stackalloc char[256]
                : (rented = ArrayPool<char>.Shared.Rent(totalLen));
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
        finally
        {
            if (rented is not null) ArrayPool<char>.Shared.Return(rented);
        }
    }

    private void RunMergeLoop(Symbol[] symbols, PriorityQueue<BgramEntry, (float, int)> queue)
    {
        while (queue.Count > 0)
        {
            BgramEntry entry = queue.Dequeue();
            ref Symbol left = ref symbols[entry.Left];
            ref Symbol right = ref symbols[entry.Right];

            // Discard stale entries: symbol deleted, no longer adjacent, or token changed since enqueue.
            // The token-ID check catches the case where a symbol was merged into something else
            // (changing its TokenId) without being marked as deleted.
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
            TryEnqueueBigram(symbols, left.Prev, entry.Left, queue);
            TryEnqueueBigram(symbols, entry.Left, nextIdx, queue);
        }
    }
}
