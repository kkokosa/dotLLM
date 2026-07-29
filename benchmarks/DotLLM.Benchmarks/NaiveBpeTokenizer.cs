using System.Text;
using System.Text.RegularExpressions;

namespace DotLLM.Benchmarks;

/// <summary>
/// Deliberately unoptimized reimplementation of the engine's tiktoken/GPT-2 BPE encoder,
/// used as the baseline in <see cref="TokenizerAllocationBenchmarks"/>.
/// </summary>
/// <remarks>
/// <para>
/// This is not a strawman. It runs the same algorithm as
/// <c>DotLLM.Tokenizers.Bpe.Gpt2TiktokenEncoding</c> and produces identical token ID
/// sequences — <see cref="TokenizerAllocationBenchmarks"/> asserts that in its setup.
/// What differs is only how the algorithm is expressed, using the constructs a C#
/// developer reaches for first:
/// </para>
/// <list type="bullet">
///   <item>a doubly-linked list of <see cref="SymbolNode"/> objects with real references,
///         one heap object per code point, instead of <c>Symbol</c> structs in a pooled
///         flat array addressed by index;</item>
///   <item><see cref="Dictionary{TKey, TValue}"/> keyed by <see cref="string"/> for both
///         vocabulary and merge-rank lookup, instead of a trie over
///         <see cref="ReadOnlySpan{T}"/> and a <c>(int, int)</c> tuple key;</item>
///   <item>string concatenation to form the bigram lookup key, instead of
///         <c>stackalloc</c>;</item>
///   <item><see cref="Regex.Matches(string)"/> materializing <see cref="Match"/> objects
///         and substrings, instead of <c>EnumerateMatches</c> over a span;</item>
///   <item><c>new</c> for every intermediate buffer, instead of
///         <see cref="System.Buffers.ArrayPool{T}"/>.</item>
/// </list>
/// </remarks>
internal sealed class NaiveBpeTokenizer
{
    /// <summary>Linked-list node: a class with references, one allocation per code point.</summary>
    private sealed class SymbolNode
    {
        public SymbolNode? Prev;
        public SymbolNode? Next;
        public int TokenId;
        public bool Deleted;

        /// <summary>
        /// Position in the initial symbol list. Mirrors the array index the engine uses as
        /// the priority-queue tie-breaker, so both implementations resolve equal-rank
        /// merges in the same order.
        /// </summary>
        public int Index;
    }

    /// <summary>Queued merge candidate — a class, so every candidate is an allocation.</summary>
    private sealed class Bigram
    {
        public SymbolNode Left = null!;
        public SymbolNode Right = null!;
        public int MergedId;
        public int ExpectedLeft;
        public int ExpectedRight;
    }

    private readonly string[] _idToToken;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<string, int> _mergeRanks;
    private readonly Dictionary<byte, char> _byteToUnicode;
    private readonly Dictionary<byte, int> _byteToTokenId;
    private readonly Regex? _preRegex;
    private readonly int _unkId;

    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="merges">Merge table entries in <c>"A B"</c> format; index = rank.</param>
    /// <param name="preRegex">Pre-tokenization regex, or <see langword="null"/> for none.</param>
    internal NaiveBpeTokenizer(string[] tokens, string[] merges, Regex? preRegex)
    {
        _idToToken = tokens;
        _preRegex = preRegex;

        _tokenToId = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
        {
            if (!string.IsNullOrEmpty(tokens[i]))
                _tokenToId[tokens[i]] = i;
        }

        _unkId = Array.FindIndex(tokens, t => t is "<unk>" or "<UNK>");
        if (_unkId < 0) _unkId = 0;

        // Merge ranks keyed by the raw "A B" line. Every lookup therefore has to build
        // that string, which is the allocation the engine's (int, int) key avoids.
        _mergeRanks = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int rank = 0; rank < merges.Length; rank++)
        {
            int sep = merges[rank].IndexOf(' ');
            if (sep < 0) continue;
            if (_tokenToId.ContainsKey(merges[rank][..sep])
                && _tokenToId.ContainsKey(merges[rank][(sep + 1)..]))
            {
                _mergeRanks[merges[rank]] = rank;
            }
        }

        _byteToUnicode = BuildByteToUnicode();

        _byteToTokenId = [];
        for (int i = 0; i < tokens.Length; i++)
        {
            if (TryParseByteLiteral(tokens[i], out byte b))
                _byteToTokenId[b] = i;
        }
    }

    /// <summary>Encodes <paramref name="text"/> to token IDs.</summary>
    public int[] Encode(string text)
    {
        // Byte-level GPT-2 mapping, via intermediate arrays and a string.
        byte[] utf8 = Encoding.UTF8.GetBytes(text);
        char[] mapped = new char[utf8.Length];
        for (int i = 0; i < utf8.Length; i++)
            mapped[i] = _byteToUnicode[utf8[i]];
        string gpt2Text = new(mapped);

        if (_preRegex is null)
            return EncodeSegment(gpt2Text).ToArray();

        // Regex.Matches materializes a MatchCollection of Match objects, and every
        // Match.Value is a fresh substring.
        var result = new List<int>();
        foreach (Match match in _preRegex.Matches(gpt2Text))
            result.AddRange(EncodeSegment(match.Value));
        return result.ToArray();
    }

    private List<int> EncodeSegment(string segment)
    {
        SymbolNode? head = BuildInitialSymbols(segment);
        if (head is null) return [];

        var queue = new PriorityQueue<Bigram, (int Rank, int Index)>();
        for (SymbolNode? s = head; s?.Next is not null; s = s.Next)
            TryEnqueueBigram(s, s.Next, queue);

        RunMergeLoop(queue);

        var ids = new List<int>();
        for (SymbolNode? s = head; s is not null; s = s.Next)
        {
            if (!s.Deleted) ids.Add(s.TokenId);
        }
        return ids;
    }

    /// <summary>
    /// Builds one node per code point, falling back to per-UTF-8-byte nodes when the code
    /// point has no vocabulary entry. Returns the list head, or null for empty input.
    /// </summary>
    private SymbolNode? BuildInitialSymbols(string text)
    {
        SymbolNode? head = null;
        SymbolNode? tail = null;
        int index = 0;

        int i = 0;
        while (i < text.Length)
        {
            int charLen = char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])
                ? 2 : 1;
            string codePoint = text.Substring(i, charLen);
            i += charLen;

            if (_tokenToId.TryGetValue(codePoint, out int tokenId))
            {
                Append(tokenId);
            }
            else
            {
                foreach (byte b in Encoding.UTF8.GetBytes(codePoint))
                    Append(_byteToTokenId.TryGetValue(b, out int byteId) ? byteId : _unkId);
            }
        }

        return head;

        void Append(int tokenId)
        {
            var node = new SymbolNode { TokenId = tokenId, Prev = tail, Index = index++ };
            if (tail is null) head = node;
            else tail.Next = node;
            tail = node;
        }
    }

    private void TryEnqueueBigram(SymbolNode left, SymbolNode right, PriorityQueue<Bigram, (int, int)> queue)
    {
        string leftText = _idToToken[left.TokenId];
        string rightText = _idToToken[right.TokenId];

        // Two string allocations per candidate: the merge-rank key and the concatenation
        // used to resolve the merged token ID.
        if (!_mergeRanks.TryGetValue(leftText + " " + rightText, out int rank))
            return;
        if (!_tokenToId.TryGetValue(leftText + rightText, out int mergedId))
            return;

        queue.Enqueue(
            new Bigram
            {
                Left = left,
                Right = right,
                MergedId = mergedId,
                ExpectedLeft = left.TokenId,
                ExpectedRight = right.TokenId,
            },
            (rank, left.Index));
    }

    private void RunMergeLoop(PriorityQueue<Bigram, (int, int)> queue)
    {
        while (queue.Count > 0)
        {
            Bigram entry = queue.Dequeue();
            SymbolNode left = entry.Left;
            SymbolNode right = entry.Right;

            // Same staleness checks as the engine, expressed over references.
            if (left.Deleted || right.Deleted
                || !ReferenceEquals(left.Next, right)
                || left.TokenId != entry.ExpectedLeft
                || right.TokenId != entry.ExpectedRight)
                continue;

            left.TokenId = entry.MergedId;
            right.Deleted = true;
            SymbolNode? next = right.Next;
            left.Next = next;
            if (next is not null) next.Prev = left;

            if (left.Prev is not null) TryEnqueueBigram(left.Prev, left, queue);
            if (next is not null) TryEnqueueBigram(left, next, queue);
        }
    }

    /// <summary>
    /// Pre-tokenization regex for a GGUF <c>tokenizer.ggml.pre</c> type.
    /// Duplicates <c>DotLLM.Tokenizers.Bpe.TiktokenPreTokenizer</c>, which is internal —
    /// the baseline has to stand on its own to be a fair comparison.
    /// </summary>
    internal static Regex? GetPreRegex(string? preType) => preType switch
    {
        "default" or "gpt2" or "deepseek-llm" => new Regex(
            @"(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
        "llama3" or "llama-bpe" or "command-r" => new Regex(
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
        "deepseek-coder" => new Regex(
            @"[a-zA-Z_][a-zA-Z0-9_]*|\p{N}+| ?[^\s\w]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
        _ => null,
    };

    /// <summary>
    /// GPT-2 byte-to-Unicode mapping. Printable ASCII (33–126) and Latin-1 (161–172,
    /// 174–255) map to the same code point; the rest map to U+0100+n.
    /// </summary>
    private static Dictionary<byte, char> BuildByteToUnicode()
    {
        var table = new Dictionary<byte, char>();
        for (int b = 33; b <= 126; b++) table[(byte)b] = (char)b;
        for (int b = 161; b <= 172; b++) table[(byte)b] = (char)b;
        for (int b = 174; b <= 255; b++) table[(byte)b] = (char)b;

        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!table.ContainsKey((byte)b))
                table[(byte)b] = (char)(0x100 + n++);
        }
        return table;
    }

    /// <summary>Parses a <c>&lt;0xNN&gt;</c> byte-literal vocabulary entry.</summary>
    private static bool TryParseByteLiteral(string token, out byte value)
    {
        value = 0;
        if (token.Length != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>')
            return false;
        return byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out value);
    }
}
