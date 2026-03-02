namespace DotLLM.Tokenizers;

/// <summary>
/// Trie node used internally by <see cref="Trie"/>.
/// </summary>
internal sealed class TrieNode
{
    public Dictionary<char, TrieNode>? Children;
    public int TokenId = -1;
    public float Score;
}

/// <summary>
/// Prefix-matching data structure for fast vocabulary lookup during BPE encoding.
/// Enables O(L) longest-prefix scan from a text position, where L is the match length.
/// Used both during initial character segmentation and in the BPE merge loop to check
/// whether adjacent symbol concatenations exist in the vocabulary.
/// </summary>
internal sealed class Trie
{
    private readonly TrieNode _root = new();

    /// <summary>Inserts a token into the trie.</summary>
    /// <param name="key">Token string (e.g. "▁hello").</param>
    /// <param name="tokenId">Vocabulary index for this token.</param>
    /// <param name="score">Merge priority score (higher = preferred merge in SentencePiece).</param>
    public void Add(ReadOnlySpan<char> key, int tokenId, float score)
    {
        TrieNode node = _root;
        foreach (char c in key)
        {
            node.Children ??= [];
            if (!node.Children.TryGetValue(c, out TrieNode? child))
            {
                child = new TrieNode();
                node.Children[c] = child;
            }
            node = child;
        }
        node.TokenId = tokenId;
        node.Score = score;
    }

    /// <summary>
    /// Finds the longest prefix of <paramref name="text"/> that exists in the trie.
    /// </summary>
    /// <param name="text">Text to scan from position 0.</param>
    /// <param name="tokenId">Token ID of the longest match, or -1 if none.</param>
    /// <param name="score">Score of the longest match.</param>
    /// <param name="matchLength">Number of characters matched (0 if no match).</param>
    /// <returns>True if at least one prefix matched.</returns>
    public bool TryMatchLongest(ReadOnlySpan<char> text, out int tokenId, out float score, out int matchLength)
    {
        TrieNode node = _root;
        int bestLen = 0;
        int bestId = -1;
        float bestScore = 0f;

        for (int i = 0; i < text.Length; i++)
        {
            if (node.Children == null || !node.Children.TryGetValue(text[i], out TrieNode? next))
                break;
            node = next;
            if (node.TokenId >= 0)
            {
                bestLen = i + 1;
                bestId = node.TokenId;
                bestScore = node.Score;
            }
        }

        if (bestLen == 0)
        {
            tokenId = -1;
            score = 0f;
            matchLength = 0;
            return false;
        }

        tokenId = bestId;
        score = bestScore;
        matchLength = bestLen;
        return true;
    }
}
