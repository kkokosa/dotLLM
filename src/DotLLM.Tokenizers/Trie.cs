using System.Threading;

namespace DotLLM.Tokenizers;

/// <summary>
/// Prefix-matching data structure for fast vocabulary lookup during BPE encoding.
/// Enables O(L) longest-prefix scan from a text position, where L is the match length.
/// Used both during initial character segmentation and in the BPE merge loop to check
/// whether adjacent symbol concatenations exist in the vocabulary.
/// </summary>
internal sealed class Trie
{
    private readonly object _freezeGate = new();
    private List<BuilderNode>? _builderNodes = [new()];
    private FlatNode[] _nodes = [new(0, 0, -1, 0f)];
    private FlatEdge[] _edges = [];
    private int _isFrozen;

    /// <summary>Inserts a token into the trie.</summary>
    /// <param name="key">Token string (e.g. "▁hello").</param>
    /// <param name="tokenId">Vocabulary index for this token.</param>
    /// <param name="score">Merge priority score (higher = preferred merge in SentencePiece).</param>
    public void Add(ReadOnlySpan<char> key, int tokenId, float score)
    {
        if (Volatile.Read(ref _isFrozen) != 0)
            throw new InvalidOperationException("Cannot add tokens after trie has been frozen.");

        lock (_freezeGate)
        {
            if (Volatile.Read(ref _isFrozen) != 0 || _builderNodes is null)
                throw new InvalidOperationException("Cannot add tokens after trie has been frozen.");

            List<BuilderNode> builderNodes = _builderNodes;
            int nodeIndex = 0;
            foreach (char c in key)
            {
                Dictionary<char, int> children = builderNodes[nodeIndex].Children ??= [];
                if (!children.TryGetValue(c, out int childIndex))
                {
                    childIndex = builderNodes.Count;
                    children[c] = childIndex;
                    builderNodes.Add(new BuilderNode());
                }
                nodeIndex = childIndex;
            }

            builderNodes[nodeIndex].TokenId = tokenId;
            builderNodes[nodeIndex].Score = score;
        }
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
        EnsureFrozen();

        int nodeIndex = 0;
        int bestLen = 0;
        int bestId = -1;
        float bestScore = 0f;

        for (int i = 0; i < text.Length; i++)
        {
            int nextNodeIndex = FindChild(nodeIndex, text[i]);
            if (nextNodeIndex < 0)
                break;

            nodeIndex = nextNodeIndex;
            if (_nodes[nodeIndex].TokenId >= 0)
            {
                bestLen = i + 1;
                bestId = _nodes[nodeIndex].TokenId;
                bestScore = _nodes[nodeIndex].Score;
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

    private void EnsureFrozen()
    {
        if (Volatile.Read(ref _isFrozen) != 0)
            return;

        lock (_freezeGate)
        {
            if (Volatile.Read(ref _isFrozen) != 0)
                return;

            List<BuilderNode> builderNodes = _builderNodes ?? throw new InvalidOperationException("Trie builder state is not available.");
            int edgeCount = 0;
            for (int i = 0; i < builderNodes.Count; i++)
                edgeCount += builderNodes[i].Children?.Count ?? 0;

            var nodes = new FlatNode[builderNodes.Count];
            var edges = new FlatEdge[edgeCount];

            int edgeIndex = 0;
            for (int i = 0; i < builderNodes.Count; i++)
            {
                BuilderNode builder = builderNodes[i];
                Dictionary<char, int>? children = builder.Children;
                int childCount = children?.Count ?? 0;
                nodes[i] = new FlatNode(edgeIndex, childCount, builder.TokenId, builder.Score);

                if (childCount == 0)
                    continue;

                var sorted = new KeyValuePair<char, int>[childCount];
                int cursor = 0;
                foreach (KeyValuePair<char, int> child in children!)
                    sorted[cursor++] = child;

                Array.Sort(sorted, static (left, right) => left.Key.CompareTo(right.Key));

                for (int c = 0; c < sorted.Length; c++)
                {
                    edges[edgeIndex++] = new FlatEdge(sorted[c].Key, sorted[c].Value);
                }
            }

            _nodes = nodes;
            _edges = edges;
            _builderNodes = null;
            Volatile.Write(ref _isFrozen, 1);
        }
    }

    private int FindChild(int nodeIndex, char c)
    {
        FlatNode node = _nodes[nodeIndex];
        int lo = node.FirstEdgeIndex;
        int hi = lo + node.EdgeCount - 1;

        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            FlatEdge edge = _edges[mid];
            if (edge.Character == c)
                return edge.ChildNodeIndex;

            if (edge.Character < c)
                lo = mid + 1;
            else
                hi = mid - 1;
        }

        return -1;
    }

    private sealed class BuilderNode
    {
        public Dictionary<char, int>? Children;
        public int TokenId = -1;
        public float Score;
    }

    private readonly record struct FlatNode(int FirstEdgeIndex, int EdgeCount, int TokenId, float Score);
    private readonly record struct FlatEdge(char Character, int ChildNodeIndex);
}
