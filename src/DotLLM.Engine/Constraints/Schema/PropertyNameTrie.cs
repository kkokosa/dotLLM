using System.Runtime.CompilerServices;

namespace DotLLM.Engine.Constraints.Schema;

/// <summary>
/// Character trie built from a set of strings (property names or enum values).
/// Used during key/value string generation to restrict valid characters at each position.
/// Immutable after construction. Shared across all constraint clones.
/// </summary>
/// <remarks>
/// Children stored as sorted <c>(char, int)[]</c> for L1 cache locality —
/// typical nodes have 1–5 children, where linear search beats dictionary lookup.
/// </remarks>
internal sealed class PropertyNameTrie
{
    /// <summary>
    /// A trie node with children as a sorted flat array for cache-friendly lookup.
    /// </summary>
    internal readonly record struct TrieNode(
        (char Key, int Child)[] Children,
        bool IsTerminal,
        string? CompleteName);

    private readonly TrieNode[] _nodes;

    /// <summary>Total number of nodes in the trie.</summary>
    public int NodeCount => _nodes.Length;

    /// <summary>
    /// Builds a trie from the given strings.
    /// </summary>
    /// <param name="values">The strings to insert (property names or enum values).</param>
    public PropertyNameTrie(IEnumerable<string> values)
    {
        // Build phase uses Dictionary for convenience, then freezes to sorted arrays.
        var builders = new List<(Dictionary<char, int> Children, bool IsTerminal, string? CompleteName)>
        {
            (new Dictionary<char, int>(), false, null) // root
        };

        foreach (string value in values)
        {
            int current = 0;
            foreach (char c in value)
            {
                if (!builders[current].Children.TryGetValue(c, out int child))
                {
                    child = builders.Count;
                    builders[current].Children[c] = child;
                    builders.Add((new Dictionary<char, int>(), false, null));
                }
                current = child;
            }
            builders[current] = (builders[current].Children, true, value);
        }

        // Freeze: convert dictionaries to sorted flat arrays
        _nodes = new TrieNode[builders.Count];
        for (int i = 0; i < builders.Count; i++)
        {
            var b = builders[i];
            var children = new (char Key, int Child)[b.Children.Count];
            int j = 0;
            foreach (var kvp in b.Children)
                children[j++] = (kvp.Key, kvp.Value);
            Array.Sort(children, (a, b) => a.Key.CompareTo(b.Key));
            _nodes[i] = new TrieNode(children, b.IsTerminal, b.CompleteName);
        }
    }

    /// <summary>
    /// Attempts to advance from <paramref name="nodeIndex"/> by character <paramref name="c"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetChild(int nodeIndex, char c, out int childIndex)
    {
        var children = _nodes[nodeIndex].Children;
        // Linear search — typically 1–5 entries, faster than binary for small N.
        for (int i = 0; i < children.Length; i++)
        {
            if (children[i].Key == c)
            {
                childIndex = children[i].Child;
                return true;
            }
            if (children[i].Key > c)
                break; // sorted — no need to continue
        }
        childIndex = 0;
        return false;
    }

    /// <summary>
    /// Returns all valid next characters at the given trie position.
    /// </summary>
    public IEnumerable<char> GetValidChars(int nodeIndex)
    {
        var children = _nodes[nodeIndex].Children;
        for (int i = 0; i < children.Length; i++)
            yield return children[i].Key;
    }

    /// <summary>
    /// Whether the given node represents a complete string (terminal).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsTerminal(int nodeIndex)
    {
        return _nodes[nodeIndex].IsTerminal;
    }

    /// <summary>
    /// Gets the complete string at a terminal node.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public string? GetCompleteName(int nodeIndex)
    {
        return _nodes[nodeIndex].CompleteName;
    }

    /// <summary>
    /// Whether the given node has any children (can continue matching).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool HasChildren(int nodeIndex)
    {
        return _nodes[nodeIndex].Children.Length > 0;
    }
}
