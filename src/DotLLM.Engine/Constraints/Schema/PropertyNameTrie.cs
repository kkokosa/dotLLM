using System.Runtime.CompilerServices;

namespace DotLLM.Engine.Constraints.Schema;

/// <summary>
/// Character trie built from a set of strings (property names or enum values).
/// Used during key/value string generation to restrict valid characters at each position.
/// Immutable after construction. Shared across all constraint clones.
/// </summary>
internal sealed class PropertyNameTrie
{
    /// <summary>
    /// A trie node with children indexed by character.
    /// </summary>
    internal readonly record struct TrieNode(
        Dictionary<char, int> Children,
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
        var nodes = new List<TrieNode> { new(new Dictionary<char, int>(), false, null) }; // root

        foreach (string value in values)
        {
            int current = 0;
            foreach (char c in value)
            {
                if (!nodes[current].Children.TryGetValue(c, out int child))
                {
                    child = nodes.Count;
                    nodes[current].Children[c] = child;
                    nodes.Add(new TrieNode(new Dictionary<char, int>(), false, null));
                }
                current = child;
            }
            // Mark terminal — replace node to set IsTerminal + CompleteName
            var existing = nodes[current];
            nodes[current] = new TrieNode(existing.Children, true, value);
        }

        _nodes = nodes.ToArray();
    }

    /// <summary>
    /// Attempts to advance from <paramref name="nodeIndex"/> by character <paramref name="c"/>.
    /// </summary>
    /// <param name="nodeIndex">Current trie node index.</param>
    /// <param name="c">The character to advance by.</param>
    /// <param name="childIndex">The child node index if found.</param>
    /// <returns>True if a valid child exists for this character.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetChild(int nodeIndex, char c, out int childIndex)
    {
        return _nodes[nodeIndex].Children.TryGetValue(c, out childIndex);
    }

    /// <summary>
    /// Returns all valid next characters at the given trie position.
    /// </summary>
    /// <param name="nodeIndex">Current trie node index.</param>
    /// <returns>Enumerable of valid characters.</returns>
    public IEnumerable<char> GetValidChars(int nodeIndex)
    {
        return _nodes[nodeIndex].Children.Keys;
    }

    /// <summary>
    /// Whether the given node represents a complete string (terminal).
    /// </summary>
    /// <param name="nodeIndex">Trie node index to check.</param>
    /// <returns>True if this is a terminal node.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsTerminal(int nodeIndex)
    {
        return _nodes[nodeIndex].IsTerminal;
    }

    /// <summary>
    /// Gets the complete string at a terminal node.
    /// </summary>
    /// <param name="nodeIndex">Trie node index (must be terminal).</param>
    /// <returns>The complete string, or null if not terminal.</returns>
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
        return _nodes[nodeIndex].Children.Count > 0;
    }
}
