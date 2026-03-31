namespace DotLLM.Engine.Constraints.Schema;

/// <summary>
/// Immutable compiled representation of a JSON Schema. Contains a flat array of
/// <see cref="SchemaNode"/> values and associated <see cref="PropertyNameTrie"/> instances.
/// Thread-safe: shared across all constraint clones without synchronization.
/// </summary>
internal sealed class CompiledSchema
{
    /// <summary>Flat array of schema nodes. Index 0 is the root.</summary>
    public SchemaNode[] Nodes { get; }

    /// <summary>
    /// Property name and enum value tries. Referenced by <see cref="SchemaNode.PropertyTrieIndex"/>
    /// and <see cref="SchemaNode.EnumTrieIndex"/>.
    /// </summary>
    public PropertyNameTrie[] PropertyTries { get; }

    /// <summary>
    /// Creates a compiled schema from pre-built node and trie arrays.
    /// </summary>
    /// <param name="nodes">Flat array of schema nodes (index 0 = root).</param>
    /// <param name="tries">Property name and enum value tries.</param>
    internal CompiledSchema(SchemaNode[] nodes, PropertyNameTrie[] tries)
    {
        Nodes = nodes;
        PropertyTries = tries;
    }
}
