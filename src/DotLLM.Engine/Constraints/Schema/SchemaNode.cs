using System.Collections.Frozen;

namespace DotLLM.Engine.Constraints.Schema;

/// <summary>
/// Allowed JSON Schema types, stored as flags for union types (anyOf).
/// </summary>
[Flags]
internal enum JsonSchemaType : byte
{
    None    = 0,
    Object  = 1 << 0,
    Array   = 1 << 1,
    String  = 1 << 2,
    Number  = 1 << 3,
    Integer = 1 << 4,
    Boolean = 1 << 5,
    Null    = 1 << 6,
}

/// <summary>
/// A compiled schema node in the flat node array. Immutable after compilation.
/// References to sub-schemas use indices into <see cref="CompiledSchema.Nodes"/>.
/// </summary>
internal readonly record struct SchemaNode
{
    /// <summary>Allowed types at this position.</summary>
    public JsonSchemaType AllowedTypes { get; init; }

    /// <summary>
    /// For Object: property name → child node index in <see cref="CompiledSchema.Nodes"/>.
    /// Null for non-object nodes.
    /// </summary>
    public FrozenDictionary<string, int>? Properties { get; init; }

    /// <summary>
    /// For Object: ordered list of property names (order matches bitmask bit positions).
    /// Null for non-object nodes.
    /// </summary>
    public string[]? PropertyNames { get; init; }

    /// <summary>
    /// For Object: bitmask of required properties (bit positions match <see cref="PropertyNames"/> order).
    /// </summary>
    public ulong RequiredBitmask { get; init; }

    /// <summary>
    /// For Object: whether additional properties beyond those in <see cref="Properties"/> are forbidden.
    /// </summary>
    public bool AdditionalPropertiesForbidden { get; init; }

    /// <summary>
    /// For Array: node index of the items schema in <see cref="CompiledSchema.Nodes"/>. -1 if unconstrained.
    /// </summary>
    public int ItemsNodeIndex { get; init; }

    /// <summary>
    /// For String enum: the allowed values. Null if unconstrained.
    /// </summary>
    public string[]? EnumValues { get; init; }

    /// <summary>
    /// For const: the JSON-encoded constant value. Null if unconstrained.
    /// </summary>
    public string? ConstValue { get; init; }

    /// <summary>
    /// For anyOf: indices of alternative schema nodes. Null if not a union.
    /// </summary>
    public int[]? AnyOfNodeIndices { get; init; }

    /// <summary>
    /// Index of the <see cref="PropertyNameTrie"/> in <see cref="CompiledSchema.PropertyTries"/>.
    /// -1 if not an object or no property constraints.
    /// </summary>
    public int PropertyTrieIndex { get; init; }

    /// <summary>
    /// Index of the enum/const value trie in <see cref="CompiledSchema.PropertyTries"/>.
    /// -1 if not an enum/const string node.
    /// </summary>
    public int EnumTrieIndex { get; init; }

    /// <summary>Creates a default unconstrained node.</summary>
    public static SchemaNode Unconstrained => new()
    {
        AllowedTypes = JsonSchemaType.Object | JsonSchemaType.Array | JsonSchemaType.String |
                       JsonSchemaType.Number | JsonSchemaType.Integer | JsonSchemaType.Boolean |
                       JsonSchemaType.Null,
        ItemsNodeIndex = -1,
        PropertyTrieIndex = -1,
        EnumTrieIndex = -1,
    };
}
