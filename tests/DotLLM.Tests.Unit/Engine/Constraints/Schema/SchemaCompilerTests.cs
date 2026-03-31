using DotLLM.Engine.Constraints.Schema;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints.Schema;

public class SchemaCompilerTests
{
    [Fact]
    public void Compile_SimpleObject_ProducesCorrectRoot()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                },
                "required": ["name"]
            }
            """);

        Assert.NotNull(schema.Nodes);
        Assert.True(schema.Nodes.Length >= 3); // root + 2 properties

        ref readonly var root = ref schema.Nodes[0];
        Assert.True(root.AllowedTypes.HasFlag(JsonSchemaType.Object));
        Assert.NotNull(root.Properties);
        Assert.Equal(2, root.Properties.Count);
        Assert.True(root.Properties.ContainsKey("name"));
        Assert.True(root.Properties.ContainsKey("age"));
        Assert.NotNull(root.PropertyNames);
        Assert.Equal(2, root.PropertyNames.Length);
    }

    [Fact]
    public void Compile_RequiredProperties_SetsCorrectBitmask()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": {
                    "a": { "type": "string" },
                    "b": { "type": "number" },
                    "c": { "type": "boolean" }
                },
                "required": ["a", "c"]
            }
            """);

        ref readonly var root = ref schema.Nodes[0];
        // a is at index 0, c is at index 2
        int aPos = Array.IndexOf(root.PropertyNames!, "a");
        int cPos = Array.IndexOf(root.PropertyNames!, "c");
        Assert.True((root.RequiredBitmask & (1UL << aPos)) != 0, "'a' should be required");
        Assert.True((root.RequiredBitmask & (1UL << cPos)) != 0, "'c' should be required");

        int bPos = Array.IndexOf(root.PropertyNames!, "b");
        Assert.True((root.RequiredBitmask & (1UL << bPos)) == 0, "'b' should not be required");
    }

    [Fact]
    public void Compile_AdditionalPropertiesFalse_SetsFlag()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": { "x": { "type": "string" } },
                "additionalProperties": false
            }
            """);

        Assert.True(schema.Nodes[0].AdditionalPropertiesForbidden);
    }

    [Fact]
    public void Compile_NestedObjects_ProducesCorrectGraph()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "zip": { "type": "string" }
                        }
                    }
                }
            }
            """);

        ref readonly var root = ref schema.Nodes[0];
        Assert.NotNull(root.Properties);
        int addressIdx = root.Properties["address"];

        ref readonly var address = ref schema.Nodes[addressIdx];
        Assert.True(address.AllowedTypes.HasFlag(JsonSchemaType.Object));
        Assert.NotNull(address.Properties);
        Assert.True(address.Properties.ContainsKey("city"));
        Assert.True(address.Properties.ContainsKey("zip"));
    }

    [Fact]
    public void Compile_ArrayWithItems_SetsItemsNodeIndex()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                }
            }
            """);

        ref readonly var root = ref schema.Nodes[0];
        int tagsIdx = root.Properties!["tags"];
        ref readonly var tags = ref schema.Nodes[tagsIdx];

        Assert.True(tags.AllowedTypes.HasFlag(JsonSchemaType.Array));
        Assert.True(tags.ItemsNodeIndex >= 0);

        ref readonly var items = ref schema.Nodes[tags.ItemsNodeIndex];
        Assert.True(items.AllowedTypes.HasFlag(JsonSchemaType.String));
    }

    [Fact]
    public void Compile_StringEnum_BuildsEnumTrie()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "enum": ["red", "green", "blue"]
                    }
                }
            }
            """);

        ref readonly var root = ref schema.Nodes[0];
        int colorIdx = root.Properties!["color"];
        ref readonly var color = ref schema.Nodes[colorIdx];

        Assert.NotNull(color.EnumValues);
        Assert.Equal(3, color.EnumValues.Length);
        Assert.True(color.EnumTrieIndex >= 0, "Enum trie should be built");
    }

    [Fact]
    public void Compile_ConstValue_SetsConst()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": {
                    "version": { "const": "1.0" }
                }
            }
            """);

        ref readonly var root = ref schema.Nodes[0];
        int versionIdx = root.Properties!["version"];
        ref readonly var version = ref schema.Nodes[versionIdx];

        Assert.Equal("1.0", version.ConstValue);
        Assert.True(version.EnumTrieIndex >= 0, "Const string should have enum trie");
    }

    [Fact]
    public void Compile_Ref_ResolvesFromDefs()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": {
                    "item": { "$ref": "#/$defs/MyItem" }
                },
                "$defs": {
                    "MyItem": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "integer" }
                        }
                    }
                }
            }
            """);

        ref readonly var root = ref schema.Nodes[0];
        int itemIdx = root.Properties!["item"];
        ref readonly var item = ref schema.Nodes[itemIdx];

        Assert.True(item.AllowedTypes.HasFlag(JsonSchemaType.Object));
        Assert.NotNull(item.Properties);
        Assert.True(item.Properties.ContainsKey("id"));
    }

    [Fact]
    public void Compile_RecursiveRef_Throws()
    {
        Assert.Throws<ArgumentException>(() => SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": {
                    "child": { "$ref": "#/$defs/Node" }
                },
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "child": { "$ref": "#/$defs/Node" }
                        }
                    }
                }
            }
            """));
    }

    [Fact]
    public void Compile_AnyOf_MergesTypes()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": {
                    "value": {
                        "anyOf": [
                            { "type": "string" },
                            { "type": "integer" }
                        ]
                    }
                }
            }
            """);

        ref readonly var root = ref schema.Nodes[0];
        int valueIdx = root.Properties!["value"];
        ref readonly var value = ref schema.Nodes[valueIdx];

        Assert.True(value.AllowedTypes.HasFlag(JsonSchemaType.String));
        Assert.True(value.AllowedTypes.HasFlag(JsonSchemaType.Integer));
        Assert.NotNull(value.AnyOfNodeIndices);
        Assert.Equal(2, value.AnyOfNodeIndices.Length);
    }

    [Fact]
    public void Compile_IntegerType_SetsIntegerFlag()
    {
        var schema = SchemaCompiler.Compile("""{ "type": "integer" }""");
        Assert.True(schema.Nodes[0].AllowedTypes.HasFlag(JsonSchemaType.Integer));
    }

    [Fact]
    public void Compile_PropertyNameTrie_IsBuilt()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "namespace": { "type": "string" }
                }
            }
            """);

        ref readonly var root = ref schema.Nodes[0];
        Assert.True(root.PropertyTrieIndex >= 0);

        var trie = schema.PropertyTries[root.PropertyTrieIndex];
        Assert.True(trie.TryGetChild(0, 'n', out int nNode));
        Assert.True(trie.TryGetChild(nNode, 'a', out _));
        Assert.False(trie.TryGetChild(0, 'x', out _));
    }

    [Fact]
    public void Compile_EmptyObject_NoRequired()
    {
        var schema = SchemaCompiler.Compile("""{ "type": "object" }""");
        ref readonly var root = ref schema.Nodes[0];

        Assert.True(root.AllowedTypes.HasFlag(JsonSchemaType.Object));
        Assert.Null(root.Properties);
        Assert.Equal(0UL, root.RequiredBitmask);
    }

    [Fact]
    public void Compile_TypeInferredFromProperties()
    {
        var schema = SchemaCompiler.Compile("""
            {
                "properties": {
                    "x": { "type": "string" }
                }
            }
            """);

        Assert.True(schema.Nodes[0].AllowedTypes.HasFlag(JsonSchemaType.Object));
    }
}
