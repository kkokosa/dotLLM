using System.Collections.Frozen;
using System.Text.Json;

namespace DotLLM.Engine.Constraints.Schema;

/// <summary>
/// Compiles a JSON Schema (as <see cref="JsonElement"/>) into a <see cref="CompiledSchema"/>.
/// Resolves <c>$ref</c> / <c>$defs</c> at compile time. Non-recursive references only.
/// </summary>
internal static class SchemaCompiler
{
    /// <summary>
    /// Compiles a JSON Schema from its JSON string representation.
    /// </summary>
    /// <param name="schemaJson">The JSON Schema as a string.</param>
    /// <returns>The compiled schema.</returns>
    /// <exception cref="ArgumentException">Thrown for unsupported or invalid schema constructs.</exception>
    public static CompiledSchema Compile(string schemaJson)
    {
        using var doc = JsonDocument.Parse(schemaJson);
        return Compile(doc.RootElement);
    }

    /// <summary>
    /// Compiles a JSON Schema from a parsed <see cref="JsonElement"/>.
    /// </summary>
    public static CompiledSchema Compile(JsonElement root)
    {
        var ctx = new CompilationContext();

        // Extract $defs for reference resolution
        if (root.TryGetProperty("$defs", out var defs))
            ctx.Defs = defs;

        CompileNode(root, ctx, new HashSet<string>());

        return new CompiledSchema(ctx.Nodes.ToArray(), ctx.Tries.ToArray());
    }

    /// <summary>
    /// Compiles a single schema node and returns its index in the node array.
    /// </summary>
    private static int CompileNode(JsonElement element, CompilationContext ctx, HashSet<string> refStack)
    {
        // Handle $ref
        if (element.TryGetProperty("$ref", out var refProp))
        {
            string refPath = refProp.GetString()!;
            if (!refStack.Add(refPath))
                throw new ArgumentException($"Recursive $ref detected: {refPath}. Recursive schemas are not supported.");

            var resolved = ResolveRef(refPath, ctx);
            int result = CompileNode(resolved, ctx, refStack);
            refStack.Remove(refPath);
            return result;
        }

        // Reserve index for this node
        int nodeIndex = ctx.Nodes.Count;
        ctx.Nodes.Add(default); // placeholder

        var allowedTypes = JsonSchemaType.None;
        FrozenDictionary<string, int>? properties = null;
        string[]? propertyNames = null;
        ulong requiredBitmask = 0;
        bool additionalPropertiesForbidden = false;
        int itemsNodeIndex = -1;
        string[]? enumValues = null;
        string? constValue = null;
        int[]? anyOfNodeIndices = null;
        int propertyTrieIndex = -1;
        int enumTrieIndex = -1;

        // Parse type
        if (element.TryGetProperty("type", out var typeProp))
        {
            if (typeProp.ValueKind == JsonValueKind.String)
            {
                allowedTypes = ParseType(typeProp.GetString()!);
            }
            else if (typeProp.ValueKind == JsonValueKind.Array)
            {
                foreach (var t in typeProp.EnumerateArray())
                    allowedTypes |= ParseType(t.GetString()!);
            }
        }

        // Parse anyOf
        if (element.TryGetProperty("anyOf", out var anyOfProp))
        {
            var indices = new List<int>();
            foreach (var alt in anyOfProp.EnumerateArray())
            {
                int altIndex = CompileNode(alt, ctx, refStack);
                indices.Add(altIndex);
                allowedTypes |= ctx.Nodes[altIndex].AllowedTypes;
            }
            anyOfNodeIndices = indices.ToArray();
        }

        // If no type and no anyOf, infer from structure or allow all
        if (allowedTypes == JsonSchemaType.None && anyOfNodeIndices == null)
        {
            if (element.TryGetProperty("properties", out _))
                allowedTypes = JsonSchemaType.Object;
            else if (element.TryGetProperty("items", out _))
                allowedTypes = JsonSchemaType.Array;
            else if (element.TryGetProperty("enum", out _) || element.TryGetProperty("const", out _))
                allowedTypes = InferEnumTypes(element);
            else
                allowedTypes = JsonSchemaType.Object | JsonSchemaType.Array | JsonSchemaType.String |
                               JsonSchemaType.Number | JsonSchemaType.Integer | JsonSchemaType.Boolean |
                               JsonSchemaType.Null;
        }

        // Parse properties (object)
        if (element.TryGetProperty("properties", out var propsProp))
        {
            var propDict = new Dictionary<string, int>();
            var names = new List<string>();

            foreach (var prop in propsProp.EnumerateObject())
            {
                names.Add(prop.Name);
                int childIndex = CompileNode(prop.Value, ctx, refStack);
                propDict[prop.Name] = childIndex;
            }

            properties = propDict.ToFrozenDictionary();
            propertyNames = names.ToArray();

            // Build property name trie
            propertyTrieIndex = ctx.Tries.Count;
            ctx.Tries.Add(new PropertyNameTrie(names));
        }

        // Parse required — add unconstrained nodes for required keys absent from properties
        if (element.TryGetProperty("required", out var reqProp))
        {
            // Ensure we have a mutable property tracking even if no "properties" block
            var propDict = properties != null
                ? new Dictionary<string, int>(properties)
                : new Dictionary<string, int>();
            var names = propertyNames != null
                ? new List<string>(propertyNames)
                : new List<string>();
            bool modified = false;

            foreach (var req in reqProp.EnumerateArray())
            {
                string reqName = req.GetString()!;
                int bitPos = names.IndexOf(reqName);

                // Required key not in properties — add unconstrained node so the
                // bitmask covers it and the trie allows the key during generation.
                if (bitPos < 0)
                {
                    bitPos = names.Count;
                    names.Add(reqName);
                    int unconstrainedIdx = ctx.Nodes.Count;
                    ctx.Nodes.Add(SchemaNode.Unconstrained);
                    propDict[reqName] = unconstrainedIdx;
                    modified = true;
                }

                if (bitPos < 64)
                    requiredBitmask |= 1UL << bitPos;
            }

            if (modified)
            {
                properties = propDict.ToFrozenDictionary();
                propertyNames = names.ToArray();
                // Rebuild property name trie with the added keys
                if (propertyTrieIndex >= 0)
                    ctx.Tries[propertyTrieIndex] = new PropertyNameTrie(names);
                else
                {
                    propertyTrieIndex = ctx.Tries.Count;
                    ctx.Tries.Add(new PropertyNameTrie(names));
                }
                if (allowedTypes == JsonSchemaType.None)
                    allowedTypes = JsonSchemaType.Object;
            }
        }

        // Parse additionalProperties
        if (element.TryGetProperty("additionalProperties", out var addlProp))
        {
            if (addlProp.ValueKind == JsonValueKind.False)
                additionalPropertiesForbidden = true;
        }

        // Parse items (array)
        if (element.TryGetProperty("items", out var itemsProp))
        {
            itemsNodeIndex = CompileNode(itemsProp, ctx, refStack);
        }

        // Parse enum
        if (element.TryGetProperty("enum", out var enumProp))
        {
            var values = new List<string>();
            foreach (var val in enumProp.EnumerateArray())
            {
                values.Add(JsonValueToString(val));
            }
            enumValues = values.ToArray();

            // Build enum value trie for string enums (the raw string values, not JSON-encoded)
            if (allowedTypes.HasFlag(JsonSchemaType.String))
            {
                var stringValues = new List<string>();
                foreach (var val in enumProp.EnumerateArray())
                {
                    if (val.ValueKind == JsonValueKind.String)
                        stringValues.Add(val.GetString()!);
                }
                if (stringValues.Count > 0)
                {
                    enumTrieIndex = ctx.Tries.Count;
                    ctx.Tries.Add(new PropertyNameTrie(stringValues));
                }
            }
        }

        // Parse const
        if (element.TryGetProperty("const", out var constProp))
        {
            constValue = JsonValueToString(constProp);

            // Build a single-entry trie for string const
            if (constProp.ValueKind == JsonValueKind.String)
            {
                enumTrieIndex = ctx.Tries.Count;
                ctx.Tries.Add(new PropertyNameTrie([constProp.GetString()!]));
            }
        }

        // Build final node
        ctx.Nodes[nodeIndex] = new SchemaNode
        {
            AllowedTypes = allowedTypes,
            Properties = properties,
            PropertyNames = propertyNames,
            RequiredBitmask = requiredBitmask,
            AdditionalPropertiesForbidden = additionalPropertiesForbidden,
            ItemsNodeIndex = itemsNodeIndex,
            EnumValues = enumValues,
            ConstValue = constValue,
            AnyOfNodeIndices = anyOfNodeIndices,
            PropertyTrieIndex = propertyTrieIndex,
            EnumTrieIndex = enumTrieIndex,
        };

        return nodeIndex;
    }

    private static JsonSchemaType ParseType(string type) => type switch
    {
        "object" => JsonSchemaType.Object,
        "array" => JsonSchemaType.Array,
        "string" => JsonSchemaType.String,
        "number" => JsonSchemaType.Number,
        "integer" => JsonSchemaType.Integer,
        "boolean" => JsonSchemaType.Boolean,
        "null" => JsonSchemaType.Null,
        _ => throw new ArgumentException($"Unknown JSON Schema type: {type}")
    };

    private static JsonSchemaType InferEnumTypes(JsonElement element)
    {
        var types = JsonSchemaType.None;

        JsonElement values;
        if (element.TryGetProperty("enum", out values))
        {
            foreach (var val in values.EnumerateArray())
                types |= InferValueType(val);
        }
        else if (element.TryGetProperty("const", out var constVal))
        {
            types = InferValueType(constVal);
        }

        return types == JsonSchemaType.None
            ? JsonSchemaType.Object | JsonSchemaType.Array | JsonSchemaType.String |
              JsonSchemaType.Number | JsonSchemaType.Integer | JsonSchemaType.Boolean |
              JsonSchemaType.Null
            : types;
    }

    private static JsonSchemaType InferValueType(JsonElement val) => val.ValueKind switch
    {
        JsonValueKind.String => JsonSchemaType.String,
        JsonValueKind.Number => val.TryGetInt64(out _) ? JsonSchemaType.Integer : JsonSchemaType.Number,
        JsonValueKind.True or JsonValueKind.False => JsonSchemaType.Boolean,
        JsonValueKind.Null => JsonSchemaType.Null,
        JsonValueKind.Object => JsonSchemaType.Object,
        JsonValueKind.Array => JsonSchemaType.Array,
        _ => JsonSchemaType.None,
    };

    private static string JsonValueToString(JsonElement val) => val.ValueKind switch
    {
        JsonValueKind.String => val.GetString()!,
        _ => val.GetRawText(),
    };

    private static JsonElement ResolveRef(string refPath, CompilationContext ctx)
    {
        // Support #/$defs/Name format
        const string defsPrefix = "#/$defs/";
        if (!refPath.StartsWith(defsPrefix))
            throw new ArgumentException($"Unsupported $ref format: {refPath}. Only #/$defs/Name is supported.");

        string defName = refPath[defsPrefix.Length..];

        if (ctx.Defs.ValueKind != JsonValueKind.Object || !ctx.Defs.TryGetProperty(defName, out var def))
            throw new ArgumentException($"$ref target not found: {refPath}");

        return def;
    }

    private sealed class CompilationContext
    {
        public List<SchemaNode> Nodes { get; } = [];
        public List<PropertyNameTrie> Tries { get; } = [];
        public JsonElement Defs { get; set; }
    }
}
