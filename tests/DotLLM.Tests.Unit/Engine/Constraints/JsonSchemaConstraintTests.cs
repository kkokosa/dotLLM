using DotLLM.Core.Constraints;
using DotLLM.Engine.Constraints;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints;

public class JsonSchemaConstraintTests
{
    /// <summary>
    /// Richer tokenizer for schema tests. Includes multi-char tokens for property names.
    /// </summary>
    private sealed class SchemaStubTokenizer : ITokenizer
    {
        // Token map: 0='{', 1='}', 2='"', 3=':', 4=',', 5='n', 6='a', 7='m', 8='e',
        // 9='g', 10='1', 11='[', 12=']', 13='t', 14='r', 15='u', 16='f', 17='l',
        // 18='s', 19=' ', 20=EOS, 21='"name"', 22='"age"', 23='-', 24='0',
        // 25='.', 26='2', 27='d', 28='v', 29='c', 30='o', 31='i', 32='b', 33='p'
        private static readonly string[] Tokens =
            ["{", "}", "\"", ":", ",", "n", "a", "m", "e",
             "g", "1", "[", "]", "t", "r", "u", "f", "l",
             "s", " ", "<eos>", "\"name\"", "\"age\"", "-", "0",
             ".", "2", "d", "v", "c", "o", "i", "b", "p"];

        public int VocabSize => Tokens.Length;
        public int BosTokenId => 0;
        public int EosTokenId => 20;

        public string DecodeToken(int tokenId) =>
            tokenId >= 0 && tokenId < Tokens.Length ? Tokens[tokenId] : "";

        public int[] Encode(string text) => throw new NotImplementedException();
        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotImplementedException();
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => throw new NotImplementedException();
        public int CountTokens(string text) => throw new NotImplementedException();
    }

    private static JsonSchemaConstraint CreateConstraint(string schemaJson)
        => new(new SchemaStubTokenizer(), schemaJson);

    [Fact]
    public void AtStart_ObjectSchema_AllowsOnlyOpenBrace()
    {
        var constraint = CreateConstraint("""
            { "type": "object", "properties": { "name": { "type": "string" } } }
            """);

        var mask = constraint.GetAllowedTokens();

        // '{' (0) should be allowed
        Assert.True(mask.IsAllowed(0), "'{' should be allowed at start for object schema");

        // '[' (11) should NOT be allowed — schema type is object only
        Assert.False(mask.IsAllowed(11), "'[' should not be allowed for object schema");

        // EOS should not be allowed
        Assert.False(mask.IsAllowed(20), "EOS not allowed at start");
    }

    [Fact]
    public void AtStart_ArraySchema_AllowsOnlyOpenBracket()
    {
        var constraint = CreateConstraint("""
            { "type": "array", "items": { "type": "string" } }
            """);

        var mask = constraint.GetAllowedTokens();

        Assert.True(mask.IsAllowed(11), "'[' should be allowed for array schema");
        Assert.False(mask.IsAllowed(0), "'{' should not be allowed for array schema");
    }

    [Fact]
    public void RequiredProperties_PreventPrematureClose()
    {
        var constraint = CreateConstraint("""
            {
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }
            """);

        constraint.Advance(0); // '{'

        var mask = constraint.GetAllowedTokens();
        // '}' (1) should NOT be allowed — "name" is required
        Assert.False(mask.IsAllowed(1), "'}' should not be allowed when required property 'name' is missing");
        // '"' (2) for key string should be allowed
        Assert.True(mask.IsAllowed(2), "'\"' should be allowed to start property key");
    }

    [Fact]
    public void RequiredProperties_AllowCloseWhenSatisfied()
    {
        var constraint = CreateConstraint("""
            {
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }
            """);

        // Build {"name":"e"}
        AdvanceString(constraint, "{\"name\":\"e\"}");

        Assert.True(constraint.IsComplete(), "Should be complete after required property emitted");
    }

    [Fact]
    public void TypeRestriction_IntegerOnly_RejectsDot()
    {
        var constraint = CreateConstraint("""
            {
                "type": "object",
                "properties": { "age": { "type": "integer" } },
                "required": ["age"]
            }
            """);

        // {"age":
        AdvanceString(constraint, "{\"age\":");

        var mask = constraint.GetAllowedTokens();
        // Should allow digits and minus, not '"', '{', '['
        Assert.True(mask.IsAllowed(10), "'1' should be allowed for integer");
        Assert.True(mask.IsAllowed(23), "'-' should be allowed for integer");
        Assert.False(mask.IsAllowed(2), "'\"' should not be allowed for integer type");

        // Advance a digit
        constraint.Advance(10); // '1'

        // Now '.' should NOT be allowed (integer type)
        mask = constraint.GetAllowedTokens();
        Assert.False(mask.IsAllowed(25), "'.' should not be allowed for integer type");
    }

    [Fact]
    public void EmptyObject_NoRequired_CanCloseImmediately()
    {
        var constraint = CreateConstraint("""{ "type": "object" }""");

        constraint.Advance(0); // '{'

        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(1), "'}' should be allowed for object with no required properties");
    }

    [Fact]
    public void IsComplete_ReturnsTrueAfterValidComplete()
    {
        var constraint = CreateConstraint("""
            { "type": "object", "properties": { "name": { "type": "string" } } }
            """);

        AdvanceString(constraint, "{\"name\":\"e\"}");

        Assert.True(constraint.IsComplete());
    }

    [Fact]
    public void IsComplete_ReturnsFalseBeforeDone()
    {
        var constraint = CreateConstraint("""{ "type": "object" }""");

        constraint.Advance(0); // '{'
        Assert.False(constraint.IsComplete());
    }

    [Fact]
    public void EosAllowed_OnlyWhenComplete()
    {
        var constraint = CreateConstraint("""{ "type": "object" }""");

        // Before any tokens: no EOS
        Assert.False(constraint.GetAllowedTokens().IsAllowed(20));

        // After {}: EOS should be allowed
        constraint.Advance(0); // '{'
        constraint.Advance(1); // '}'

        Assert.True(constraint.IsComplete());
        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(20), "EOS should be allowed when schema is satisfied");
    }

    [Fact]
    public void Clone_ProducesIndependentState()
    {
        var constraint = CreateConstraint("""{ "type": "object" }""");
        constraint.Advance(0); // '{'

        var clone = (JsonSchemaConstraint)constraint.Clone();

        constraint.Advance(1); // '}'
        Assert.True(constraint.IsComplete());
        Assert.False(clone.IsComplete(), "Clone should be independent");

        clone.Advance(1); // '}'
        Assert.True(clone.IsComplete());
    }

    [Fact]
    public void Reset_ReturnsToInitialState()
    {
        var constraint = CreateConstraint("""{ "type": "object" }""");
        constraint.Advance(0); // '{'
        constraint.Advance(1); // '}'
        Assert.True(constraint.IsComplete());

        constraint.Reset();
        Assert.False(constraint.IsComplete());

        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(0), "'{' should be allowed after reset");
    }

    [Fact]
    public void MaskCaching_SameState_ReturnsCachedMask()
    {
        var constraint = CreateConstraint("""{ "type": "object" }""");

        var mask1 = constraint.GetAllowedTokens();
        var mask2 = constraint.GetAllowedTokens();

        // Both should be structurally identical
        for (int i = 0; i < new SchemaStubTokenizer().VocabSize; i++)
            Assert.Equal(mask1.IsAllowed(i), mask2.IsAllowed(i));
    }

    [Fact]
    public void BooleanType_AllowsTrueAndFalse()
    {
        var constraint = CreateConstraint("""
            {
                "type": "object",
                "properties": { "flag": { "type": "boolean" } },
                "required": ["flag"]
            }
            """);

        AdvanceString(constraint, "{\"flag\":");

        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(13), "'t' should be allowed for boolean (true)");
        Assert.True(mask.IsAllowed(16), "'f' should be allowed for boolean (false)");
        Assert.False(mask.IsAllowed(10), "'1' should not be allowed for boolean");
        Assert.False(mask.IsAllowed(2), "'\"' should not be allowed for boolean");
    }

    [Fact]
    public void NullType_AllowsN()
    {
        var constraint = CreateConstraint("""
            {
                "type": "object",
                "properties": { "val": { "type": "null" } },
                "required": ["val"]
            }
            """);

        AdvanceString(constraint, "{\"val\":");

        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(5), "'n' should be allowed for null type");
        Assert.False(mask.IsAllowed(13), "'t' should not be allowed for null type");
    }

    [Fact]
    public void NestedObject_TracksCorrectly()
    {
        var constraint = CreateConstraint("""
            {
                "type": "object",
                "properties": {
                    "addr": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" }
                        }
                    }
                }
            }
            """);

        // {"addr":{"name":"e"}}
        AdvanceString(constraint, "{\"addr\":{\"name\":\"e\"}}");
        Assert.True(constraint.IsComplete());
    }

    [Fact]
    public void ArrayWithStringItems_EnforcesItemType()
    {
        var constraint = CreateConstraint("""
            {
                "type": "array",
                "items": { "type": "string" }
            }
            """);

        constraint.Advance(11); // '['

        var mask = constraint.GetAllowedTokens();
        // Should allow '"' for string items and ']' for empty array
        Assert.True(mask.IsAllowed(2), "'\"' should be allowed for string array items");
        Assert.True(mask.IsAllowed(12), "']' should be allowed for empty array");
        // Should NOT allow '{', '[', digits for string-only items
        Assert.False(mask.IsAllowed(0), "'{' should not be allowed for string items");
        Assert.False(mask.IsAllowed(10), "'1' should not be allowed for string items");
    }

    [Fact]
    public void AnyOf_AllowsUnionTypes()
    {
        var constraint = CreateConstraint("""
            {
                "type": "object",
                "properties": {
                    "value": {
                        "anyOf": [
                            { "type": "string" },
                            { "type": "integer" }
                        ]
                    }
                },
                "required": ["value"]
            }
            """);

        AdvanceString(constraint, "{\"value\":");

        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(2), "'\"' should be allowed (string branch)");
        Assert.True(mask.IsAllowed(10), "'1' should be allowed (integer branch)");
        Assert.False(mask.IsAllowed(0), "'{' should not be allowed (neither branch is object)");
    }

    [Fact]
    public void MultipleProperties_SecondPropertyTypeEnforced()
    {
        // Regression: after first property value completes, _currentNodeIndex must
        // restore to parent object so the second key lookup works correctly.
        var constraint = CreateConstraint("""
            {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                },
                "required": ["name", "age"]
            }
            """);

        // {"name":"e",
        AdvanceString(constraint, "{\"name\":\"e\",");

        // At ObjectNextKey — should still allow key string
        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(2), "'\"' should be allowed to start next key");

        // "age":
        AdvanceString(constraint, "\"age\":");

        // Now at ValueStart for "age" which is integer type
        mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(10), "'1' should be allowed for integer");
        Assert.False(mask.IsAllowed(2), "'\"' should NOT be allowed — age is integer, not string");
    }

    [Fact]
    public void StringEnum_RestrictsToEnumValues()
    {
        var constraint = CreateConstraint("""
            {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "enum": ["red", "green"]
                    }
                },
                "required": ["color"]
            }
            """);

        // Navigate to value string position
        AdvanceString(constraint, "{\"color\":\"");

        var mask = constraint.GetAllowedTokens();
        // 'r' (14) and 'g' (9) should be allowed (first chars of "red" and "green")
        Assert.True(mask.IsAllowed(14), "'r' should be allowed (start of 'red')");
        Assert.True(mask.IsAllowed(9), "'g' should be allowed (start of 'green')");
        // Other chars like 'n' should NOT be allowed
        Assert.False(mask.IsAllowed(5), "'n' should not be allowed (not an enum prefix)");
    }

    // -------------------------------------------------------------------------
    // LRU cache eviction
    // -------------------------------------------------------------------------

    [Fact]
    public void MaskCache_LruEvicts_OldestEntryOnOverflow()
    {
        // Cache capacity = 2. Walk through 3 distinct schema states, then return to state 1.
        // If the cache were LRU-correct, state 1 should still be hot after 2 intermediate
        // lookups; if it cleared on overflow (old behavior), all state would be gone.
        var tokenizer = new SchemaStubTokenizer();
        var constraint = new JsonSchemaConstraint(tokenizer, """
            { "type": "object", "properties": { "name": { "type": "string" } } }
            """, maxCacheEntries: 2);

        // State A: at-start — mask includes '{'
        var maskA1 = constraint.GetAllowedTokens();
        Assert.True(maskA1.IsAllowed(0));

        // State B: after '{' — mask includes '"'
        constraint.Advance(0);
        var maskB1 = constraint.GetAllowedTokens();
        Assert.True(maskB1.IsAllowed(2));

        // State C: after '{"' — mask no longer includes '{'
        constraint.Advance(2);
        var maskC1 = constraint.GetAllowedTokens();
        // This third insertion should evict state A (LRU), not state B (MRU).

        // Walk a second constraint back to state B to verify its mask is still computable.
        // (We cannot probe cache contents directly; this test mainly asserts no crash and
        // that mask contents remain schema-correct after eviction has occurred.)
        var constraint2 = new JsonSchemaConstraint(tokenizer, """
            { "type": "object", "properties": { "name": { "type": "string" } } }
            """, maxCacheEntries: 2);
        constraint2.Advance(0);
        var maskB2 = constraint2.GetAllowedTokens();

        // Masks for structurally-equivalent states must match bit-for-bit.
        for (int i = 0; i < tokenizer.VocabSize; i++)
            Assert.Equal(maskB1.IsAllowed(i), maskB2.IsAllowed(i));
    }

    [Fact]
    public void MaskCache_UnderCapacity_Reuses()
    {
        // Repeated GetAllowedTokens() at the same state should return the cached mask
        // (not rebuild). Only observable via correctness — two calls at the same state
        // must produce identical mask contents.
        var constraint = CreateConstraint("""{ "type": "object" }""");

        var mask1 = constraint.GetAllowedTokens();
        var mask2 = constraint.GetAllowedTokens();
        var mask3 = constraint.GetAllowedTokens();

        var tokenizer = new SchemaStubTokenizer();
        for (int i = 0; i < tokenizer.VocabSize; i++)
        {
            Assert.Equal(mask1.IsAllowed(i), mask2.IsAllowed(i));
            Assert.Equal(mask1.IsAllowed(i), mask3.IsAllowed(i));
        }
    }

    /// <summary>
    /// Helper: advances the constraint character-by-character by finding single-char tokens.
    /// </summary>
    private static void AdvanceString(JsonSchemaConstraint constraint, string text)
    {
        var tokenizer = new SchemaStubTokenizer();
        foreach (char c in text)
        {
            int tokenId = FindSingleCharToken(tokenizer, c);
            Assert.True(tokenId >= 0, $"No single-char token found for '{c}'");
            constraint.Advance(tokenId);
        }
    }

    private static int FindSingleCharToken(SchemaStubTokenizer tokenizer, char c)
    {
        string target = c.ToString();
        for (int i = 0; i < tokenizer.VocabSize; i++)
        {
            if (tokenizer.DecodeToken(i) == target)
                return i;
        }
        return -1;
    }
}
