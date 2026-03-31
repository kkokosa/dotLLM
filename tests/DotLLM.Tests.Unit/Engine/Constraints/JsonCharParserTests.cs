using DotLLM.Engine.Constraints;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints;

public class JsonCharParserTests
{
    private static bool Parse(string input)
    {
        var parser = new JsonCharParser();
        foreach (char c in input)
        {
            if (!parser.TryAdvance(c))
                return false;
        }
        return true;
    }

    private static bool ParseComplete(string input)
    {
        var parser = new JsonCharParser();
        foreach (char c in input)
        {
            if (!parser.TryAdvance(c))
                return false;
        }
        return parser.IsComplete;
    }

    // ── Valid complete JSON ──────────────────────────────────────────

    [Theory]
    [InlineData("{}")]
    [InlineData("[]")]
    [InlineData("{\"a\":1}")]
    [InlineData("{\"key\":\"value\"}")]
    [InlineData("{\"a\":true,\"b\":false,\"c\":null}")]
    [InlineData("{\"n\":0}")]
    [InlineData("{\"n\":-1}")]
    [InlineData("{\"n\":3.14}")]
    [InlineData("{\"n\":1e10}")]
    [InlineData("{\"n\":-2.5E-3}")]
    [InlineData("[1,2,3]")]
    [InlineData("[\"a\",\"b\"]")]
    [InlineData("[true,false,null]")]
    [InlineData("{\"a\":{\"b\":[1,2,{\"c\":3}]}}")]
    [InlineData("[{},{},{}]")]
    [InlineData("{ \"key\" : \"value\" }")]
    [InlineData("{\n  \"key\": \"value\"\n}")]
    public void ParseComplete_ValidJson_ReturnsTrue(string json)
    {
        Assert.True(ParseComplete(json), $"Expected complete for: {json}");
    }

    // ── Valid partial JSON (parseable but not complete) ──────────────

    [Theory]
    [InlineData("{")]
    [InlineData("[")]
    [InlineData("{\"a\"")]
    [InlineData("{\"a\":")]
    [InlineData("{\"a\":1,")]
    [InlineData("[1,")]
    [InlineData("{\"a\":{\"b\":")]
    public void Parse_ValidPartial_ReturnsTrueButNotComplete(string json)
    {
        Assert.True(Parse(json), $"Expected parseable for: {json}");
        var parser = new JsonCharParser();
        foreach (char c in json)
            parser.TryAdvance(c);
        Assert.False(parser.IsComplete, $"Expected not complete for: {json}");
    }

    // ── Invalid JSON ────────────────────────────────────────────────

    [Theory]
    [InlineData("}")]       // close brace at start
    [InlineData("]")]       // close bracket at start
    [InlineData("a")]       // bare letter at start (root must be { or [)
    [InlineData("1")]       // number at root
    [InlineData("\"str\"")] // string at root
    [InlineData("true")]    // literal at root
    public void Parse_InvalidRoot_ReturnsFalse(string json)
    {
        Assert.False(Parse(json), $"Expected reject for: {json}");
    }

    [Fact]
    public void Parse_TrailingContent_Rejected()
    {
        var parser = new JsonCharParser();
        foreach (char c in "{}")
            Assert.True(parser.TryAdvance(c));
        Assert.True(parser.IsComplete);

        // Non-whitespace after complete JSON is rejected
        Assert.False(parser.TryAdvance('{'));
    }

    [Fact]
    public void Parse_TrailingWhitespace_Rejected()
    {
        var parser = new JsonCharParser();
        foreach (char c in "{}")
            Assert.True(parser.TryAdvance(c));
        Assert.True(parser.IsComplete);

        // No trailing characters allowed — forces EOS for generation
        Assert.False(parser.TryAdvance(' '));
    }

    // ── String escapes ──────────────────────────────────────────────

    [Theory]
    [InlineData("{\"a\":\"hello\\\"world\"}")]   // escaped quote
    [InlineData("{\"a\":\"line\\n\"}")]           // escaped newline
    [InlineData("{\"a\":\"tab\\t\"}")]            // escaped tab
    [InlineData("{\"a\":\"back\\\\slash\"}")]     // escaped backslash
    [InlineData("{\"a\":\"slash\\/\"}")]          // escaped forward slash
    [InlineData("{\"a\":\"\\u0041\"}")]           // unicode escape
    [InlineData("{\"a\":\"\\uaBcD\"}")]           // mixed-case unicode hex
    public void ParseComplete_StringEscapes_Valid(string json)
    {
        Assert.True(ParseComplete(json), $"Expected complete for: {json}");
    }

    [Fact]
    public void Parse_InvalidEscape_Rejected()
    {
        // \x is not a valid JSON escape
        Assert.False(Parse("{\"a\":\"\\x\"}"));
    }

    [Fact]
    public void Parse_IncompleteUnicode_NotComplete()
    {
        // \u00 is only 2 hex digits, needs 4
        Assert.True(Parse("{\"a\":\"\\u00"));
        Assert.False(ParseComplete("{\"a\":\"\\u00"));
    }

    [Fact]
    public void Parse_ControlCharInString_Rejected()
    {
        // Control chars (< 0x20) not allowed unescaped in strings
        var parser = new JsonCharParser();
        Assert.True(parser.TryAdvance('{'));
        Assert.True(parser.TryAdvance('"'));
        Assert.True(parser.TryAdvance('a'));
        Assert.False(parser.TryAdvance('\x01')); // control char
    }

    // ── Numbers ─────────────────────────────────────────────────────

    [Theory]
    [InlineData("{\"n\":0}")]
    [InlineData("{\"n\":1}")]
    [InlineData("{\"n\":42}")]
    [InlineData("{\"n\":-1}")]
    [InlineData("{\"n\":-0}")]
    [InlineData("{\"n\":3.14}")]
    [InlineData("{\"n\":0.5}")]
    [InlineData("{\"n\":1e10}")]
    [InlineData("{\"n\":1E10}")]
    [InlineData("{\"n\":1e+10}")]
    [InlineData("{\"n\":1e-10}")]
    [InlineData("{\"n\":-2.5E-3}")]
    [InlineData("{\"n\":100}")]
    public void ParseComplete_Numbers_Valid(string json)
    {
        Assert.True(ParseComplete(json), $"Expected complete for: {json}");
    }

    [Fact]
    public void Parse_LeadingZero_Rejected()
    {
        // 01 is not valid JSON — after 0, only ., e/E, or non-number char allowed
        var parser = new JsonCharParser();
        Assert.True(parser.TryAdvance('{'));
        Assert.True(parser.TryAdvance('"'));
        Assert.True(parser.TryAdvance('n'));
        Assert.True(parser.TryAdvance('"'));
        Assert.True(parser.TryAdvance(':'));
        Assert.True(parser.TryAdvance('0'));
        // Next char '1' — number termination triggers, then '1' is invalid after value
        Assert.False(parser.TryAdvance('1'));
    }

    [Fact]
    public void Parse_DotWithoutFracDigit_Rejected()
    {
        // "0." needs at least one digit after dot
        var parser = new JsonCharParser();
        foreach (char c in "{\"n\":0.")
            Assert.True(parser.TryAdvance(c));
        // Next must be a digit, not }
        Assert.False(parser.TryAdvance('}'));
    }

    // ── Nested structures ───────────────────────────────────────────

    [Fact]
    public void ParseComplete_DeepNesting()
    {
        // 10 levels deep
        string json = "{\"a\":{\"b\":{\"c\":{\"d\":{\"e\":{\"f\":{\"g\":{\"h\":{\"i\":{\"j\":1}}}}}}}}}}";
        Assert.True(ParseComplete(json));
    }

    [Fact]
    public void ParseComplete_MixedNesting()
    {
        string json = "[{\"a\":[1,{\"b\":[2,3]},4]},5]";
        Assert.True(ParseComplete(json));
    }

    [Fact]
    public void ParseComplete_EmptyNestedContainers()
    {
        Assert.True(ParseComplete("{\"a\":{},\"b\":[]}"));
        Assert.True(ParseComplete("[[],[],{}]"));
    }

    // ── Literals ────────────────────────────────────────────────────

    [Fact]
    public void ParseComplete_Literals()
    {
        Assert.True(ParseComplete("{\"a\":true}"));
        Assert.True(ParseComplete("{\"a\":false}"));
        Assert.True(ParseComplete("{\"a\":null}"));
        Assert.True(ParseComplete("[true,false,null]"));
    }

    [Fact]
    public void Parse_PartialLiteral_Rejected()
    {
        // "tru" followed by } — the 'e' is expected, not '}'
        var parser = new JsonCharParser();
        foreach (char c in "{\"a\":tru")
            Assert.True(parser.TryAdvance(c));
        Assert.False(parser.TryAdvance('}'));
    }

    // ── Clone and Reset ─────────────────────────────────────────────

    [Fact]
    public void StructCopy_ProducesIndependentCopy()
    {
        var parser = new JsonCharParser();
        foreach (char c in "{\"a\":")
            parser.TryAdvance(c);

        var clone = parser; // struct copy — zero allocations

        // Advance original further
        parser.TryAdvance('1');
        parser.TryAdvance('}');
        Assert.True(parser.IsComplete);

        // Clone should still be at the same state as before
        Assert.False(clone.IsComplete);
        Assert.True(clone.TryAdvance('2'));
        Assert.True(clone.TryAdvance('}'));
        Assert.True(clone.IsComplete);
    }

    [Fact]
    public void Reset_ReturnsToInitialState()
    {
        var parser = new JsonCharParser();
        foreach (char c in "{\"a\":1}")
            parser.TryAdvance(c);
        Assert.True(parser.IsComplete);

        parser.Reset();
        Assert.False(parser.IsComplete);
        Assert.True(parser.TryAdvance('{'));
    }

    // ── Effective state key ─────────────────────────────────────────

    [Fact]
    public void GetEffectiveStateKey_SameState_SameKey()
    {
        var parser1 = new JsonCharParser();
        var parser2 = new JsonCharParser();

        // Both at Start state
        Assert.Equal(parser1.GetEffectiveStateKey(), parser2.GetEffectiveStateKey());
    }

    [Fact]
    public void GetEffectiveStateKey_DifferentState_DifferentKey()
    {
        var parser1 = new JsonCharParser();
        var parser2 = new JsonCharParser();
        parser2.TryAdvance('{');

        Assert.NotEqual(parser1.GetEffectiveStateKey(), parser2.GetEffectiveStateKey());
    }

    // ── Whitespace handling ─────────────────────────────────────────

    [Fact]
    public void ParseComplete_WhitespaceBetweenTokens()
    {
        Assert.True(ParseComplete("{ \t\n\r\"key\" \t: \t\"val\" \t}"));
    }

    [Fact]
    public void Parse_LeadingWhitespace_Rejected()
    {
        // No leading whitespace — forces immediate { or [
        var parser = new JsonCharParser();
        Assert.False(parser.TryAdvance(' '));
    }

    // ── Array edge cases ────────────────────────────────────────────

    [Fact]
    public void ParseComplete_SingleElementArray()
    {
        Assert.True(ParseComplete("[1]"));
        Assert.True(ParseComplete("[\"hello\"]"));
        Assert.True(ParseComplete("[true]"));
        Assert.True(ParseComplete("[{}]"));
        Assert.True(ParseComplete("[[]]"));
    }

    [Fact]
    public void Parse_TrailingCommaInArray_Rejected()
    {
        // [1,] is not valid JSON
        var parser = new JsonCharParser();
        foreach (char c in "[1,")
            Assert.True(parser.TryAdvance(c));
        // Next must be a value, not ]
        Assert.False(parser.TryAdvance(']'));
    }

    [Fact]
    public void Parse_TrailingCommaInObject_Rejected()
    {
        // {"a":1,} is not valid JSON
        var parser = new JsonCharParser();
        foreach (char c in "{\"a\":1,")
            Assert.True(parser.TryAdvance(c));
        // After comma in object, must have a key string, not }
        Assert.False(parser.TryAdvance('}'));
    }

    // ── Unicode escape in key (regression: KeyStringFlag preservation) ──

    [Fact]
    public void ParseComplete_UnicodeEscapeInKey()
    {
        // \u0061 = 'a' — key with unicode escape must still route to ObjectColon
        Assert.True(ParseComplete("{\"\\u0061\":1}"));
    }

    [Fact]
    public void ParseComplete_UnicodeEscapeInValue()
    {
        Assert.True(ParseComplete("{\"key\":\"\\u0041\"}"));
    }

    [Fact]
    public void ParseComplete_MultipleUnicodeEscapesInKey()
    {
        Assert.True(ParseComplete("{\"\\u0061\\u0062\":true}"));
    }

    // ── Max depth rejection ─────────────────────────────────────────

    [Fact]
    public void Parse_ExceedMaxDepth_Rejected()
    {
        var parser = new JsonCharParser();
        // 64 nested arrays = MaxDepth
        for (int i = 0; i < 64; i++)
            Assert.True(parser.TryAdvance('['));
        // 65th should be rejected
        Assert.False(parser.TryAdvance('['));
    }

    [Fact]
    public void ParseComplete_ExactlyMaxDepth()
    {
        // 64 nested arrays with innermost value, then 64 closes
        var parser = new JsonCharParser();
        for (int i = 0; i < 64; i++)
            Assert.True(parser.TryAdvance('['));
        Assert.True(parser.TryAdvance('1'));
        for (int i = 0; i < 64; i++)
            Assert.True(parser.TryAdvance(']'));
        Assert.True(parser.IsComplete);
    }
}
