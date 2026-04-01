using DotLLM.Core.Constraints;
using DotLLM.Engine.Constraints;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints;

public class RegexConstraintTests
{
    /// <summary>
    /// Minimal tokenizer stub for testing regex constraint behavior.
    /// Maps single characters to token IDs for predictable testing.
    /// </summary>
    private sealed class StubTokenizer : ITokenizer
    {
        // Tokens: '0'-'9' as 0-9, '-' as 10, 'a'-'f' as 11-16, EOS as 17, " " as 18, "12" as 19
        private static readonly string[] Tokens =
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "-", "a", "b", "c", "d", "e", "f", "<eos>", " ", "12"];

        public int VocabSize => Tokens.Length;
        public int BosTokenId => 0;
        public int EosTokenId => 17;

        public string DecodeToken(int tokenId) =>
            tokenId >= 0 && tokenId < Tokens.Length ? Tokens[tokenId] : "";

        public int[] Encode(string text) => throw new NotImplementedException();
        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotImplementedException();
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => throw new NotImplementedException();
        public int CountTokens(string text) => throw new NotImplementedException();
    }

    [Fact]
    public void GetAllowedTokens_AtStart_AllowsOnlyDigitsForDigitPattern()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "\\d+");
        var mask = constraint.GetAllowedTokens();

        // Digits 0-9 should be allowed
        for (int i = 0; i <= 9; i++)
            Assert.True(mask.IsAllowed(i), $"Token '{i}' (digit) should be allowed at start of \\d+");

        // Non-digits should not be allowed
        Assert.False(mask.IsAllowed(10), "'-' should not be allowed at start of \\d+");
        Assert.False(mask.IsAllowed(11), "'a' should not be allowed at start of \\d+");

        // EOS should not be allowed (+ requires at least one digit)
        Assert.False(mask.IsAllowed(17), "EOS should not be allowed at start of \\d+");
    }

    [Fact]
    public void GetAllowedTokens_AfterDigit_AllowsDigitsAndEos()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "\\d+");
        constraint.Advance(5); // '5'

        var mask = constraint.GetAllowedTokens();

        // More digits should be allowed
        Assert.True(mask.IsAllowed(3), "Digit should be allowed after first digit");
        // EOS should be allowed (at least one digit matched)
        Assert.True(mask.IsAllowed(17), "EOS should be allowed after first digit in \\d+");
    }

    [Fact]
    public void Advance_DatePattern_IsComplete()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "\\d{4}-\\d{2}-\\d{2}");

        // Advance through "2024-01-15"
        int[] tokens = [2, 0, 2, 4, 10, 0, 1, 10, 1, 5];
        foreach (int t in tokens)
            constraint.Advance(t);

        Assert.True(constraint.IsComplete());
    }

    [Fact]
    public void GetAllowedTokens_DatePartial_DisallowsEos()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "\\d{4}-\\d{2}-\\d{2}");

        // Advance "202"
        constraint.Advance(2); // '2'
        constraint.Advance(0); // '0'
        constraint.Advance(2); // '2'

        var mask = constraint.GetAllowedTokens();
        Assert.False(mask.IsAllowed(17), "EOS should not be allowed mid-date");
    }

    [Fact]
    public void GetAllowedTokens_DateAfterYear_AllowsOnlyHyphen()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "\\d{4}-\\d{2}-\\d{2}");

        // Advance "2024"
        constraint.Advance(2);
        constraint.Advance(0);
        constraint.Advance(2);
        constraint.Advance(4);

        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(10), "'-' should be allowed after year digits");
        for (int i = 0; i <= 9; i++)
            Assert.False(mask.IsAllowed(i), $"Digit '{i}' should not be allowed after year (need hyphen)");
    }

    [Fact]
    public void GetAllowedTokens_MultiCharToken_AllowedIfFullyValid()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "\\d+");
        var mask = constraint.GetAllowedTokens();

        // Token 19 = "12" — both characters are digits, should be allowed
        Assert.True(mask.IsAllowed(19), "Multi-char token '12' should be allowed for \\d+");
    }

    [Fact]
    public void GetAllowedTokens_MultiCharToken_DisallowedIfPartiallyInvalid()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "\\d{1}");
        var mask = constraint.GetAllowedTokens();

        // Token 19 = "12" — pattern only allows one digit, second char would be invalid
        Assert.False(mask.IsAllowed(19), "Multi-char token '12' should be disallowed for \\d{1}");
    }

    [Fact]
    public void Clone_ProducesIndependentConstraint()
    {
        var original = new RegexConstraint(new StubTokenizer(), "\\d+");
        original.Advance(5); // '5'

        var clone = (RegexConstraint)original.Clone();
        clone.Advance(3); // '3' — advances only the clone

        // Original should still be at the same state as before cloning advanced further
        Assert.True(original.IsComplete()); // already has one digit
        Assert.True(clone.IsComplete());
    }

    [Fact]
    public void Reset_ReturnsToInitialState()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "\\d+");
        constraint.Advance(5);
        Assert.True(constraint.IsComplete());

        constraint.Reset();
        Assert.False(constraint.IsComplete());
    }

    [Fact]
    public void GetAllowedTokens_CacheHit_ReturnsSameMask()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "\\d+");
        var mask1 = constraint.GetAllowedTokens();
        var mask2 = constraint.GetAllowedTokens();

        // Should be the same reference (cached)
        Assert.Equal(mask1.AsSpan().ToArray(), mask2.AsSpan().ToArray());
    }

    [Fact]
    public void Advance_EosToken_DoesNotChangeState()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "\\d+");
        constraint.Advance(5); // '5'
        Assert.True(constraint.IsComplete());

        constraint.Advance(17); // EOS — should be no-op
        Assert.True(constraint.IsComplete());
    }

    [Fact]
    public void GetAllowedTokens_Alternation_AllowsBothBranches()
    {
        var constraint = new RegexConstraint(new StubTokenizer(), "(0|1)");
        var mask = constraint.GetAllowedTokens();

        Assert.True(mask.IsAllowed(0), "'0' should be allowed");
        Assert.True(mask.IsAllowed(1), "'1' should be allowed");
        Assert.False(mask.IsAllowed(2), "'2' should not be allowed for (0|1)");
    }

    [Fact]
    public void GetAllowedTokens_HexPattern_AllowsDigitsAndHexLetters()
    {
        // [0-9a-f]+
        var constraint = new RegexConstraint(new StubTokenizer(), "[0-9a-f]+");
        var mask = constraint.GetAllowedTokens();

        for (int i = 0; i <= 9; i++)
            Assert.True(mask.IsAllowed(i), $"Digit '{i}' should be allowed");
        for (int i = 11; i <= 16; i++)
            Assert.True(mask.IsAllowed(i), $"Hex letter token {i} should be allowed");
        Assert.False(mask.IsAllowed(10), "'-' should not be allowed");
    }
}
