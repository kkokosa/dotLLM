using DotLLM.Core.Constraints;
using DotLLM.Engine.Constraints;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints;

public class GrammarConstraintTests
{
    /// <summary>
    /// Minimal tokenizer stub for testing grammar constraint behavior.
    /// Maps single characters to token IDs for predictable testing.
    /// </summary>
    private sealed class StubTokenizer : ITokenizer
    {
        // Tokens: 'h'=0, 'i'=1, ' '=2, 'J'=3, 'o'=4, 'e'=5, 'y'=6, 's'=7,
        //         'n'=8, 'a'=9, 'b'=10, 'c'=11, EOS=12
        private static readonly string[] Tokens =
            ["h", "i", " ", "J", "o", "e", "y", "s", "n", "a", "b", "c", "<eos>"];

        public int VocabSize => Tokens.Length;
        public int BosTokenId => 0;
        public int EosTokenId => 12;

        public string DecodeToken(int tokenId) =>
            tokenId >= 0 && tokenId < Tokens.Length ? Tokens[tokenId] : "";

        public int[] Encode(string text) => throw new NotImplementedException();
        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotImplementedException();
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => throw new NotImplementedException();
        public int CountTokens(string text) => throw new NotImplementedException();
    }

    [Fact]
    public void GetAllowedTokens_SimpleLiteral_AllowsFirstChar()
    {
        var constraint = new GrammarConstraint(new StubTokenizer(), "root ::= \"hi\"");
        var mask = constraint.GetAllowedTokens();

        Assert.True(mask.IsAllowed(0), "'h' should be allowed at start of 'hi'");
        Assert.False(mask.IsAllowed(1), "'i' should not be allowed at start");
        Assert.False(mask.IsAllowed(12), "EOS should not be allowed at start");
    }

    [Fact]
    public void Advance_SimpleLiteral_CompletesCorrectly()
    {
        var constraint = new GrammarConstraint(new StubTokenizer(), "root ::= \"hi\"");
        constraint.Advance(0); // 'h'
        Assert.False(constraint.IsComplete());

        constraint.Advance(1); // 'i'
        Assert.True(constraint.IsComplete());
    }

    [Fact]
    public void GetAllowedTokens_AfterComplete_AllowsEos()
    {
        var constraint = new GrammarConstraint(new StubTokenizer(), "root ::= \"hi\"");
        constraint.Advance(0); // 'h'
        constraint.Advance(1); // 'i'

        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(12), "EOS should be allowed after complete match");
    }

    [Fact]
    public void GetAllowedTokens_Alternation_AllowsBothFirstChars()
    {
        var constraint = new GrammarConstraint(new StubTokenizer(),
            "root ::= \"hi\" | \"no\"");
        var mask = constraint.GetAllowedTokens();

        Assert.True(mask.IsAllowed(0), "'h' should be allowed (start of 'hi')");
        Assert.True(mask.IsAllowed(8), "'n' should be allowed (start of 'no')");
        Assert.False(mask.IsAllowed(9), "'a' should not be allowed");
    }

    [Fact]
    public void Clone_ProducesIndependentConstraint()
    {
        var original = new GrammarConstraint(new StubTokenizer(), "root ::= \"hi\"");
        original.Advance(0); // 'h'

        var clone = (GrammarConstraint)original.Clone();
        clone.Advance(1); // 'i' — advances only the clone

        Assert.False(original.IsComplete());
        Assert.True(clone.IsComplete());
    }

    [Fact]
    public void Reset_ReturnsToInitialState()
    {
        var constraint = new GrammarConstraint(new StubTokenizer(), "root ::= \"hi\"");
        constraint.Advance(0); // 'h'
        constraint.Advance(1); // 'i'
        Assert.True(constraint.IsComplete());

        constraint.Reset();
        Assert.False(constraint.IsComplete());
    }

    [Fact]
    public void Advance_EosToken_DoesNotChangeState()
    {
        var constraint = new GrammarConstraint(new StubTokenizer(), "root ::= \"hi\"");
        constraint.Advance(0); // 'h'
        constraint.Advance(12); // EOS — should be no-op
        Assert.False(constraint.IsComplete());

        // Should still be able to advance 'i'
        constraint.Advance(1);
        Assert.True(constraint.IsComplete());
    }

    [Fact]
    public void GetAllowedTokens_RuleReference_FollowsRule()
    {
        var constraint = new GrammarConstraint(new StubTokenizer(),
            """
            root ::= greeting
            greeting ::= "hi"
            """);
        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(0), "'h' should be allowed at start of greeting rule");
    }

    [Fact]
    public void Advance_NestedRules_CompletesCorrectly()
    {
        var constraint = new GrammarConstraint(new StubTokenizer(),
            """
            root ::= greeting " " name
            greeting ::= "hi"
            name ::= [a-z]+
            """);

        constraint.Advance(0); // 'h'
        constraint.Advance(1); // 'i'
        Assert.False(constraint.IsComplete());

        constraint.Advance(2); // ' '
        Assert.False(constraint.IsComplete());

        constraint.Advance(9); // 'a'
        Assert.True(constraint.IsComplete()); // name has at least one char
    }
}
