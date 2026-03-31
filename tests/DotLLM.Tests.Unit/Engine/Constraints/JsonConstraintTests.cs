using DotLLM.Core.Constraints;
using DotLLM.Engine.Constraints;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints;

public class JsonConstraintTests
{
    /// <summary>
    /// Minimal tokenizer stub for testing constraint behavior.
    /// Maps single characters to token IDs for predictable testing.
    /// </summary>
    private sealed class StubTokenizer : ITokenizer
    {
        // Token 0='{', 1='}', 2='"', 3=':', 4=',', 5='a', 6='1', 7='[', 8=']',
        // 9='t', 10='r', 11='u', 12='e', 13=EOS, 14=' '
        private static readonly string[] Tokens =
            ["{", "}", "\"", ":", ",", "a", "1", "[", "]", "t", "r", "u", "e", "<eos>", " "];

        public int VocabSize => Tokens.Length;
        public int BosTokenId => 0;
        public int EosTokenId => 13;

        public string DecodeToken(int tokenId) =>
            tokenId >= 0 && tokenId < Tokens.Length ? Tokens[tokenId] : "";

        public int[] Encode(string text) => throw new NotImplementedException();
        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotImplementedException();
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => throw new NotImplementedException();
        public int CountTokens(string text) => throw new NotImplementedException();
    }

    private static JsonConstraint CreateConstraint() => new(new StubTokenizer());

    [Fact]
    public void GetAllowedTokens_AtStart_AllowsOpenBraceAndBracket()
    {
        var constraint = CreateConstraint();
        var mask = constraint.GetAllowedTokens();

        // '{' (token 0) and '[' (token 7) should be allowed; whitespace is NOT
        Assert.True(mask.IsAllowed(0), "'{' should be allowed at start");
        Assert.True(mask.IsAllowed(7), "'[' should be allowed at start");
        Assert.False(mask.IsAllowed(14), "' ' should not be allowed at start (no leading ws)");

        // '}', '"', ':', ',', digits, literals should not be allowed
        Assert.False(mask.IsAllowed(1), "'}' should not be allowed at start");
        Assert.False(mask.IsAllowed(6), "'1' should not be allowed at start");

        // EOS should not be allowed (no complete JSON yet)
        Assert.False(mask.IsAllowed(13), "EOS should not be allowed at start");
    }

    [Fact]
    public void GetAllowedTokens_AfterOpenBrace_AllowsQuoteAndCloseBrace()
    {
        var constraint = CreateConstraint();
        constraint.Advance(0); // '{'

        var mask = constraint.GetAllowedTokens();

        // '"' (token 2) for key string, '}' (token 1) for empty object, ' ' for whitespace
        Assert.True(mask.IsAllowed(2), "'\"' should be allowed after '{'");
        Assert.True(mask.IsAllowed(1), "'}' should be allowed after '{'");
        Assert.True(mask.IsAllowed(14), "' ' should be allowed after '{'");

        // '[', '{', digits should not be allowed (need key string first)
        Assert.False(mask.IsAllowed(0), "'{' should not be allowed after '{'");
        Assert.False(mask.IsAllowed(6), "'1' should not be allowed after '{'");
    }

    [Fact]
    public void Advance_EmptyObject_IsComplete()
    {
        var constraint = CreateConstraint();
        constraint.Advance(0); // '{'
        Assert.False(constraint.IsComplete());

        constraint.Advance(1); // '}'
        Assert.True(constraint.IsComplete());
    }

    [Fact]
    public void GetAllowedTokens_WhenComplete_AllowsEos()
    {
        var constraint = CreateConstraint();
        constraint.Advance(0); // '{'
        constraint.Advance(1); // '}'

        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(13), "EOS should be allowed when JSON is complete");
    }

    [Fact]
    public void Clone_ProducesIndependentConstraint()
    {
        var constraint = CreateConstraint();
        constraint.Advance(0); // '{'

        var clone = (JsonConstraint)constraint.Clone();

        // Advance original to complete
        constraint.Advance(1); // '}'
        Assert.True(constraint.IsComplete());

        // Clone should still be in progress
        Assert.False(clone.IsComplete());

        // Clone can independently complete
        clone.Advance(1); // '}'
        Assert.True(clone.IsComplete());
    }

    [Fact]
    public void Reset_ReturnsToInitialState()
    {
        var constraint = CreateConstraint();
        constraint.Advance(0); // '{'
        constraint.Advance(1); // '}'
        Assert.True(constraint.IsComplete());

        constraint.Reset();
        Assert.False(constraint.IsComplete());

        // Should be able to start fresh
        var mask = constraint.GetAllowedTokens();
        Assert.True(mask.IsAllowed(0), "'{' should be allowed after reset");
    }

    [Fact]
    public void GetAllowedTokens_CacheHit_ReturnsSameMask()
    {
        var constraint = CreateConstraint();

        // Call twice at the same state — should return cached mask
        var mask1 = constraint.GetAllowedTokens();
        var mask2 = constraint.GetAllowedTokens();

        // Both should be structurally identical (and ideally the same object)
        for (int i = 0; i < new StubTokenizer().VocabSize; i++)
            Assert.Equal(mask1.IsAllowed(i), mask2.IsAllowed(i));
    }

    [Fact]
    public void Advance_EosToken_DoesNotChangeState()
    {
        var constraint = CreateConstraint();
        constraint.Advance(0); // '{'

        // Advance with EOS should be a no-op
        bool wasBefore = constraint.IsComplete();
        constraint.Advance(13); // EOS
        Assert.Equal(wasBefore, constraint.IsComplete());
    }

    [Fact]
    public void Advance_FullKeyValuePair_TracksCorrectly()
    {
        var constraint = CreateConstraint();

        // Build {"a":"a"}
        constraint.Advance(0);  // '{'
        Assert.False(constraint.IsComplete());

        constraint.Advance(2);  // '"'
        constraint.Advance(5);  // 'a'
        constraint.Advance(2);  // '"'
        constraint.Advance(3);  // ':'
        constraint.Advance(2);  // '"'
        constraint.Advance(5);  // 'a'
        constraint.Advance(2);  // '"'

        Assert.False(constraint.IsComplete());

        constraint.Advance(1);  // '}'
        Assert.True(constraint.IsComplete());
    }
}
