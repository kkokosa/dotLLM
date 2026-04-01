using DotLLM.Engine.Constraints;
using DotLLM.Engine.Constraints.Regex;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints.Regex;

public class RegexParserTests
{
    [Fact]
    public void Parse_SingleLiteral_ReturnsLiteral()
    {
        var node = RegexParser.Parse("a");
        Assert.IsType<RegexNode.Literal>(node);
        Assert.Equal('a', ((RegexNode.Literal)node).Ch);
    }

    [Fact]
    public void Parse_Concatenation_ReturnsConcat()
    {
        var node = RegexParser.Parse("abc");
        var concat = Assert.IsType<RegexNode.Concat>(node);
        Assert.Equal(3, concat.Children.Length);
        Assert.Equal('a', ((RegexNode.Literal)concat.Children[0]).Ch);
        Assert.Equal('b', ((RegexNode.Literal)concat.Children[1]).Ch);
        Assert.Equal('c', ((RegexNode.Literal)concat.Children[2]).Ch);
    }

    [Fact]
    public void Parse_Alternation_ReturnsAlternation()
    {
        var node = RegexParser.Parse("a|b|c");
        var alt = Assert.IsType<RegexNode.Alternation>(node);
        Assert.Equal(3, alt.Children.Length);
    }

    [Fact]
    public void Parse_KleeneStar_ReturnsRepeat()
    {
        var node = RegexParser.Parse("a*");
        var repeat = Assert.IsType<RegexNode.Repeat>(node);
        Assert.Equal(0, repeat.Min);
        Assert.Equal(int.MaxValue, repeat.Max);
    }

    [Fact]
    public void Parse_Plus_ReturnsRepeatMin1()
    {
        var node = RegexParser.Parse("a+");
        var repeat = Assert.IsType<RegexNode.Repeat>(node);
        Assert.Equal(1, repeat.Min);
        Assert.Equal(int.MaxValue, repeat.Max);
    }

    [Fact]
    public void Parse_Optional_ReturnsRepeat01()
    {
        var node = RegexParser.Parse("a?");
        var repeat = Assert.IsType<RegexNode.Repeat>(node);
        Assert.Equal(0, repeat.Min);
        Assert.Equal(1, repeat.Max);
    }

    [Fact]
    public void Parse_BoundedRepeat_Exact()
    {
        var node = RegexParser.Parse("a{3}");
        var repeat = Assert.IsType<RegexNode.Repeat>(node);
        Assert.Equal(3, repeat.Min);
        Assert.Equal(3, repeat.Max);
    }

    [Fact]
    public void Parse_BoundedRepeat_Range()
    {
        var node = RegexParser.Parse("a{2,5}");
        var repeat = Assert.IsType<RegexNode.Repeat>(node);
        Assert.Equal(2, repeat.Min);
        Assert.Equal(5, repeat.Max);
    }

    [Fact]
    public void Parse_BoundedRepeat_Unbounded()
    {
        var node = RegexParser.Parse("a{2,}");
        var repeat = Assert.IsType<RegexNode.Repeat>(node);
        Assert.Equal(2, repeat.Min);
        Assert.Equal(int.MaxValue, repeat.Max);
    }

    [Fact]
    public void Parse_CharClass_Simple()
    {
        var node = RegexParser.Parse("[abc]");
        var cc = Assert.IsType<RegexNode.CharClass>(node);
        Assert.False(cc.Negated);
        Assert.Equal(3, cc.Ranges.Length);
    }

    [Fact]
    public void Parse_CharClass_Range()
    {
        var node = RegexParser.Parse("[a-z]");
        var cc = Assert.IsType<RegexNode.CharClass>(node);
        Assert.Single(cc.Ranges);
        Assert.Equal('a', cc.Ranges[0].Lo);
        Assert.Equal('z', cc.Ranges[0].Hi);
    }

    [Fact]
    public void Parse_CharClass_Negated()
    {
        var node = RegexParser.Parse("[^0-9]");
        var cc = Assert.IsType<RegexNode.CharClass>(node);
        Assert.True(cc.Negated);
    }

    [Fact]
    public void Parse_Dot_MatchesAnyExceptNewline()
    {
        var node = RegexParser.Parse(".");
        var cc = Assert.IsType<RegexNode.CharClass>(node);
        Assert.True(cc.Negated);
        Assert.Single(cc.Ranges);
        Assert.Equal('\n', cc.Ranges[0].Lo);
    }

    [Fact]
    public void Parse_Group_ReturnsInner()
    {
        var node = RegexParser.Parse("(ab)");
        var concat = Assert.IsType<RegexNode.Concat>(node);
        Assert.Equal(2, concat.Children.Length);
    }

    [Fact]
    public void Parse_EscapeDigit_ReturnsCharClass()
    {
        var node = RegexParser.Parse("\\d");
        var cc = Assert.IsType<RegexNode.CharClass>(node);
        Assert.Single(cc.Ranges);
        Assert.Equal('0', cc.Ranges[0].Lo);
        Assert.Equal('9', cc.Ranges[0].Hi);
    }

    [Fact]
    public void Parse_EscapedLiteral_ReturnsLiteral()
    {
        var node = RegexParser.Parse("\\.");
        Assert.IsType<RegexNode.Literal>(node);
        Assert.Equal('.', ((RegexNode.Literal)node).Ch);
    }

    [Fact]
    public void Parse_DatePattern_Complex()
    {
        // \d{4}-\d{2}-\d{2}
        var node = RegexParser.Parse("\\d{4}-\\d{2}-\\d{2}");
        var concat = Assert.IsType<RegexNode.Concat>(node);
        Assert.Equal(5, concat.Children.Length); // \d{4}, -, \d{2}, -, \d{2}
    }

    [Fact]
    public void Parse_Backreference_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() => RegexParser.Parse("(a)\\1"));
        Assert.Contains("Backreferences", ex.Message);
    }

    [Fact]
    public void Parse_Lookahead_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() => RegexParser.Parse("(?=a)b"));
        Assert.Contains("Lookahead", ex.Message);
    }

    [Fact]
    public void Parse_LazyQuantifier_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() => RegexParser.Parse("a*?"));
        Assert.Contains("Lazy quantifiers", ex.Message);
    }

    [Fact]
    public void Parse_Anchor_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() => RegexParser.Parse("^abc$"));
        Assert.Contains("Anchors", ex.Message);
    }

    [Fact]
    public void Parse_EmptyPattern_Throws()
    {
        Assert.Throws<ArgumentException>(() => RegexParser.Parse(""));
    }

    [Fact]
    public void Parse_NonCapturingGroup_Succeeds()
    {
        var node = RegexParser.Parse("(?:ab)");
        var concat = Assert.IsType<RegexNode.Concat>(node);
        Assert.Equal(2, concat.Children.Length);
    }
}
