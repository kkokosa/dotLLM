using DotLLM.Engine.Constraints.Grammar;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints.Grammar;

public class GbnfParserTests
{
    [Fact]
    public void Parse_SimpleLiteral_ProducesLiteral()
    {
        var grammar = GbnfParser.Parse("root ::= \"hello\"");
        Assert.Single(grammar.Rules);
        Assert.Equal("root", grammar.RootRuleName);
        var body = Assert.IsType<GbnfNode.Literal>(grammar.Rules["root"].Body);
        Assert.Equal("hello", body.Text);
    }

    [Fact]
    public void Parse_Alternation_ProducesAlternation()
    {
        var grammar = GbnfParser.Parse("root ::= \"yes\" | \"no\"");
        var body = Assert.IsType<GbnfNode.Alternation>(grammar.Rules["root"].Body);
        Assert.Equal(2, body.Alternatives.Length);
    }

    [Fact]
    public void Parse_CharClass_ProducesCharClass()
    {
        var grammar = GbnfParser.Parse("root ::= [a-z]");
        var body = Assert.IsType<GbnfNode.CharClass>(grammar.Rules["root"].Body);
        Assert.False(body.Negated);
        Assert.Single(body.Ranges);
        Assert.Equal('a', body.Ranges[0].Lo);
        Assert.Equal('z', body.Ranges[0].Hi);
    }

    [Fact]
    public void Parse_NegatedCharClass_SetsNegatedFlag()
    {
        var grammar = GbnfParser.Parse("root ::= [^0-9]");
        var body = Assert.IsType<GbnfNode.CharClass>(grammar.Rules["root"].Body);
        Assert.True(body.Negated);
    }

    [Fact]
    public void Parse_Sequence_ProducesSequence()
    {
        var grammar = GbnfParser.Parse("root ::= \"a\" \"b\"");
        var body = Assert.IsType<GbnfNode.Sequence>(grammar.Rules["root"].Body);
        Assert.Equal(2, body.Elements.Length);
    }

    [Fact]
    public void Parse_Repetition_Star()
    {
        var grammar = GbnfParser.Parse("root ::= [a-z]*");
        var body = Assert.IsType<GbnfNode.Repeat>(grammar.Rules["root"].Body);
        Assert.Equal(0, body.Min);
        Assert.Equal(int.MaxValue, body.Max);
    }

    [Fact]
    public void Parse_Repetition_Plus()
    {
        var grammar = GbnfParser.Parse("root ::= [a-z]+");
        var body = Assert.IsType<GbnfNode.Repeat>(grammar.Rules["root"].Body);
        Assert.Equal(1, body.Min);
        Assert.Equal(int.MaxValue, body.Max);
    }

    [Fact]
    public void Parse_Repetition_Optional()
    {
        var grammar = GbnfParser.Parse("root ::= [a-z]?");
        var body = Assert.IsType<GbnfNode.Repeat>(grammar.Rules["root"].Body);
        Assert.Equal(0, body.Min);
        Assert.Equal(1, body.Max);
    }

    [Fact]
    public void Parse_RuleReference_ProducesRuleRef()
    {
        var grammar = GbnfParser.Parse("root ::= greeting\ngreeting ::= \"hi\"");
        Assert.Equal(2, grammar.Rules.Count);
        var rootBody = Assert.IsType<GbnfNode.RuleRef>(grammar.Rules["root"].Body);
        Assert.Equal("greeting", rootBody.RuleName);
    }

    [Fact]
    public void Parse_MultipleRules_FirstIsRoot()
    {
        var grammar = GbnfParser.Parse("root ::= name\nname ::= [A-Z] [a-z]+");
        Assert.Equal("root", grammar.RootRuleName);
        Assert.Equal(2, grammar.Rules.Count);
    }

    [Fact]
    public void Parse_Comment_Ignored()
    {
        var grammar = GbnfParser.Parse("# this is a comment\nroot ::= \"hello\"");
        Assert.Single(grammar.Rules);
    }

    [Fact]
    public void Parse_InlineComment_Stripped()
    {
        var grammar = GbnfParser.Parse("root ::= \"hello\" # inline comment");
        var body = Assert.IsType<GbnfNode.Literal>(grammar.Rules["root"].Body);
        Assert.Equal("hello", body.Text);
    }

    [Fact]
    public void Parse_Group_Grouping()
    {
        var grammar = GbnfParser.Parse("root ::= (\"a\" | \"b\") \"c\"");
        var body = Assert.IsType<GbnfNode.Sequence>(grammar.Rules["root"].Body);
        Assert.Equal(2, body.Elements.Length);
        Assert.IsType<GbnfNode.Alternation>(body.Elements[0]);
    }

    [Fact]
    public void Parse_EscapedLiteral_HandlesEscapes()
    {
        var grammar = GbnfParser.Parse("root ::= \"hello\\nworld\"");
        var body = Assert.IsType<GbnfNode.Literal>(grammar.Rules["root"].Body);
        Assert.Equal("hello\nworld", body.Text);
    }

    [Fact]
    public void Parse_UndefinedRule_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            GbnfParser.Parse("root ::= missing-rule"));
        Assert.Contains("Undefined rule reference", ex.Message);
    }

    [Fact]
    public void Parse_LeftRecursion_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            GbnfParser.Parse("root ::= root \"a\""));
        Assert.Contains("Left recursion", ex.Message);
    }

    [Fact]
    public void Parse_EmptyGrammar_Throws()
    {
        Assert.Throws<ArgumentException>(() => GbnfParser.Parse(""));
    }

    [Fact]
    public void Parse_DuplicateRule_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            GbnfParser.Parse("root ::= \"a\"\nroot ::= \"b\""));
        Assert.Contains("Duplicate rule", ex.Message);
    }

    [Fact]
    public void Parse_ComplexGrammar_Succeeds()
    {
        var grammar = GbnfParser.Parse(
            """
            root ::= greeting " " name
            greeting ::= "hello" | "hi" | "hey"
            name ::= [A-Z] [a-z]+
            """);
        Assert.Equal(3, grammar.Rules.Count);
        Assert.Equal("root", grammar.RootRuleName);
    }
}
