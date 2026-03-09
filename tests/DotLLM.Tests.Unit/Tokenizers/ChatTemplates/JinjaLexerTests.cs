using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

public class JinjaLexerTests
{
    private static List<JinjaToken> Lex(string source) => new JinjaLexer(source).Tokenize();

    [Fact]
    public void PlainText_SingleTextToken()
    {
        var tokens = Lex("Hello, world!");
        Assert.Equal(JinjaTokenType.Text, tokens[0].Type);
        Assert.Equal("Hello, world!", tokens[0].Value);
        Assert.Equal(JinjaTokenType.Eof, tokens[1].Type);
    }

    [Fact]
    public void Expression_IdentifierTokens()
    {
        var tokens = Lex("{{ name }}");
        Assert.Equal(JinjaTokenType.ExprStart, tokens[0].Type);
        Assert.Equal(JinjaTokenType.Identifier, tokens[1].Type);
        Assert.Equal("name", tokens[1].Value);
        Assert.Equal(JinjaTokenType.ExprEnd, tokens[2].Type);
        Assert.Equal(JinjaTokenType.Eof, tokens[3].Type);
    }

    [Fact]
    public void Statement_IfKeyword()
    {
        var tokens = Lex("{% if x %}yes{% endif %}");
        Assert.Equal(JinjaTokenType.StmtStart, tokens[0].Type);
        Assert.Equal(JinjaTokenType.If, tokens[1].Type);
        Assert.Equal(JinjaTokenType.Identifier, tokens[2].Type);
        Assert.Equal("x", tokens[2].Value);
        Assert.Equal(JinjaTokenType.StmtEnd, tokens[3].Type);
        Assert.Equal(JinjaTokenType.Text, tokens[4].Type);
        Assert.Equal("yes", tokens[4].Value);
    }

    [Fact]
    public void WhitespaceControl_StripLeft()
    {
        var tokens = Lex("Hello   \n  {{- name }}");
        // {{- strips trailing whitespace from preceding text
        Assert.Equal(JinjaTokenType.Text, tokens[0].Type);
        Assert.Equal("Hello", tokens[0].Value);
        Assert.Equal(JinjaTokenType.ExprStart, tokens[1].Type);
    }

    [Fact]
    public void WhitespaceControl_StripRight()
    {
        var tokens = Lex("{{ name -}}   \n  world");
        Assert.Equal(JinjaTokenType.ExprStart, tokens[0].Type);
        Assert.Equal(JinjaTokenType.Identifier, tokens[1].Type);
        Assert.Equal(JinjaTokenType.ExprEnd, tokens[2].Type);
        // -}} strips leading whitespace from following text
        Assert.Equal(JinjaTokenType.Text, tokens[3].Type);
        Assert.Equal("world", tokens[3].Value);
    }

    [Fact]
    public void Comment_Discarded()
    {
        var tokens = Lex("before{# comment #}after");
        Assert.Equal(JinjaTokenType.Text, tokens[0].Type);
        Assert.Equal("before", tokens[0].Value);
        Assert.Equal(JinjaTokenType.Text, tokens[1].Type);
        Assert.Equal("after", tokens[1].Value);
    }

    [Fact]
    public void StringLiteral_DoubleQuotes()
    {
        var tokens = Lex("{{ \"hello\" }}");
        Assert.Equal(JinjaTokenType.StringLiteral, tokens[1].Type);
        Assert.Equal("hello", tokens[1].Value);
    }

    [Fact]
    public void StringLiteral_SingleQuotes()
    {
        var tokens = Lex("{{ 'world' }}");
        Assert.Equal(JinjaTokenType.StringLiteral, tokens[1].Type);
        Assert.Equal("world", tokens[1].Value);
    }

    [Fact]
    public void StringLiteral_EscapedQuotes()
    {
        var tokens = Lex("{{ \"he said \\\"hi\\\"\" }}");
        Assert.Equal(JinjaTokenType.StringLiteral, tokens[1].Type);
        Assert.Equal("he said \"hi\"", tokens[1].Value);
    }

    [Fact]
    public void IntLiteral()
    {
        var tokens = Lex("{{ 42 }}");
        Assert.Equal(JinjaTokenType.IntLiteral, tokens[1].Type);
        Assert.Equal(42, tokens[1].Value);
    }

    [Fact]
    public void BoolLiterals()
    {
        var tokens = Lex("{{ true }}{{ false }}");
        Assert.Equal(JinjaTokenType.BoolLiteral, tokens[1].Type);
        Assert.Equal(true, tokens[1].Value);
        Assert.Equal(JinjaTokenType.BoolLiteral, tokens[4].Type);
        Assert.Equal(false, tokens[4].Value);
    }

    [Fact]
    public void NoneLiteral()
    {
        var tokens = Lex("{{ none }}");
        Assert.Equal(JinjaTokenType.NoneLiteral, tokens[1].Type);
    }

    [Fact]
    public void AllOperators()
    {
        var tokens = Lex("{% == != < > <= >= + - * / % ~ %}");
        var types = tokens.Select(t => t.Type).ToList();
        Assert.Contains(JinjaTokenType.Eq, types);
        Assert.Contains(JinjaTokenType.Ne, types);
        Assert.Contains(JinjaTokenType.Lt, types);
        Assert.Contains(JinjaTokenType.Gt, types);
        Assert.Contains(JinjaTokenType.Le, types);
        Assert.Contains(JinjaTokenType.Ge, types);
        Assert.Contains(JinjaTokenType.Plus, types);
        Assert.Contains(JinjaTokenType.Minus, types);
        Assert.Contains(JinjaTokenType.Multiply, types);
        Assert.Contains(JinjaTokenType.Divide, types);
        Assert.Contains(JinjaTokenType.Modulo, types);
        Assert.Contains(JinjaTokenType.Tilde, types);
    }

    [Fact]
    public void Punctuation()
    {
        var tokens = Lex("{{ a.b[c](d)|e,f = g }}");
        var types = tokens.Select(t => t.Type).ToList();
        Assert.Contains(JinjaTokenType.Dot, types);
        Assert.Contains(JinjaTokenType.LeftBracket, types);
        Assert.Contains(JinjaTokenType.RightBracket, types);
        Assert.Contains(JinjaTokenType.LeftParen, types);
        Assert.Contains(JinjaTokenType.RightParen, types);
        Assert.Contains(JinjaTokenType.Pipe, types);
        Assert.Contains(JinjaTokenType.Comma, types);
        Assert.Contains(JinjaTokenType.Assign, types);
    }

    [Fact]
    public void Keywords()
    {
        var tokens = Lex("{% for x in items %}{% endfor %}");
        var types = tokens.Select(t => t.Type).ToList();
        Assert.Contains(JinjaTokenType.For, types);
        Assert.Contains(JinjaTokenType.In, types);
        Assert.Contains(JinjaTokenType.Endfor, types);
    }

    [Fact]
    public void LogicalKeywords()
    {
        var tokens = Lex("{% if x and y or not z %}{% endif %}");
        var types = tokens.Select(t => t.Type).ToList();
        Assert.Contains(JinjaTokenType.And, types);
        Assert.Contains(JinjaTokenType.Or, types);
        Assert.Contains(JinjaTokenType.Not, types);
    }

    [Fact]
    public void StatementWhitespaceControl()
    {
        var tokens = Lex("  \n  {%- if x -%}  \n  ");
        // {%- strips trailing whitespace from preceding text
        // -%} strips leading whitespace from following text
        var textTokens = tokens.Where(t => t.Type == JinjaTokenType.Text).ToList();
        Assert.Empty(textTokens);
    }

    [Fact]
    public void MixedTextAndTags()
    {
        var tokens = Lex("Hello {{ name }}, welcome to {% if place %}{{ place }}{% endif %}!");
        var types = tokens.Select(t => t.Type).ToList();
        // Verify structure: Text, ExprStart..ExprEnd, Text, StmtStart..StmtEnd, ExprStart..ExprEnd, StmtStart..StmtEnd, Text
        Assert.Equal(JinjaTokenType.Text, types[0]); // "Hello "
        Assert.Equal(JinjaTokenType.ExprStart, types[1]);
    }

    [Fact]
    public void TrueAndFalse_PythonCase()
    {
        var tokens = Lex("{{ True }}{{ False }}");
        Assert.Equal(JinjaTokenType.BoolLiteral, tokens[1].Type);
        Assert.Equal(true, tokens[1].Value);
        Assert.Equal(JinjaTokenType.BoolLiteral, tokens[4].Type);
        Assert.Equal(false, tokens[4].Value);
    }

    [Fact]
    public void NoneKeyword()
    {
        var tokensLower = Lex("{{ none }}");
        Assert.Equal(JinjaTokenType.NoneLiteral, tokensLower[1].Type);

        var tokensUpper = Lex("{{ None }}");
        Assert.Equal(JinjaTokenType.NoneLiteral, tokensUpper[1].Type);
    }
}
