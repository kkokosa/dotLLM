using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

public class JinjaParserTests
{
    private static JinjaTemplate Parse(string source)
    {
        var tokens = new JinjaLexer(source).Tokenize();
        return new JinjaParser(tokens).Parse();
    }

    [Fact]
    public void PlainText_ParsesAsTextNode()
    {
        var ast = Parse("Hello!");
        Assert.Single(ast.Nodes);
        Assert.IsType<TextNode>(ast.Nodes[0]);
        Assert.Equal("Hello!", ((TextNode)ast.Nodes[0]).Text);
    }

    [Fact]
    public void Expression_ParsesAsExpressionOutputNode()
    {
        var ast = Parse("{{ name }}");
        Assert.Single(ast.Nodes);
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        Assert.IsType<IdentifierExpr>(output.Expression);
        Assert.Equal("name", ((IdentifierExpr)output.Expression).Name);
    }

    [Fact]
    public void MemberAccess_DotNotation()
    {
        var ast = Parse("{{ message.role }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var member = Assert.IsType<MemberAccessExpr>(output.Expression);
        Assert.Equal("role", member.Member);
        Assert.IsType<IdentifierExpr>(member.Object);
    }

    [Fact]
    public void MemberAccess_BracketWithStringLiteral_NormalizedToMemberAccess()
    {
        var ast = Parse("{{ message['role'] }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        // String-keyed bracket access is normalized to MemberAccessExpr by the parser
        var member = Assert.IsType<MemberAccessExpr>(output.Expression);
        Assert.Equal("role", member.Member);
    }

    [Fact]
    public void IndexAccess_NumericIndex()
    {
        var ast = Parse("{{ items[0] }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var index = Assert.IsType<IndexAccessExpr>(output.Expression);
        Assert.IsType<LiteralExpr>(index.Index);
        Assert.Equal(0, ((LiteralExpr)index.Index).Value);
    }

    [Fact]
    public void OperatorPrecedence_AdditionBeforeComparison()
    {
        var ast = Parse("{{ a + b == c }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        // Should parse as (a + b) == c
        var eq = Assert.IsType<BinaryExpr>(output.Expression);
        Assert.Equal(BinaryOp.Eq, eq.Op);
        var add = Assert.IsType<BinaryExpr>(eq.Left);
        Assert.Equal(BinaryOp.Add, add.Op);
    }

    [Fact]
    public void ForStatement_BasicLoop()
    {
        var ast = Parse("{% for item in items %}{{ item }}{% endfor %}");
        var forNode = Assert.IsType<ForNode>(ast.Nodes[0]);
        Assert.Equal("item", forNode.VariableName);
        Assert.Null(forNode.SecondVariableName);
        Assert.IsType<IdentifierExpr>(forNode.Iterable);
        Assert.Single(forNode.Body);
    }

    [Fact]
    public void ForStatement_TupleUnpacking()
    {
        var ast = Parse("{% for key, value in items %}{{ key }}{% endfor %}");
        var forNode = Assert.IsType<ForNode>(ast.Nodes[0]);
        Assert.Equal("key", forNode.VariableName);
        Assert.Equal("value", forNode.SecondVariableName);
    }

    [Fact]
    public void IfStatement_SimpleCondition()
    {
        var ast = Parse("{% if x %}yes{% endif %}");
        var ifNode = Assert.IsType<IfNode>(ast.Nodes[0]);
        Assert.Single(ifNode.Branches);
        Assert.Null(ifNode.ElseBody);
    }

    [Fact]
    public void IfStatement_WithElse()
    {
        var ast = Parse("{% if x %}yes{% else %}no{% endif %}");
        var ifNode = Assert.IsType<IfNode>(ast.Nodes[0]);
        Assert.Single(ifNode.Branches);
        Assert.NotNull(ifNode.ElseBody);
    }

    [Fact]
    public void IfStatement_WithElifAndElse()
    {
        var ast = Parse("{% if a %}1{% elif b %}2{% elif c %}3{% else %}4{% endif %}");
        var ifNode = Assert.IsType<IfNode>(ast.Nodes[0]);
        Assert.Equal(3, ifNode.Branches.Count);
        Assert.NotNull(ifNode.ElseBody);
    }

    [Fact]
    public void SetStatement_SimpleAssignment()
    {
        var ast = Parse("{% set x = 42 %}");
        var setNode = Assert.IsType<SetNode>(ast.Nodes[0]);
        Assert.Equal("x", setNode.Name);
        Assert.IsType<LiteralExpr>(setNode.Value);
    }

    [Fact]
    public void SetAttribute_NamespaceMutation()
    {
        var ast = Parse("{% set ns.found = true %}");
        var setAttr = Assert.IsType<SetAttributeNode>(ast.Nodes[0]);
        Assert.Equal("ns", setAttr.ObjectName);
        Assert.Equal("found", setAttr.AttributeName);
    }

    [Fact]
    public void FilterChain()
    {
        var ast = Parse("{{ x | trim | tojson }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        // Outer filter is tojson, inner is trim
        var outerFilter = Assert.IsType<FilterExpr>(output.Expression);
        Assert.Equal("tojson", outerFilter.FilterName);
        var innerFilter = Assert.IsType<FilterExpr>(outerFilter.Input);
        Assert.Equal("trim", innerFilter.FilterName);
    }

    [Fact]
    public void Ternary_IfElseExpression()
    {
        var ast = Parse("{{ 'yes' if condition else 'no' }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var cond = Assert.IsType<ConditionalExpr>(output.Expression);
        Assert.IsType<LiteralExpr>(cond.TrueValue);
        Assert.IsType<IdentifierExpr>(cond.Condition);
        Assert.IsType<LiteralExpr>(cond.FalseValue);
    }

    [Fact]
    public void FunctionCall_WithArgs()
    {
        var ast = Parse("{{ raise_exception('error') }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var call = Assert.IsType<FunctionCallExpr>(output.Expression);
        Assert.Equal("raise_exception", call.Name);
        Assert.Single(call.Args);
    }

    [Fact]
    public void FunctionCall_NamedArgs()
    {
        var ast = Parse("{{ namespace(found=false) }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var call = Assert.IsType<FunctionCallExpr>(output.Expression);
        Assert.Equal("namespace", call.Name);
        Assert.Single(call.Args);
        Assert.Equal("found", call.Args[0].Name);
    }

    [Fact]
    public void IsDefined_Test()
    {
        var ast = Parse("{{ x is defined }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var isDef = Assert.IsType<IsDefinedExpr>(output.Expression);
        Assert.False(isDef.Negated);
    }

    [Fact]
    public void IsNotDefined_Test()
    {
        var ast = Parse("{{ x is not defined }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var isDef = Assert.IsType<IsDefinedExpr>(output.Expression);
        Assert.True(isDef.Negated);
    }

    [Fact]
    public void ListLiteral()
    {
        var ast = Parse("{{ [1, 2, 3] }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var list = Assert.IsType<ListExpr>(output.Expression);
        Assert.Equal(3, list.Items.Count);
    }

    [Fact]
    public void MethodCall_OnExpression()
    {
        var ast = Parse("{{ text.strip() }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var method = Assert.IsType<MethodCallExpr>(output.Expression);
        Assert.Equal("strip", method.MethodName);
        Assert.Empty(method.Args);
    }

    [Fact]
    public void UnaryNot()
    {
        var ast = Parse("{{ not x }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var unary = Assert.IsType<UnaryExpr>(output.Expression);
        Assert.Equal(UnaryOp.Not, unary.Op);
    }

    [Fact]
    public void UnaryNegate()
    {
        var ast = Parse("{{ -1 }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var unary = Assert.IsType<UnaryExpr>(output.Expression);
        Assert.Equal(UnaryOp.Negate, unary.Op);
    }

    [Fact]
    public void StringConcatenation_Tilde()
    {
        var ast = Parse("{{ 'a' ~ 'b' }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var binary = Assert.IsType<BinaryExpr>(output.Expression);
        Assert.Equal(BinaryOp.Concat, binary.Op);
    }

    [Fact]
    public void InOperator()
    {
        var ast = Parse("{{ x in items }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var binary = Assert.IsType<BinaryExpr>(output.Expression);
        Assert.Equal(BinaryOp.In, binary.Op);
    }

    [Fact]
    public void FilterWithArgs()
    {
        var ast = Parse("{{ x | default('fallback') }}");
        var output = Assert.IsType<ExpressionOutputNode>(ast.Nodes[0]);
        var filter = Assert.IsType<FilterExpr>(output.Expression);
        Assert.Equal("default", filter.FilterName);
        Assert.Single(filter.Args);
    }
}
