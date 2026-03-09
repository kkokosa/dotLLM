namespace DotLLM.Tokenizers.ChatTemplates;

// ── Template Nodes (statements that produce output or control flow) ──

internal interface ITemplateNode { }

internal sealed record TextNode(string Text) : ITemplateNode;

internal sealed record ExpressionOutputNode(IExpression Expression) : ITemplateNode;

internal sealed record ForNode(
    string VariableName,
    string? SecondVariableName, // for "key, value in dict" unpacking
    IExpression Iterable,
    IReadOnlyList<ITemplateNode> Body,
    IReadOnlyList<ITemplateNode>? ElseBody) : ITemplateNode;

internal sealed record IfNode(
    IReadOnlyList<(IExpression Condition, IReadOnlyList<ITemplateNode> Body)> Branches,
    IReadOnlyList<ITemplateNode>? ElseBody) : ITemplateNode;

internal sealed record SetNode(string Name, IExpression Value) : ITemplateNode;

internal sealed record SetAttributeNode(
    string ObjectName, string AttributeName, IExpression Value) : ITemplateNode;

// ── Root ──

internal sealed record JinjaTemplate(IReadOnlyList<ITemplateNode> Nodes);

// ── Expression Nodes ──

internal interface IExpression { }

internal sealed record LiteralExpr(object? Value) : IExpression;

internal sealed record IdentifierExpr(string Name) : IExpression;

internal sealed record MemberAccessExpr(IExpression Object, string Member) : IExpression;

internal sealed record IndexAccessExpr(IExpression Object, IExpression Index) : IExpression;

internal sealed record BinaryExpr(IExpression Left, BinaryOp Op, IExpression Right) : IExpression;

internal sealed record UnaryExpr(UnaryOp Op, IExpression Operand) : IExpression;

internal sealed record FilterExpr(IExpression Input, string FilterName, IReadOnlyList<IExpression> Args) : IExpression;

internal sealed record ConditionalExpr(IExpression TrueValue, IExpression Condition, IExpression FalseValue) : IExpression;

internal sealed record FunctionCallExpr(string Name, IReadOnlyList<(string? Name, IExpression Value)> Args) : IExpression;

internal sealed record MethodCallExpr(IExpression Object, string MethodName, IReadOnlyList<IExpression> Args) : IExpression;

internal sealed record IsDefinedExpr(IExpression Target, bool Negated) : IExpression;

internal sealed record IsTestExpr(IExpression Target, string TestName, bool Negated) : IExpression;

internal sealed record ListExpr(IReadOnlyList<IExpression> Items) : IExpression;

internal sealed record DictExpr(IReadOnlyList<(IExpression Key, IExpression Value)> Pairs) : IExpression;

internal sealed record SliceExpr(IExpression Object, IExpression? Start, IExpression? Stop) : IExpression;

// ── Operators ──

internal enum BinaryOp
{
    Add, Subtract, Multiply, Divide, Modulo,
    Eq, Ne, Lt, Gt, Le, Ge,
    And, Or,
    In,
    Concat // ~ (string concatenation)
}

internal enum UnaryOp
{
    Not, Negate
}
