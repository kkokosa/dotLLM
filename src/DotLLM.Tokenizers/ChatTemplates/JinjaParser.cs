namespace DotLLM.Tokenizers.ChatTemplates;

/// <summary>
/// Recursive-descent parser that transforms a JinjaLexer token list into an AST.
/// Handles template nodes (text, {{ }}, {% %}) and expression parsing with precedence climbing.
/// </summary>
internal sealed class JinjaParser
{
    private readonly List<JinjaToken> _tokens;
    private int _pos;

    public JinjaParser(List<JinjaToken> tokens)
    {
        _tokens = tokens;
    }

    public JinjaTemplate Parse()
    {
        _pos = 0;
        var nodes = ParseBody();
        return new JinjaTemplate(nodes);
    }

    // ── Template-level parsing ──

    private List<ITemplateNode> ParseBody(params JinjaTokenType[] stopAt)
    {
        var nodes = new List<ITemplateNode>();

        while (!IsAtEnd())
        {
            var token = Current();

            // Check stop conditions (endfor, endif, elif, else, etc.)
            if (token.Type == JinjaTokenType.StmtStart && stopAt.Length > 0)
            {
                var next = Peek(1);
                if (next.HasValue && Array.IndexOf(stopAt, next.Value.Type) >= 0)
                    break;
            }

            switch (token.Type)
            {
                case JinjaTokenType.Text:
                    nodes.Add(new TextNode((string)token.Value!));
                    Advance();
                    break;

                case JinjaTokenType.ExprStart:
                    nodes.Add(ParseExpressionOutput());
                    break;

                case JinjaTokenType.StmtStart:
                    nodes.Add(ParseStatement());
                    break;

                default:
                    throw Error($"Unexpected token {token.Type}");
            }
        }

        return nodes;
    }

    private ExpressionOutputNode ParseExpressionOutput()
    {
        Expect(JinjaTokenType.ExprStart);
        var expr = ParseExpression();
        Expect(JinjaTokenType.ExprEnd);
        return new ExpressionOutputNode(expr);
    }

    private ITemplateNode ParseStatement()
    {
        Expect(JinjaTokenType.StmtStart);
        var keyword = Current();

        return keyword.Type switch
        {
            JinjaTokenType.If => ParseIf(),
            JinjaTokenType.For => ParseFor(),
            JinjaTokenType.Set => ParseSet(),
            _ => throw Error($"Unexpected statement keyword: {keyword.Type}")
        };
    }

    private IfNode ParseIf()
    {
        var branches = new List<(IExpression, IReadOnlyList<ITemplateNode>)>();
        List<ITemplateNode>? elseBody = null;

        // Parse {% if condition %}
        Expect(JinjaTokenType.If);
        var condition = ParseExpression();
        Expect(JinjaTokenType.StmtEnd);

        var body = ParseBody(JinjaTokenType.Elif, JinjaTokenType.Else, JinjaTokenType.Endif);
        branches.Add((condition, body));

        // Parse {% elif %} chains
        while (CurrentIs(JinjaTokenType.StmtStart) && PeekIs(1, JinjaTokenType.Elif))
        {
            Expect(JinjaTokenType.StmtStart);
            Expect(JinjaTokenType.Elif);
            condition = ParseExpression();
            Expect(JinjaTokenType.StmtEnd);

            body = ParseBody(JinjaTokenType.Elif, JinjaTokenType.Else, JinjaTokenType.Endif);
            branches.Add((condition, body));
        }

        // Parse optional {% else %}
        if (CurrentIs(JinjaTokenType.StmtStart) && PeekIs(1, JinjaTokenType.Else))
        {
            Expect(JinjaTokenType.StmtStart);
            Expect(JinjaTokenType.Else);
            Expect(JinjaTokenType.StmtEnd);
            elseBody = ParseBody(JinjaTokenType.Endif);
        }

        // Parse {% endif %}
        Expect(JinjaTokenType.StmtStart);
        Expect(JinjaTokenType.Endif);
        Expect(JinjaTokenType.StmtEnd);

        return new IfNode(branches, elseBody);
    }

    private ForNode ParseFor()
    {
        Expect(JinjaTokenType.For);

        string varName = ExpectIdentifier();
        string? secondVar = null;

        // Check for tuple unpacking: "for key, value in items"
        if (CurrentIs(JinjaTokenType.Comma))
        {
            Advance();
            secondVar = ExpectIdentifier();
        }

        Expect(JinjaTokenType.In);
        var iterable = ParseExpression();
        // Handle optional filter on the iterable (e.g., "for x in items | reverse")
        // Filters are already handled in ParseExpression -> ParsePostfix
        Expect(JinjaTokenType.StmtEnd);

        // Check for optional recursive keyword - we don't support it but skip gracefully

        var body = ParseBody(JinjaTokenType.Endfor, JinjaTokenType.Else);

        List<ITemplateNode>? elseBody = null;
        if (CurrentIs(JinjaTokenType.StmtStart) && PeekIs(1, JinjaTokenType.Else))
        {
            Expect(JinjaTokenType.StmtStart);
            Expect(JinjaTokenType.Else);
            Expect(JinjaTokenType.StmtEnd);
            elseBody = ParseBody(JinjaTokenType.Endfor);
        }

        Expect(JinjaTokenType.StmtStart);
        Expect(JinjaTokenType.Endfor);
        Expect(JinjaTokenType.StmtEnd);

        return new ForNode(varName, secondVar, iterable, body, elseBody);
    }

    private ITemplateNode ParseSet()
    {
        Expect(JinjaTokenType.Set);

        string name = ExpectIdentifier();

        // Check for attribute set: {% set ns.key = value %}
        if (CurrentIs(JinjaTokenType.Dot))
        {
            Advance();
            string attr = ExpectIdentifier();
            Expect(JinjaTokenType.Assign);
            var value = ParseExpression();
            Expect(JinjaTokenType.StmtEnd);
            return new SetAttributeNode(name, attr, value);
        }

        Expect(JinjaTokenType.Assign);
        var expr = ParseExpression();
        Expect(JinjaTokenType.StmtEnd);
        return new SetNode(name, expr);
    }

    // ── Expression parsing with precedence climbing ──

    private IExpression ParseExpression() => ParseConditional();

    /// <summary>
    /// conditional → or_expr (IF or_expr ELSE conditional)?
    /// Jinja2 ternary: value_if_true if condition else value_if_false
    /// </summary>
    private IExpression ParseConditional()
    {
        var expr = ParseOr();

        if (CurrentIs(JinjaTokenType.If))
        {
            Advance();
            var condition = ParseOr();
            Expect(JinjaTokenType.Else);
            var falseValue = ParseConditional();
            return new ConditionalExpr(expr, condition, falseValue);
        }

        return expr;
    }

    private IExpression ParseOr()
    {
        var left = ParseAnd();
        while (CurrentIs(JinjaTokenType.Or))
        {
            Advance();
            var right = ParseAnd();
            left = new BinaryExpr(left, BinaryOp.Or, right);
        }
        return left;
    }

    private IExpression ParseAnd()
    {
        var left = ParseNot();
        while (CurrentIs(JinjaTokenType.And))
        {
            Advance();
            var right = ParseNot();
            left = new BinaryExpr(left, BinaryOp.And, right);
        }
        return left;
    }

    private IExpression ParseNot()
    {
        if (CurrentIs(JinjaTokenType.Not))
        {
            Advance();
            var operand = ParseNot();
            return new UnaryExpr(UnaryOp.Not, operand);
        }
        return ParseComparison();
    }

    private IExpression ParseComparison()
    {
        var left = ParseAddition();

        while (true)
        {
            BinaryOp? op = Current().Type switch
            {
                JinjaTokenType.Eq => BinaryOp.Eq,
                JinjaTokenType.Ne => BinaryOp.Ne,
                JinjaTokenType.Lt => BinaryOp.Lt,
                JinjaTokenType.Gt => BinaryOp.Gt,
                JinjaTokenType.Le => BinaryOp.Le,
                JinjaTokenType.Ge => BinaryOp.Ge,
                JinjaTokenType.In => BinaryOp.In,
                _ => null
            };

            if (op == null)
            {
                // Check for "is (not) defined/none/mapping/..."
                if (CurrentIs(JinjaTokenType.Is))
                {
                    Advance();
                    bool negated = false;
                    if (CurrentIs(JinjaTokenType.Not))
                    {
                        negated = true;
                        Advance();
                    }

                    if (CurrentIs(JinjaTokenType.Defined))
                    {
                        Advance();
                        return new IsDefinedExpr(left, negated);
                    }

                    if (CurrentIs(JinjaTokenType.NoneLiteral))
                    {
                        Advance();
                        return new IsTestExpr(left, "none", negated);
                    }

                    if (CurrentIs(JinjaTokenType.BoolLiteral))
                    {
                        bool boolVal = (bool)Current().Value!;
                        Advance();
                        return new IsTestExpr(left, boolVal ? "true" : "false", negated);
                    }

                    if (CurrentIs(JinjaTokenType.Identifier))
                    {
                        string testName = (string)Current().Value!;
                        Advance();
                        return new IsTestExpr(left, testName, negated);
                    }

                    throw Error("Expected test name after 'is'");
                }

                // Check for "not in"
                if (CurrentIs(JinjaTokenType.Not) && PeekIs(1, JinjaTokenType.In))
                {
                    Advance(); // not
                    Advance(); // in
                    var right = ParseAddition();
                    return new UnaryExpr(UnaryOp.Not, new BinaryExpr(left, BinaryOp.In, right));
                }

                break;
            }

            Advance();
            var rhs = ParseAddition();
            left = new BinaryExpr(left, op.Value, rhs);
        }

        return left;
    }

    private IExpression ParseAddition()
    {
        var left = ParseMultiplication();

        while (true)
        {
            BinaryOp? op = Current().Type switch
            {
                JinjaTokenType.Plus => BinaryOp.Add,
                JinjaTokenType.Minus => BinaryOp.Subtract,
                JinjaTokenType.Tilde => BinaryOp.Concat,
                _ => null
            };

            if (op == null) break;
            Advance();
            var right = ParseMultiplication();
            left = new BinaryExpr(left, op.Value, right);
        }

        return left;
    }

    private IExpression ParseMultiplication()
    {
        var left = ParseUnary();

        while (true)
        {
            BinaryOp? op = Current().Type switch
            {
                JinjaTokenType.Multiply => BinaryOp.Multiply,
                JinjaTokenType.Divide => BinaryOp.Divide,
                JinjaTokenType.Modulo => BinaryOp.Modulo,
                _ => null
            };

            if (op == null) break;
            Advance();
            var right = ParseUnary();
            left = new BinaryExpr(left, op.Value, right);
        }

        return left;
    }

    private IExpression ParseUnary()
    {
        if (CurrentIs(JinjaTokenType.Minus))
        {
            Advance();
            var operand = ParseUnary();
            return new UnaryExpr(UnaryOp.Negate, operand);
        }

        return ParsePostfix();
    }

    private IExpression ParsePostfix()
    {
        var expr = ParsePrimary();

        while (true)
        {
            // Member access: .ident
            if (CurrentIs(JinjaTokenType.Dot))
            {
                Advance();
                string member = ExpectIdentifier();

                // Check for method call: .method(args)
                if (CurrentIs(JinjaTokenType.LeftParen))
                {
                    Advance();
                    var args = ParseArgList();
                    Expect(JinjaTokenType.RightParen);
                    expr = new MethodCallExpr(expr, member, args);
                }
                else
                {
                    expr = new MemberAccessExpr(expr, member);
                }
                continue;
            }

            // Index/bracket access: [expr] or slice: [start:stop]
            if (CurrentIs(JinjaTokenType.LeftBracket))
            {
                Advance();

                IExpression? start = null;
                IExpression? stop = null;
                bool isSlice = false;

                if (CurrentIs(JinjaTokenType.Colon))
                {
                    // [:stop] or [:]
                    isSlice = true;
                    Advance();
                    if (!CurrentIs(JinjaTokenType.RightBracket))
                        stop = ParseExpression();
                }
                else
                {
                    start = ParseExpression();
                    if (CurrentIs(JinjaTokenType.Colon))
                    {
                        // [start:stop] or [start:]
                        isSlice = true;
                        Advance();
                        if (!CurrentIs(JinjaTokenType.RightBracket))
                            stop = ParseExpression();
                    }
                }

                Expect(JinjaTokenType.RightBracket);

                if (isSlice)
                {
                    expr = new SliceExpr(expr, start, stop);
                }
                else
                {
                    // Normalize string-keyed bracket access to member access
                    if (start is LiteralExpr { Value: string key })
                        expr = new MemberAccessExpr(expr, key);
                    else
                        expr = new IndexAccessExpr(expr, start!);
                }
                continue;
            }

            // Filter: | filter_name or | filter_name(args)
            if (CurrentIs(JinjaTokenType.Pipe))
            {
                Advance();
                string filterName = ExpectIdentifier();
                var args = new List<IExpression>();

                if (CurrentIs(JinjaTokenType.LeftParen))
                {
                    Advance();
                    args = ParseArgList();
                    Expect(JinjaTokenType.RightParen);
                }

                expr = new FilterExpr(expr, filterName, args);
                continue;
            }

            break;
        }

        return expr;
    }

    private IExpression ParsePrimary()
    {
        var token = Current();

        switch (token.Type)
        {
            case JinjaTokenType.StringLiteral:
                Advance();
                return new LiteralExpr(token.Value);

            case JinjaTokenType.IntLiteral:
                Advance();
                return new LiteralExpr(token.Value);

            case JinjaTokenType.BoolLiteral:
                Advance();
                return new LiteralExpr(token.Value);

            case JinjaTokenType.NoneLiteral:
                Advance();
                return new LiteralExpr(null);

            case JinjaTokenType.Identifier:
            {
                string name = (string)token.Value!;
                Advance();

                // Function call: name(args)
                if (CurrentIs(JinjaTokenType.LeftParen))
                {
                    Advance();
                    var args = ParseNamedArgList();
                    Expect(JinjaTokenType.RightParen);
                    return new FunctionCallExpr(name, args);
                }

                return new IdentifierExpr(name);
            }

            case JinjaTokenType.Namespace:
            {
                Advance();
                Expect(JinjaTokenType.LeftParen);
                var args = ParseNamedArgList();
                Expect(JinjaTokenType.RightParen);
                return new FunctionCallExpr("namespace", args);
            }

            case JinjaTokenType.LeftParen:
            {
                Advance();
                var expr = ParseExpression();
                Expect(JinjaTokenType.RightParen);
                return expr;
            }

            case JinjaTokenType.LeftBracket:
            {
                Advance();
                var items = new List<IExpression>();
                if (!CurrentIs(JinjaTokenType.RightBracket))
                {
                    items.Add(ParseExpression());
                    while (CurrentIs(JinjaTokenType.Comma))
                    {
                        Advance();
                        if (CurrentIs(JinjaTokenType.RightBracket))
                            break; // trailing comma
                        items.Add(ParseExpression());
                    }
                }
                Expect(JinjaTokenType.RightBracket);
                return new ListExpr(items);
            }

            case JinjaTokenType.LeftBrace:
            {
                Advance();
                var pairs = new List<(IExpression Key, IExpression Value)>();
                if (!CurrentIs(JinjaTokenType.RightBrace))
                {
                    var key = ParseExpression();
                    Expect(JinjaTokenType.Colon);
                    var value = ParseExpression();
                    pairs.Add((key, value));

                    while (CurrentIs(JinjaTokenType.Comma))
                    {
                        Advance();
                        if (CurrentIs(JinjaTokenType.RightBrace))
                            break; // trailing comma
                        key = ParseExpression();
                        Expect(JinjaTokenType.Colon);
                        value = ParseExpression();
                        pairs.Add((key, value));
                    }
                }
                Expect(JinjaTokenType.RightBrace);
                return new DictExpr(pairs);
            }

            // Allow keywords to be used as identifiers in expression context
            // (e.g., message.not, message.or — unlikely but safe)
            case JinjaTokenType.Not:
                // This should be handled by ParseNot before we get here, but as identifier fallback:
                Advance();
                return new UnaryExpr(UnaryOp.Not, ParsePostfix());

            default:
                throw Error($"Expected expression, got {token.Type}");
        }
    }

    private List<IExpression> ParseArgList()
    {
        var args = new List<IExpression>();
        if (!CurrentIs(JinjaTokenType.RightParen))
        {
            args.Add(ParsePosOrKwArg());
            while (CurrentIs(JinjaTokenType.Comma))
            {
                Advance();
                if (CurrentIs(JinjaTokenType.RightParen))
                    break; // trailing comma
                args.Add(ParsePosOrKwArg());
            }
        }
        return args;
    }

    /// <summary>
    /// Parses a positional or keyword argument. For keyword args (name=value),
    /// the name is discarded and the value is returned as a positional arg.
    /// </summary>
    private IExpression ParsePosOrKwArg()
    {
        if (CurrentIs(JinjaTokenType.Identifier) && PeekIs(1, JinjaTokenType.Assign))
        {
            Advance(); // skip name
            Advance(); // skip =
            return ParseExpression();
        }
        return ParseExpression();
    }

    private List<(string? Name, IExpression Value)> ParseNamedArgList()
    {
        var args = new List<(string?, IExpression)>();
        if (CurrentIs(JinjaTokenType.RightParen))
            return args;

        args.Add(ParseNamedArg());
        while (CurrentIs(JinjaTokenType.Comma))
        {
            Advance();
            if (CurrentIs(JinjaTokenType.RightParen))
                break; // trailing comma
            args.Add(ParseNamedArg());
        }
        return args;
    }

    private (string? Name, IExpression Value) ParseNamedArg()
    {
        // Check if this is name=value or just a positional expression
        if (CurrentIs(JinjaTokenType.Identifier) && PeekIs(1, JinjaTokenType.Assign))
        {
            string name = (string)Current().Value!;
            Advance(); // identifier
            Advance(); // =
            var value = ParseExpression();
            return (name, value);
        }

        return (null, ParseExpression());
    }

    // ── Helpers ──

    private JinjaToken Current() =>
        _pos < _tokens.Count ? _tokens[_pos] : new JinjaToken(JinjaTokenType.Eof, null, 0, 0);

    private JinjaToken? Peek(int offset)
    {
        int idx = _pos + offset;
        return idx < _tokens.Count ? _tokens[idx] : null;
    }

    private bool CurrentIs(JinjaTokenType type) => Current().Type == type;

    private bool PeekIs(int offset, JinjaTokenType type) => Peek(offset)?.Type == type;

    private bool IsAtEnd() => Current().Type == JinjaTokenType.Eof;

    private void Advance() => _pos++;

    private void Expect(JinjaTokenType type)
    {
        var token = Current();
        if (token.Type != type)
            throw Error($"Expected {type}, got {token.Type}");
        Advance();
    }

    private string ExpectIdentifier()
    {
        var token = Current();
        if (token.Type == JinjaTokenType.Identifier)
        {
            Advance();
            return (string)token.Value!;
        }

        // Allow keywords as identifiers in certain contexts (e.g., dict keys, attribute names)
        if (IsKeyword(token.Type))
        {
            Advance();
            return token.Type.ToString().ToLowerInvariant();
        }

        throw Error($"Expected identifier, got {token.Type}");
    }

    private static bool IsKeyword(JinjaTokenType type) => type switch
    {
        JinjaTokenType.If or JinjaTokenType.Else or JinjaTokenType.Elif or
        JinjaTokenType.Endif or JinjaTokenType.For or JinjaTokenType.In or
        JinjaTokenType.Endfor or JinjaTokenType.Set or JinjaTokenType.Not or
        JinjaTokenType.And or JinjaTokenType.Or or JinjaTokenType.Is or
        JinjaTokenType.Defined or JinjaTokenType.Namespace => true,
        _ => false
    };

    private JinjaException Error(string message)
    {
        var token = Current();
        return new JinjaException(message, token.Line, token.Column);
    }
}
