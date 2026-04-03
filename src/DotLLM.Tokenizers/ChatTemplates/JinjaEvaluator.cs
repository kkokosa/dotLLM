using System.Collections;
using System.Text;

namespace DotLLM.Tokenizers.ChatTemplates;

/// <summary>
/// Tree-walking evaluator: executes a JinjaTemplate AST against a variable context
/// and produces the rendered output string.
/// </summary>
internal sealed class JinjaEvaluator
{
    /// <summary>Sentinel value representing an undefined variable (distinct from null).</summary>
    internal static readonly object Undefined = new();

    private readonly List<Dictionary<string, object?>> _scopes = new();
    private readonly StringBuilder _output = new();

    private static readonly Dictionary<string, Func<object?, IReadOnlyList<object?>, object?>> Filters = new()
    {
        ["trim"] = (input, _) => input?.ToString()?.Trim() ?? "",
        ["tojson"] = (input, args) => ToJson(input, args.Count > 0 ? ToInt(args[0]) : 0),
        ["length"] = (input, _) => GetLength(input),
        ["default"] = (input, args) =>
        {
            if (input is null || ReferenceEquals(input, Undefined))
                return args.Count > 0 ? args[0] : "";
            return input;
        },
        ["upper"] = (input, _) => input?.ToString()?.ToUpperInvariant() ?? "",
        ["lower"] = (input, _) => input?.ToString()?.ToLowerInvariant() ?? "",
        ["first"] = (input, _) => GetFirst(input),
        ["last"] = (input, _) => GetLast(input),
        ["string"] = (input, _) => Stringify(input),
        ["list"] = (input, _) => input is IList list ? list : input is IEnumerable e ? ToList(e) : new List<object?> { input },
        ["items"] = (input, _) => GetDictItems(input),
        ["join"] = (input, args) => JoinFilter(input, args),
        ["replace"] = (input, args) => ReplaceFilter(input, args),
        ["selectattr"] = (input, args) => SelectAttrFilter(input, args),
        ["map"] = (input, args) => MapFilter(input, args),
        ["safe"] = (input, _) => input, // no-op: marks string as HTML-safe (irrelevant for LLM templates)
    };

    public JinjaEvaluator(Dictionary<string, object?> context)
    {
        _scopes.Add(context);
    }

    public string Evaluate(JinjaTemplate template)
    {
        _output.Clear();
        foreach (var node in template.Nodes)
            EvaluateNode(node);
        return _output.ToString();
    }

    // ── Node evaluation ──

    private void EvaluateNode(ITemplateNode node)
    {
        switch (node)
        {
            case TextNode text:
                _output.Append(text.Text);
                break;

            case ExpressionOutputNode exprOut:
                var value = EvalExpr(exprOut.Expression);
                _output.Append(Stringify(value));
                break;

            case ForNode forNode:
                EvaluateFor(forNode);
                break;

            case IfNode ifNode:
                EvaluateIf(ifNode);
                break;

            case SetNode setNode:
                SetVariable(setNode.Name, EvalExpr(setNode.Value));
                break;

            case SetAttributeNode setAttr:
                EvaluateSetAttribute(setAttr);
                break;
        }
    }

    private void EvaluateFor(ForNode node)
    {
        var iterableValue = EvalExpr(node.Iterable);
        var items = ToEnumerable(iterableValue);
        var itemList = items.ToList();

        if (itemList.Count == 0)
        {
            if (node.ElseBody != null)
                foreach (var n in node.ElseBody)
                    EvaluateNode(n);
            return;
        }

        int length = itemList.Count;
        for (int i = 0; i < length; i++)
        {
            PushScope();

            var item = itemList[i];

            // Handle tuple unpacking for dict iteration
            if (node.SecondVariableName != null && item is KeyValuePair<string, object?> kvp)
            {
                SetVariable(node.VariableName, kvp.Key);
                SetVariable(node.SecondVariableName, kvp.Value);
            }
            else if (node.SecondVariableName != null && item is IList tupleList && tupleList.Count >= 2)
            {
                SetVariable(node.VariableName, tupleList[0]);
                SetVariable(node.SecondVariableName, tupleList[1]);
            }
            else
            {
                SetVariable(node.VariableName, item);
            }

            // Set loop variable
            var loopVar = new Dictionary<string, object?>
            {
                ["index"] = i + 1,
                ["index0"] = i,
                ["first"] = i == 0,
                ["last"] = i == length - 1,
                ["length"] = length,
                ["revindex"] = length - i,
                ["revindex0"] = length - i - 1,
            };
            SetVariable("loop", loopVar);

            foreach (var bodyNode in node.Body)
                EvaluateNode(bodyNode);

            PopScope();
        }
    }

    private void EvaluateIf(IfNode node)
    {
        foreach (var (condition, body) in node.Branches)
        {
            if (IsTruthy(EvalExpr(condition)))
            {
                foreach (var n in body)
                    EvaluateNode(n);
                return;
            }
        }

        if (node.ElseBody != null)
            foreach (var n in node.ElseBody)
                EvaluateNode(n);
    }

    private void EvaluateSetAttribute(SetAttributeNode node)
    {
        var obj = LookupVariable(node.ObjectName);
        if (obj is Dictionary<string, object?> dict)
        {
            dict[node.AttributeName] = EvalExpr(node.Value);
        }
        else
        {
            throw new JinjaException($"Cannot set attribute on non-object: {node.ObjectName}");
        }
    }

    // ── Expression evaluation ──

    private object? EvalExpr(IExpression expr)
    {
        switch (expr)
        {
            case LiteralExpr lit:
                return lit.Value;

            case IdentifierExpr ident:
                return LookupVariable(ident.Name);

            case MemberAccessExpr member:
                return EvalMemberAccess(member);

            case IndexAccessExpr index:
                return EvalIndexAccess(index);

            case BinaryExpr binary:
                return EvalBinary(binary);

            case UnaryExpr unary:
                return EvalUnary(unary);

            case FilterExpr filter:
                return EvalFilter(filter);

            case ConditionalExpr cond:
                return IsTruthy(EvalExpr(cond.Condition))
                    ? EvalExpr(cond.TrueValue)
                    : EvalExpr(cond.FalseValue);

            case FunctionCallExpr func:
                return EvalFunctionCall(func);

            case MethodCallExpr method:
                return EvalMethodCall(method);

            case IsDefinedExpr isDef:
                return EvalIsDefined(isDef);

            case IsTestExpr isTest:
                return EvalIsTest(isTest);

            case ListExpr list:
                return list.Items.Select(EvalExpr).ToList();

            case DictExpr dict:
                var result = new Dictionary<string, object?>();
                foreach (var (key, value) in dict.Pairs)
                    result[Stringify(EvalExpr(key))] = EvalExpr(value);
                return result;

            case SliceExpr slice:
                return EvalSlice(slice);

            default:
                throw new JinjaException($"Unknown expression type: {expr.GetType().Name}");
        }
    }

    private object? EvalMemberAccess(MemberAccessExpr expr)
    {
        var obj = EvalExpr(expr.Object);

        if (obj is Dictionary<string, object?> dict)
        {
            if (dict.TryGetValue(expr.Member, out var val))
                return val;
            return Undefined;
        }

        if (obj is null || ReferenceEquals(obj, Undefined))
            return Undefined;

        // Handle string properties
        if (obj is string s && expr.Member == "length")
            return s.Length;

        // Handle list properties
        if (obj is IList list && expr.Member == "length")
            return list.Count;

        return Undefined;
    }

    private object? EvalIndexAccess(IndexAccessExpr expr)
    {
        var obj = EvalExpr(expr.Object);
        var index = EvalExpr(expr.Index);

        if (obj is IList list && index is int i)
        {
            if (i < 0) i += list.Count; // Python-style negative indexing
            return i >= 0 && i < list.Count ? list[i] : Undefined;
        }

        if (obj is Dictionary<string, object?> dict && index is string key)
        {
            return dict.TryGetValue(key, out var val) ? val : Undefined;
        }

        if (obj is string s && index is int idx)
        {
            if (idx < 0) idx += s.Length;
            return idx >= 0 && idx < s.Length ? s[idx].ToString() : Undefined;
        }

        return Undefined;
    }

    private object? EvalBinary(BinaryExpr expr)
    {
        // Short-circuit for And/Or
        if (expr.Op == BinaryOp.And)
        {
            var left = EvalExpr(expr.Left);
            return !IsTruthy(left) ? left : EvalExpr(expr.Right);
        }

        if (expr.Op == BinaryOp.Or)
        {
            var left = EvalExpr(expr.Left);
            return IsTruthy(left) ? left : EvalExpr(expr.Right);
        }

        var lhs = EvalExpr(expr.Left);
        var rhs = EvalExpr(expr.Right);

        return expr.Op switch
        {
            BinaryOp.Add => Add(lhs, rhs),
            BinaryOp.Subtract => Subtract(lhs, rhs),
            BinaryOp.Multiply => Multiply(lhs, rhs),
            BinaryOp.Divide => Divide(lhs, rhs),
            BinaryOp.Modulo => Modulo(lhs, rhs),
            BinaryOp.Concat => Stringify(lhs) + Stringify(rhs),
            BinaryOp.Eq => Equals(lhs, rhs),
            BinaryOp.Ne => !Equals(lhs, rhs),
            BinaryOp.Lt => Compare(lhs, rhs) < 0,
            BinaryOp.Gt => Compare(lhs, rhs) > 0,
            BinaryOp.Le => Compare(lhs, rhs) <= 0,
            BinaryOp.Ge => Compare(lhs, rhs) >= 0,
            BinaryOp.In => Contains(rhs, lhs),
            _ => throw new JinjaException($"Unknown binary operator: {expr.Op}")
        };
    }

    private object? EvalUnary(UnaryExpr expr)
    {
        var operand = EvalExpr(expr.Operand);
        return expr.Op switch
        {
            UnaryOp.Not => !IsTruthy(operand),
            UnaryOp.Negate => operand is int i ? -i : operand is double d ? -d : throw new JinjaException("Cannot negate non-number"),
            _ => throw new JinjaException($"Unknown unary operator: {expr.Op}")
        };
    }

    private object? EvalFilter(FilterExpr expr)
    {
        var input = EvalExpr(expr.Input);
        var args = expr.Args.Select(EvalExpr).ToList();

        if (Filters.TryGetValue(expr.FilterName, out var filter))
            return filter(input, args);

        throw new JinjaException($"Unknown filter: {expr.FilterName}");
    }

    private object? EvalFunctionCall(FunctionCallExpr expr)
    {
        switch (expr.Name)
        {
            case "raise_exception":
            {
                var msg = expr.Args.Count > 0 ? Stringify(EvalExpr(expr.Args[0].Value)) : "Template error";
                throw new JinjaException(msg);
            }

            case "namespace":
            {
                var ns = new Dictionary<string, object?>();
                foreach (var (name, value) in expr.Args)
                {
                    if (name != null)
                        ns[name] = EvalExpr(value);
                }
                return ns;
            }

            case "range":
            {
                var args = expr.Args.Select(a => EvalExpr(a.Value)).ToList();
                return args.Count switch
                {
                    1 => Enumerable.Range(0, ToInt(args[0])).Cast<object?>().ToList(),
                    2 => Enumerable.Range(ToInt(args[0]), Math.Max(0, ToInt(args[1]) - ToInt(args[0]))).Cast<object?>().ToList(),
                    3 => RangeWithStep(ToInt(args[0]), ToInt(args[1]), ToInt(args[2])),
                    _ => throw new JinjaException("range() takes 1 to 3 arguments")
                };
            }

            case "dict":
            {
                var dict = new Dictionary<string, object?>();
                foreach (var (name, value) in expr.Args)
                {
                    if (name != null)
                        dict[name] = EvalExpr(value);
                }
                return dict;
            }

            case "cycler":
            {
                // Simple cycler implementation - return first arg for now
                return expr.Args.Count > 0 ? EvalExpr(expr.Args[0].Value) : null;
            }

            case "strftime_now":
            {
                var format = expr.Args.Count > 0 ? Stringify(EvalExpr(expr.Args[0].Value)) : "%Y-%m-%d";
                return FormatStrftime(DateTime.UtcNow, format);
            }

            default:
            {
                // Check if it's a variable that's callable (e.g., user-provided function)
                var func = LookupVariable(expr.Name);
                if (func is Func<object?[], object?> callable)
                {
                    var args = expr.Args.Select(a => EvalExpr(a.Value)).ToArray();
                    return callable(args);
                }
                throw new JinjaException($"Unknown function: {expr.Name}");
            }
        }
    }

    private object? EvalMethodCall(MethodCallExpr expr)
    {
        var obj = EvalExpr(expr.Object);
        var args = expr.Args.Select(EvalExpr).ToList();

        if (obj is string s)
        {
            return expr.MethodName switch
            {
                "strip" => s.Trim(),
                "lstrip" => s.TrimStart(),
                "rstrip" => s.TrimEnd(),
                "upper" => s.ToUpperInvariant(),
                "lower" => s.ToLowerInvariant(),
                "startswith" => args.Count > 0 && s.StartsWith(Stringify(args[0]), StringComparison.Ordinal),
                "endswith" => args.Count > 0 && s.EndsWith(Stringify(args[0]), StringComparison.Ordinal),
                "replace" => args.Count >= 2 ? s.Replace(Stringify(args[0]), Stringify(args[1])) : s,
                "split" => args.Count > 0
                    ? s.Split(Stringify(args[0])).Cast<object?>().ToList()
                    : s.Split().Cast<object?>().ToList(),
                "join" => s, // string.join doesn't make sense — pass through
                _ => throw new JinjaException($"Unknown string method: {expr.MethodName}")
            };
        }

        if (obj is IList list)
        {
            return expr.MethodName switch
            {
                "append" => AppendToList(list, args),
                "insert" => InsertIntoList(list, args),
                _ => throw new JinjaException($"Unknown list method: {expr.MethodName}")
            };
        }

        if (obj is Dictionary<string, object?> dict)
        {
            return expr.MethodName switch
            {
                "items" => dict.Select(kvp => (object?)new List<object?> { kvp.Key, kvp.Value }).ToList(),
                "keys" => dict.Keys.Cast<object?>().ToList(),
                "values" => dict.Values.ToList(),
                "get" => args.Count >= 1 && dict.TryGetValue(Stringify(args[0]), out var val) ? val :
                          args.Count >= 2 ? args[1] : null,
                "update" => UpdateDict(dict, args),
                _ => throw new JinjaException($"Unknown dict method: {expr.MethodName}")
            };
        }

        // Fallback: coerce to string for string-like methods
        if (obj is not null && expr.MethodName is "strip" or "lstrip" or "rstrip"
                or "upper" or "lower" or "startswith" or "endswith" or "replace" or "split")
        {
            var str = Stringify(obj);
            return expr.MethodName switch
            {
                "strip" => str.Trim(),
                "lstrip" => str.TrimStart(),
                "rstrip" => str.TrimEnd(),
                "upper" => str.ToUpperInvariant(),
                "lower" => str.ToLowerInvariant(),
                "startswith" => args.Count > 0 && str.StartsWith(Stringify(args[0]), StringComparison.Ordinal),
                "endswith" => args.Count > 0 && str.EndsWith(Stringify(args[0]), StringComparison.Ordinal),
                "replace" => args.Count >= 2 ? str.Replace(Stringify(args[0]), Stringify(args[1])) : str,
                "split" => args.Count > 0
                    ? str.Split(Stringify(args[0])).Cast<object?>().ToList()
                    : (object)str.Split().Cast<object?>().ToList(),
                _ => str,
            };
        }

        throw new JinjaException($"Cannot call method '{expr.MethodName}' on {obj?.GetType().Name ?? "null"}");
    }

    private object? EvalIsDefined(IsDefinedExpr expr)
    {
        // Evaluate target but track if it resolves to Undefined
        var value = EvalExpr(expr.Target);
        bool defined = !ReferenceEquals(value, Undefined);
        return expr.Negated ? !defined : defined;
    }

    private object? EvalIsTest(IsTestExpr expr)
    {
        var value = EvalExpr(expr.Target);
        bool result = expr.TestName switch
        {
            "none" => value is null,
            "mapping" => value is Dictionary<string, object?>,
            "string" => value is string,
            "sequence" => value is IList,
            "number" or "integer" => value is int or double or long,
            "iterable" => value is IEnumerable and not string,
            "true" => value is true,
            "false" => value is false,
            "callable" => false, // we don't support callable test
            _ => throw new JinjaException($"Unknown test: {expr.TestName}")
        };
        return expr.Negated ? !result : result;
    }

    private object? EvalSlice(SliceExpr expr)
    {
        var obj = EvalExpr(expr.Object);
        int? startVal = expr.Start != null ? ToInt(EvalExpr(expr.Start)) : null;
        int? stopVal = expr.Stop != null ? ToInt(EvalExpr(expr.Stop)) : null;

        if (obj is string s)
        {
            int from = startVal ?? 0;
            int to = stopVal ?? s.Length;
            if (from < 0) from = Math.Max(0, s.Length + from);
            if (to < 0) to = Math.Max(0, s.Length + to);
            to = Math.Min(to, s.Length);
            from = Math.Min(from, to);
            return s[from..to];
        }

        if (obj is IList list)
        {
            int from = startVal ?? 0;
            int to = stopVal ?? list.Count;
            if (from < 0) from = Math.Max(0, list.Count + from);
            if (to < 0) to = Math.Max(0, list.Count + to);
            to = Math.Min(to, list.Count);
            from = Math.Min(from, to);
            var sliced = new List<object?>();
            for (int i = from; i < to; i++)
                sliced.Add(list[i]);
            return sliced;
        }

        return Undefined;
    }

    // ── Scope management ──

    private void PushScope() => _scopes.Add(new Dictionary<string, object?>());

    private void PopScope() => _scopes.RemoveAt(_scopes.Count - 1);

    private void SetVariable(string name, object? value)
    {
        _scopes[^1][name] = value;
    }

    private object? LookupVariable(string name)
    {
        for (int i = _scopes.Count - 1; i >= 0; i--)
        {
            if (_scopes[i].TryGetValue(name, out var value))
                return value;
        }
        return Undefined;
    }

    // ── Type helpers ──

    internal static bool IsTruthy(object? value)
    {
        if (value is null || ReferenceEquals(value, Undefined))
            return false;
        if (value is bool b) return b;
        if (value is int i) return i != 0;
        if (value is double d) return d != 0.0;
        if (value is string s) return s.Length > 0;
        if (value is ICollection c) return c.Count > 0;
        return true;
    }

    internal static string Stringify(object? value)
    {
        if (value is null || ReferenceEquals(value, Undefined))
            return "";
        if (value is bool b)
            return b ? "True" : "False";
        if (value is string s)
            return s;
        if (value is int or double or long)
            return value.ToString()!;
        if (value is IList list)
            return "[" + string.Join(", ", list.Cast<object?>().Select(item => StringifyRepr(item))) + "]";
        if (value is Dictionary<string, object?> dict)
            return "{" + string.Join(", ", dict.Select(kvp => $"'{kvp.Key}': {StringifyRepr(kvp.Value)}")) + "}";
        return value.ToString() ?? "";
    }

    private static string StringifyRepr(object? value)
    {
        if (value is string s) return $"'{s}'";
        return Stringify(value);
    }

    private static string ToJson(object? value, int indent = 0)
    {
        if (value is null || ReferenceEquals(value, Undefined))
            return "null";
        if (value is bool b)
            return b ? "true" : "false";
        if (value is int i)
            return i.ToString();
        if (value is double d)
            return d.ToString("G");
        if (value is string s)
            return JsonQuoteString(s);
        if (value is Dictionary<string, object?> dict)
            return SerializeDictToJson(dict, indent, 0);
        if (value is IList list)
            return SerializeListToJson(list, indent, 0);
        // Fallback for unexpected types — avoid reflection-based JsonSerializer.Serialize(object)
        // for Native AOT compatibility. All known Jinja value types are handled above.
        return value.ToString() ?? "null";
    }

    /// <summary>
    /// JSON-escapes a string and wraps it in double quotes (RFC 8259 §7).
    /// Trim/AOT-safe replacement for <c>JsonSerializer.Serialize(string)</c>.
    /// </summary>
    private static string JsonQuoteString(string s)
    {
        // Fast path: most template strings (dict keys like "role", "content", tool names)
        // contain no characters requiring JSON escaping. Avoid StringBuilder allocation.
        var needsEscape = false;
        foreach (var c in s)
        {
            if (c is '"' or '\\' || c < ' ') { needsEscape = true; break; }
        }

        if (!needsEscape)
            return string.Concat("\"", s, "\"");

        var sb = new StringBuilder(s.Length + 8);
        sb.Append('"');
        foreach (var c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < ' ')
                    {
                        // Zero-alloc hex escape — control chars are always 00xx
                        sb.Append("\\u00");
                        sb.Append("0123456789ABCDEF"[(c >> 4) & 0xF]);
                        sb.Append("0123456789ABCDEF"[c & 0xF]);
                    }
                    else
                        sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
        return sb.ToString();
    }

    private static string SerializeDictToJson(Dictionary<string, object?> dict, int indent, int depth)
    {
        if (indent <= 0)
        {
            var sb = new StringBuilder();
            sb.Append('{');
            bool first = true;
            foreach (var kvp in dict)
            {
                if (!first) sb.Append(", ");
                first = false;
                sb.Append(JsonQuoteString(kvp.Key));
                sb.Append(": ");
                sb.Append(ToJson(kvp.Value));
            }
            sb.Append('}');
            return sb.ToString();
        }

        var isb = new StringBuilder();
        isb.Append("{\n");
        string childIndent = new(' ', indent * (depth + 1));
        string closingIndent = new(' ', indent * depth);
        bool ifirst = true;
        foreach (var kvp in dict)
        {
            if (!ifirst) isb.Append(",\n");
            ifirst = false;
            isb.Append(childIndent);
            isb.Append(JsonQuoteString(kvp.Key));
            isb.Append(": ");
            isb.Append(SerializeValueToJson(kvp.Value, indent, depth + 1));
        }
        isb.Append('\n');
        isb.Append(closingIndent);
        isb.Append('}');
        return isb.ToString();
    }

    private static string SerializeListToJson(IList list, int indent, int depth)
    {
        if (indent <= 0)
        {
            var sb = new StringBuilder();
            sb.Append('[');
            for (int i = 0; i < list.Count; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(ToJson(list[i]));
            }
            sb.Append(']');
            return sb.ToString();
        }

        var isb = new StringBuilder();
        isb.Append("[\n");
        string childIndent = new(' ', indent * (depth + 1));
        string closingIndent = new(' ', indent * depth);
        for (int i = 0; i < list.Count; i++)
        {
            if (i > 0) isb.Append(",\n");
            isb.Append(childIndent);
            isb.Append(SerializeValueToJson(list[i], indent, depth + 1));
        }
        isb.Append('\n');
        isb.Append(closingIndent);
        isb.Append(']');
        return isb.ToString();
    }

    private static string SerializeValueToJson(object? value, int indent, int depth)
    {
        if (value is Dictionary<string, object?> dict)
            return SerializeDictToJson(dict, indent, depth);
        if (value is IList list)
            return SerializeListToJson(list, indent, depth);
        return ToJson(value);
    }

    private static new bool Equals(object? a, object? b)
    {
        if (a is null || ReferenceEquals(a, Undefined))
            return b is null || ReferenceEquals(b, Undefined);
        if (b is null || ReferenceEquals(b, Undefined))
            return false;

        // Numeric comparison
        if (a is int ai && b is int bi) return ai == bi;
        if (IsNumber(a) && IsNumber(b)) return ToDouble(a) == ToDouble(b);

        return a.Equals(b);
    }

    private static int Compare(object? a, object? b)
    {
        if (IsNumber(a) && IsNumber(b))
            return ToDouble(a).CompareTo(ToDouble(b));
        if (a is string sa && b is string sb)
            return string.Compare(sa, sb, StringComparison.Ordinal);
        return 0;
    }

    private static bool Contains(object? container, object? item)
    {
        if (container is string s && item is string sub)
            return s.Contains(sub, StringComparison.Ordinal);
        if (container is string s2 && item is not null)
            return s2.Contains(Stringify(item), StringComparison.Ordinal);
        if (container is IList list)
        {
            foreach (var element in list)
                if (Equals(element, item))
                    return true;
            return false;
        }
        if (container is Dictionary<string, object?> dict)
            return item is string key && dict.ContainsKey(key);
        return false;
    }

    private static object? Add(object? a, object? b)
    {
        if (a is string sa)
            return sa + Stringify(b);
        if (a is int ia && b is int ib) return ia + ib;
        if (IsNumber(a) && IsNumber(b)) return ToDouble(a) + ToDouble(b);
        if (a is IList la && b is IList lb)
        {
            var result = new List<object?>(la.Cast<object?>());
            result.AddRange(lb.Cast<object?>());
            return result;
        }
        return Stringify(a) + Stringify(b);
    }

    private static object? Subtract(object? a, object? b)
    {
        if (a is int ia && b is int ib) return ia - ib;
        if (IsNumber(a) && IsNumber(b)) return ToDouble(a) - ToDouble(b);
        throw new JinjaException("Cannot subtract non-numbers");
    }

    private static object? Multiply(object? a, object? b)
    {
        if (a is int ia && b is int ib) return ia * ib;
        if (IsNumber(a) && IsNumber(b)) return ToDouble(a) * ToDouble(b);
        // String repetition: "abc" * 3
        if (a is string s && b is int n) return string.Concat(Enumerable.Repeat(s, n));
        throw new JinjaException("Cannot multiply");
    }

    private static object? Divide(object? a, object? b)
    {
        if (IsNumber(a) && IsNumber(b)) return ToDouble(a) / ToDouble(b);
        throw new JinjaException("Cannot divide non-numbers");
    }

    private static object? Modulo(object? a, object? b)
    {
        if (a is int ia && b is int ib) return ia % ib;
        if (IsNumber(a) && IsNumber(b)) return ToDouble(a) % ToDouble(b);
        throw new JinjaException("Cannot modulo non-numbers");
    }

    private static bool IsNumber(object? v) => v is int or double or long or float;

    private static double ToDouble(object? v) => Convert.ToDouble(v);

    private static int ToInt(object? v) => Convert.ToInt32(v);

    private static List<object?> RangeWithStep(int start, int stop, int step)
    {
        if (step == 0) throw new JinjaException("range() step must not be zero");
        var result = new List<object?>();
        if (step > 0)
            for (int i = start; i < stop; i += step) result.Add(i);
        else
            for (int i = start; i > stop; i += step) result.Add(i);
        return result;
    }

    private static object? GetLength(object? v)
    {
        if (v is string s) return s.Length;
        if (v is ICollection c) return c.Count;
        if (v is IList l) return l.Count;
        return 0;
    }

    private static object? GetFirst(object? v)
    {
        if (v is IList list && list.Count > 0) return list[0];
        if (v is string s && s.Length > 0) return s[0].ToString();
        return null;
    }

    private static object? GetLast(object? v)
    {
        if (v is IList list && list.Count > 0) return list[^1];
        if (v is string s && s.Length > 0) return s[^1].ToString();
        return null;
    }

    private static IEnumerable<object?> ToEnumerable(object? value)
    {
        if (value is IList list)
            return list.Cast<object?>();
        if (value is Dictionary<string, object?> dict)
            return dict.Select(kvp => (object?)kvp);
        if (value is IEnumerable<object?> enumerable)
            return enumerable;
        if (value is string s)
            return s.Select(c => (object?)c.ToString());
        return [];
    }

    private static List<object?> ToList(IEnumerable e)
    {
        var result = new List<object?>();
        foreach (var item in e)
            result.Add(item);
        return result;
    }

    private static object? GetDictItems(object? input)
    {
        if (input is Dictionary<string, object?> dict)
            return dict.Select(kvp => (object?)new List<object?> { kvp.Key, kvp.Value }).ToList();
        return new List<object?>();
    }

    private static object? JoinFilter(object? input, IReadOnlyList<object?> args)
    {
        var separator = args.Count > 0 ? Stringify(args[0]) : "";
        if (input is IList list)
            return string.Join(separator, list.Cast<object?>().Select(Stringify));
        return Stringify(input);
    }

    private static object? ReplaceFilter(object? input, IReadOnlyList<object?> args)
    {
        if (input is string s && args.Count >= 2)
            return s.Replace(Stringify(args[0]), Stringify(args[1]));
        return input;
    }

    private static object? SelectAttrFilter(object? input, IReadOnlyList<object?> args)
    {
        if (input is not IList list || args.Count < 1)
            return input;

        var attrName = Stringify(args[0]);
        var result = new List<object?>();

        foreach (var item in list)
        {
            if (item is Dictionary<string, object?> dict && dict.TryGetValue(attrName, out var val))
            {
                if (args.Count == 1)
                {
                    if (IsTruthy(val)) result.Add(item);
                }
                else if (args.Count >= 3)
                {
                    var op = Stringify(args[1]);
                    var expected = args[2];
                    bool match = op switch
                    {
                        "equalto" or "==" or "eq" => Equals(val, expected),
                        "ne" or "!=" => !Equals(val, expected),
                        _ => false
                    };
                    if (match) result.Add(item);
                }
            }
        }

        return result;
    }

    private static object? MapFilter(object? input, IReadOnlyList<object?> args)
    {
        if (input is not IList list || args.Count < 1)
            return input;

        var attr = Stringify(args[0]);
        if (attr == "attribute" && args.Count >= 2)
            attr = Stringify(args[1]);

        var result = new List<object?>();
        foreach (var item in list)
        {
            if (item is Dictionary<string, object?> dict && dict.TryGetValue(attr, out var val))
                result.Add(val);
            else
                result.Add(Undefined);
        }
        return result;
    }

    private static object? AppendToList(IList list, List<object?> args)
    {
        if (args.Count > 0) list.Add(args[0]);
        return null; // append returns None in Python
    }

    private static object? InsertIntoList(IList list, List<object?> args)
    {
        if (args.Count >= 2 && args[0] is int idx)
            list.Insert(idx, args[1]);
        return null;
    }

    private static object? UpdateDict(Dictionary<string, object?> dict, List<object?> args)
    {
        if (args.Count > 0 && args[0] is Dictionary<string, object?> other)
        {
            foreach (var kvp in other)
                dict[kvp.Key] = kvp.Value;
        }
        return null;
    }

    private static string FormatStrftime(DateTime dt, string pythonFormat)
    {
        var sb = new StringBuilder(pythonFormat.Length);
        for (int i = 0; i < pythonFormat.Length; i++)
        {
            if (pythonFormat[i] == '%' && i + 1 < pythonFormat.Length)
            {
                char spec = pythonFormat[++i];
                sb.Append(spec switch
                {
                    'd' => dt.Day.ToString("D2"),
                    'm' => dt.Month.ToString("D2"),
                    'Y' => dt.Year.ToString("D4"),
                    'H' => dt.Hour.ToString("D2"),
                    'M' => dt.Minute.ToString("D2"),
                    'S' => dt.Second.ToString("D2"),
                    'b' => dt.ToString("MMM", System.Globalization.CultureInfo.InvariantCulture),
                    'B' => dt.ToString("MMMM", System.Globalization.CultureInfo.InvariantCulture),
                    'A' => dt.ToString("dddd", System.Globalization.CultureInfo.InvariantCulture),
                    'a' => dt.ToString("ddd", System.Globalization.CultureInfo.InvariantCulture),
                    'I' => (dt.Hour % 12 == 0 ? 12 : dt.Hour % 12).ToString("D2"),
                    'p' => dt.Hour >= 12 ? "PM" : "AM",
                    '%' => "%",
                    _ => $"%{spec}" // unknown → pass through
                });
            }
            else
            {
                sb.Append(pythonFormat[i]);
            }
        }
        return sb.ToString();
    }
}
