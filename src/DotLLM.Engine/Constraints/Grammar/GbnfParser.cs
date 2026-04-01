namespace DotLLM.Engine.Constraints.Grammar;

/// <summary>
/// Parses GBNF (llama.cpp grammar format) text into an AST.
/// </summary>
/// <remarks>
/// GBNF syntax:
/// <code>
/// rule-name ::= definition
/// "literal"            — string literal
/// [a-z]                — character class
/// [^a-z]               — negated character class
/// ( ... )              — grouping
/// |                    — alternation
/// * + ?                — repetition
/// rule-name            — rule reference
/// # comment            — line comment
/// </code>
/// </remarks>
internal static class GbnfParser
{
    /// <summary>
    /// Parses GBNF grammar text into a <see cref="GbnfGrammar"/>.
    /// The first rule defined is the root rule.
    /// </summary>
    /// <param name="grammar">The GBNF grammar text.</param>
    /// <returns>The parsed grammar AST.</returns>
    /// <exception cref="ArgumentException">Thrown for syntax errors or undefined rule references.</exception>
    public static GbnfGrammar Parse(string grammar)
    {
        if (string.IsNullOrWhiteSpace(grammar))
            throw new ArgumentException("Grammar cannot be empty.", nameof(grammar));

        var rules = new Dictionary<string, GbnfRule>();
        string? rootRuleName = null;

        var lines = grammar.Split('\n');
        int i = 0;

        while (i < lines.Length)
        {
            var line = lines[i].TrimEnd('\r');

            // Skip blank lines and comments
            var trimmed = line.TrimStart();
            if (trimmed.Length == 0 || trimmed[0] == '#')
            {
                i++;
                continue;
            }

            // Collect continuation lines (rule bodies can span multiple lines,
            // but only the first line has ::=)
            string fullLine = line;
            while (i + 1 < lines.Length)
            {
                var nextLine = lines[i + 1].TrimEnd('\r');
                var nextTrimmed = nextLine.TrimStart();

                // Next line is a continuation if it doesn't contain ::= and isn't blank/comment
                // and starts with whitespace or |
                if (nextTrimmed.Length > 0 && nextTrimmed[0] != '#' && !nextTrimmed.Contains("::=")
                    && (nextLine.Length > 0 && (nextLine[0] == ' ' || nextLine[0] == '\t' || nextTrimmed[0] == '|')))
                {
                    fullLine += " " + nextTrimmed;
                    i++;
                }
                else
                {
                    break;
                }
            }
            i++;

            // Parse rule: name ::= body
            int defIdx = fullLine.IndexOf("::=", StringComparison.Ordinal);
            if (defIdx < 0)
                throw new ArgumentException($"Expected rule definition (name ::= body), got: '{fullLine}'");

            string name = fullLine[..defIdx].Trim();
            if (name.Length == 0)
                throw new ArgumentException("Rule name cannot be empty.");
            ValidateRuleName(name);

            string body = fullLine[(defIdx + 3)..].Trim();
            // Strip trailing comment
            int commentIdx = FindUnquotedComment(body);
            if (commentIdx >= 0)
                body = body[..commentIdx].TrimEnd();

            if (body.Length == 0)
                throw new ArgumentException($"Rule '{name}' has empty body.");

            int pos = 0;
            var node = ParseAlternation(body, ref pos);

            // Skip trailing whitespace
            while (pos < body.Length && char.IsWhiteSpace(body[pos]))
                pos++;

            if (pos != body.Length)
                throw new ArgumentException($"Unexpected content in rule '{name}' at position {pos}: '{body[pos..]}'");

            if (rules.ContainsKey(name))
                throw new ArgumentException($"Duplicate rule definition: '{name}'.");

            rules[name] = new GbnfRule(name, node);
            rootRuleName ??= name;
        }

        if (rules.Count == 0)
            throw new ArgumentException("Grammar contains no rules.");

        // Validate all rule references
        ValidateRuleReferences(rules);

        // Check for left recursion
        DetectLeftRecursion(rules);

        return new GbnfGrammar(rules, rootRuleName!);
    }

    private static void ValidateRuleName(string name)
    {
        foreach (char c in name)
        {
            if (!char.IsLetterOrDigit(c) && c != '-' && c != '_')
                throw new ArgumentException($"Invalid character '{c}' in rule name '{name}'.");
        }
    }

    private static int FindUnquotedComment(string body)
    {
        bool inString = false;
        bool inCharClass = false;
        for (int i = 0; i < body.Length; i++)
        {
            char c = body[i];
            if (c == '"' && !inCharClass) inString = !inString;
            else if (c == '[' && !inString) inCharClass = true;
            else if (c == ']' && !inString) inCharClass = false;
            else if (c == '\\' && (inString || inCharClass) && i + 1 < body.Length) i++; // skip escaped
            else if (c == '#' && !inString && !inCharClass) return i;
        }
        return -1;
    }

    private static GbnfNode ParseAlternation(string body, ref int pos)
    {
        var first = ParseSequence(body, ref pos);
        SkipWhitespace(body, ref pos);
        if (pos >= body.Length || body[pos] != '|')
            return first;

        var alternatives = new List<GbnfNode> { first };
        while (pos < body.Length && body[pos] == '|')
        {
            pos++; // skip '|'
            SkipWhitespace(body, ref pos);
            alternatives.Add(ParseSequence(body, ref pos));
            SkipWhitespace(body, ref pos);
        }
        return new GbnfNode.Alternation([.. alternatives]);
    }

    private static GbnfNode ParseSequence(string body, ref int pos)
    {
        var elements = new List<GbnfNode>();
        while (pos < body.Length)
        {
            SkipWhitespace(body, ref pos);
            if (pos >= body.Length) break;

            char c = body[pos];
            // End of sequence: alternation separator or close paren
            if (c == '|' || c == ')')
                break;

            elements.Add(ParseQuantified(body, ref pos));
        }

        return elements.Count switch
        {
            0 => throw new ArgumentException("Empty sub-expression in grammar."),
            1 => elements[0],
            _ => new GbnfNode.Sequence([.. elements])
        };
    }

    private static GbnfNode ParseQuantified(string body, ref int pos)
    {
        var atom = ParseAtom(body, ref pos);
        if (pos >= body.Length)
            return atom;

        return body[pos] switch
        {
            '*' => Advance(ref pos, new GbnfNode.Repeat(atom, 0, int.MaxValue)),
            '+' => Advance(ref pos, new GbnfNode.Repeat(atom, 1, int.MaxValue)),
            '?' => Advance(ref pos, new GbnfNode.Repeat(atom, 0, 1)),
            _ => atom
        };

        static GbnfNode Advance(ref int pos, GbnfNode node) { pos++; return node; }
    }

    private static GbnfNode ParseAtom(string body, ref int pos)
    {
        if (pos >= body.Length)
            throw new ArgumentException("Unexpected end of grammar rule body.");

        char c = body[pos];
        return c switch
        {
            '"' => ParseLiteral(body, ref pos),
            '[' => ParseCharClass(body, ref pos),
            '(' => ParseGroup(body, ref pos),
            _ when char.IsLetterOrDigit(c) || c == '-' || c == '_' => ParseRuleRef(body, ref pos),
            _ => throw new ArgumentException($"Unexpected character '{c}' at position {pos} in grammar rule body.")
        };
    }

    private static GbnfNode ParseLiteral(string body, ref int pos)
    {
        pos++; // skip opening '"'
        var chars = new List<char>();
        while (pos < body.Length && body[pos] != '"')
        {
            if (body[pos] == '\\')
            {
                pos++;
                if (pos >= body.Length)
                    throw new ArgumentException("Unterminated escape in string literal.");
                chars.Add(body[pos] switch
                {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    _ => throw new ArgumentException($"Unknown escape '\\{body[pos]}' in string literal.")
                });
            }
            else
            {
                chars.Add(body[pos]);
            }
            pos++;
        }

        if (pos >= body.Length)
            throw new ArgumentException("Unterminated string literal.");
        pos++; // skip closing '"'

        return new GbnfNode.Literal(new string(chars.ToArray()));
    }

    private static GbnfNode ParseCharClass(string body, ref int pos)
    {
        pos++; // skip '['
        bool negated = false;
        if (pos < body.Length && body[pos] == '^')
        {
            negated = true;
            pos++;
        }

        var ranges = new List<CharRange>();
        while (pos < body.Length && body[pos] != ']')
        {
            char lo = ParseCharClassChar(body, ref pos);
            if (pos + 1 < body.Length && body[pos] == '-' && body[pos + 1] != ']')
            {
                pos++; // skip '-'
                char hi = ParseCharClassChar(body, ref pos);
                if (lo > hi)
                    throw new ArgumentException($"Invalid character range: '{lo}'-'{hi}'.");
                ranges.Add(new CharRange(lo, hi));
            }
            else
            {
                ranges.Add(new CharRange(lo, lo));
            }
        }

        if (pos >= body.Length)
            throw new ArgumentException("Unterminated character class.");
        pos++; // skip ']'

        if (ranges.Count == 0)
            throw new ArgumentException("Empty character class.");

        return new GbnfNode.CharClass([.. ranges], negated);
    }

    private static char ParseCharClassChar(string body, ref int pos)
    {
        if (body[pos] == '\\')
        {
            pos++;
            if (pos >= body.Length)
                throw new ArgumentException("Unterminated escape in character class.");
            char escaped = body[pos++];
            return escaped switch
            {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '\\' => '\\',
                ']' => ']',
                '-' => '-',
                '^' => '^',
                _ => throw new ArgumentException($"Unknown escape '\\{escaped}' in character class.")
            };
        }

        return body[pos++];
    }

    private static GbnfNode ParseGroup(string body, ref int pos)
    {
        pos++; // skip '('
        SkipWhitespace(body, ref pos);
        var inner = ParseAlternation(body, ref pos);
        SkipWhitespace(body, ref pos);
        if (pos >= body.Length || body[pos] != ')')
            throw new ArgumentException("Unmatched '(' in grammar rule body.");
        pos++; // skip ')'
        return inner;
    }

    private static GbnfNode ParseRuleRef(string body, ref int pos)
    {
        int start = pos;
        while (pos < body.Length && (char.IsLetterOrDigit(body[pos]) || body[pos] == '-' || body[pos] == '_'))
            pos++;
        string name = body[start..pos];
        return new GbnfNode.RuleRef(name);
    }

    private static void SkipWhitespace(string body, ref int pos)
    {
        while (pos < body.Length && body[pos] is ' ' or '\t')
            pos++;
    }

    private static void ValidateRuleReferences(Dictionary<string, GbnfRule> rules)
    {
        foreach (var rule in rules.Values)
        {
            ValidateNodeReferences(rule.Body, rules, rule.Name);
        }
    }

    private static void ValidateNodeReferences(GbnfNode node, Dictionary<string, GbnfRule> rules, string context)
    {
        switch (node)
        {
            case GbnfNode.RuleRef rr:
                if (!rules.ContainsKey(rr.RuleName))
                    throw new ArgumentException($"Undefined rule reference '{rr.RuleName}' in rule '{context}'.");
                break;
            case GbnfNode.Sequence seq:
                foreach (var child in seq.Elements) ValidateNodeReferences(child, rules, context);
                break;
            case GbnfNode.Alternation alt:
                foreach (var child in alt.Alternatives) ValidateNodeReferences(child, rules, context);
                break;
            case GbnfNode.Repeat rep:
                ValidateNodeReferences(rep.Child, rules, context);
                break;
        }
    }

    /// <summary>
    /// Detects direct left recursion (rule → rule ...) and throws.
    /// Indirect left recursion is not checked (complex, and rare in practice).
    /// </summary>
    private static void DetectLeftRecursion(Dictionary<string, GbnfRule> rules)
    {
        foreach (var rule in rules.Values)
        {
            var firstRefs = new HashSet<string>();
            CollectFirstRuleRefs(rule.Body, rules, firstRefs, []);
            if (firstRefs.Contains(rule.Name))
                throw new ArgumentException(
                    $"Left recursion detected in rule '{rule.Name}'. Left-recursive grammars are not supported.");
        }
    }

    private static void CollectFirstRuleRefs(GbnfNode node, Dictionary<string, GbnfRule> rules,
                                              HashSet<string> firstRefs, HashSet<string> visited)
    {
        switch (node)
        {
            case GbnfNode.RuleRef rr:
                if (firstRefs.Add(rr.RuleName) && visited.Add(rr.RuleName) && rules.TryGetValue(rr.RuleName, out var target))
                    CollectFirstRuleRefs(target.Body, rules, firstRefs, visited);
                break;
            case GbnfNode.Sequence seq when seq.Elements.Length > 0:
                CollectFirstRuleRefs(seq.Elements[0], rules, firstRefs, visited);
                break;
            case GbnfNode.Alternation alt:
                foreach (var child in alt.Alternatives)
                    CollectFirstRuleRefs(child, rules, firstRefs, visited);
                break;
            case GbnfNode.Repeat rep:
                CollectFirstRuleRefs(rep.Child, rules, firstRefs, visited);
                break;
        }
    }
}
