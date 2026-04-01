namespace DotLLM.Engine.Constraints.Regex;

/// <summary>
/// Parses a regex pattern string into an AST for DFA construction.
/// Supports: literals, char classes (<c>[a-z]</c>, <c>[^...]</c>), predefined classes
/// (<c>\d</c>, <c>\w</c>, <c>\s</c>, <c>.</c>), quantifiers (<c>*</c>, <c>+</c>, <c>?</c>,
/// <c>{n,m}</c>), alternation (<c>|</c>), grouping (<c>(...)</c>), standard escapes.
/// Does NOT support backreferences, lookahead/lookbehind, or capture groups.
/// </summary>
internal static class RegexParser
{
    /// <summary>
    /// Parses a regex pattern into its AST representation.
    /// </summary>
    /// <param name="pattern">The regex pattern string.</param>
    /// <returns>The root AST node.</returns>
    /// <exception cref="ArgumentException">Thrown for invalid or unsupported regex constructs.</exception>
    public static RegexNode Parse(string pattern)
    {
        if (string.IsNullOrEmpty(pattern))
            throw new ArgumentException("Regex pattern cannot be empty.", nameof(pattern));

        int pos = 0;
        var result = ParseAlternation(pattern, ref pos);
        if (pos != pattern.Length)
            throw new ArgumentException($"Unexpected character '{pattern[pos]}' at position {pos}.");
        return result;
    }

    private static RegexNode ParseAlternation(string pattern, ref int pos)
    {
        var first = ParseConcat(pattern, ref pos);
        if (pos >= pattern.Length || pattern[pos] != '|')
            return first;

        var alternatives = new List<RegexNode> { first };
        while (pos < pattern.Length && pattern[pos] == '|')
        {
            pos++; // skip '|'
            alternatives.Add(ParseConcat(pattern, ref pos));
        }
        return new RegexNode.Alternation([.. alternatives]);
    }

    private static RegexNode ParseConcat(string pattern, ref int pos)
    {
        var children = new List<RegexNode>();
        while (pos < pattern.Length && pattern[pos] != '|' && pattern[pos] != ')')
        {
            children.Add(ParseQuantified(pattern, ref pos));
        }

        return children.Count switch
        {
            0 => throw new ArgumentException("Empty sub-expression in regex."),
            1 => children[0],
            _ => new RegexNode.Concat([.. children])
        };
    }

    private static RegexNode ParseQuantified(string pattern, ref int pos)
    {
        var atom = ParseAtom(pattern, ref pos);
        if (pos >= pattern.Length)
            return atom;

        switch (pattern[pos])
        {
            case '*':
                pos++;
                RejectLazyQuantifier(pattern, ref pos);
                return new RegexNode.Repeat(atom, 0, int.MaxValue);
            case '+':
                pos++;
                RejectLazyQuantifier(pattern, ref pos);
                return new RegexNode.Repeat(atom, 1, int.MaxValue);
            case '?':
                pos++;
                RejectLazyQuantifier(pattern, ref pos);
                return new RegexNode.Repeat(atom, 0, 1);
            case '{':
                return ParseBoundedRepeat(pattern, ref pos, atom);
            default:
                return atom;
        }
    }

    private static void RejectLazyQuantifier(string pattern, ref int pos)
    {
        if (pos < pattern.Length && pattern[pos] == '?')
            throw new ArgumentException(
                $"Lazy quantifiers (e.g., *?, +?) are not supported in DFA-based regex at position {pos}. " +
                "DFA regex is inherently greedy.");
    }

    private static RegexNode ParseBoundedRepeat(string pattern, ref int pos, RegexNode atom)
    {
        pos++; // skip '{'
        int min = ParseInt(pattern, ref pos);
        int max;

        if (pos < pattern.Length && pattern[pos] == '}')
        {
            max = min; // {n} exact
            pos++;
        }
        else if (pos < pattern.Length && pattern[pos] == ',')
        {
            pos++; // skip ','
            if (pos < pattern.Length && pattern[pos] == '}')
            {
                max = int.MaxValue; // {n,}
                pos++;
            }
            else
            {
                max = ParseInt(pattern, ref pos); // {n,m}
                if (pos >= pattern.Length || pattern[pos] != '}')
                    throw new ArgumentException("Expected '}' in bounded repetition.");
                pos++;
            }
        }
        else
        {
            throw new ArgumentException("Invalid bounded repetition syntax.");
        }

        if (min > max)
            throw new ArgumentException($"Invalid repetition bounds: {{{min},{max}}}.");
        if (max != int.MaxValue && max > 1000)
            throw new ArgumentException($"Repetition bound {max} is too large (max 1000).");

        RejectLazyQuantifier(pattern, ref pos);
        return new RegexNode.Repeat(atom, min, max);
    }

    private static int ParseInt(string pattern, ref int pos)
    {
        int start = pos;
        while (pos < pattern.Length && char.IsAsciiDigit(pattern[pos]))
            pos++;
        if (pos == start)
            throw new ArgumentException($"Expected integer at position {pos}.");
        return int.Parse(pattern.AsSpan(start, pos - start));
    }

    private static RegexNode ParseAtom(string pattern, ref int pos)
    {
        if (pos >= pattern.Length)
            throw new ArgumentException("Unexpected end of regex pattern.");

        char c = pattern[pos];
        switch (c)
        {
            case '(':
                return ParseGroup(pattern, ref pos);
            case '[':
                return ParseCharClass(pattern, ref pos);
            case '\\':
                return ParseEscape(pattern, ref pos);
            case '.':
                pos++;
                // Match any character except newline (standard regex semantics)
                return new RegexNode.CharClass(
                    [new CharRange('\n', '\n')],
                    Negated: true);
            case '^' or '$':
                throw new ArgumentException(
                    $"Anchors ('{c}') are not supported — regex is implicitly anchored to the full output.");
            case ')':
                throw new ArgumentException("Unmatched ')' in regex.");
            case '*' or '+' or '?' or '{':
                throw new ArgumentException($"Unexpected quantifier '{c}' without preceding atom.");
            default:
                pos++;
                return new RegexNode.Literal(c);
        }
    }

    private static RegexNode ParseGroup(string pattern, ref int pos)
    {
        pos++; // skip '('

        // Check for unsupported group types
        if (pos < pattern.Length && pattern[pos] == '?')
        {
            if (pos + 1 < pattern.Length)
            {
                char next = pattern[pos + 1];
                if (next is '=' or '!' or '<')
                    throw new ArgumentException(
                        $"Lookahead/lookbehind assertions are not supported in DFA-based regex at position {pos}.");
                if (next == ':')
                {
                    // Non-capturing group (?:...) — skip the ?:
                    pos += 2;
                }
                else if (next == 'P' || char.IsLetter(next))
                {
                    throw new ArgumentException(
                        $"Named/special groups are not supported in DFA-based regex at position {pos}.");
                }
            }
        }

        var inner = ParseAlternation(pattern, ref pos);
        if (pos >= pattern.Length || pattern[pos] != ')')
            throw new ArgumentException("Unmatched '(' in regex.");
        pos++; // skip ')'
        return inner;
    }

    private static RegexNode ParseCharClass(string pattern, ref int pos)
    {
        pos++; // skip '['
        bool negated = false;
        if (pos < pattern.Length && pattern[pos] == '^')
        {
            negated = true;
            pos++;
        }

        var ranges = new List<CharRange>();
        // ] as first character (or after ^) is literal
        bool first = true;
        while (pos < pattern.Length && (first || pattern[pos] != ']'))
        {
            first = false;
            if (pattern[pos] == '\\')
            {
                pos++;
                if (pos >= pattern.Length)
                    throw new ArgumentException("Unterminated escape in character class.");
                var escaped = ParseEscapedCharOrClass(pattern, ref pos);
                if (escaped.ranges != null)
                    ranges.AddRange(escaped.ranges);
                else
                    AddCharOrRange(pattern, ref pos, ranges, escaped.ch);
            }
            else
            {
                char ch = pattern[pos++];
                AddCharOrRange(pattern, ref pos, ranges, ch);
            }
        }

        if (pos >= pattern.Length)
            throw new ArgumentException("Unterminated character class.");
        pos++; // skip ']'

        return new RegexNode.CharClass([.. ranges], negated);
    }

    private static void AddCharOrRange(string pattern, ref int pos, List<CharRange> ranges, char lo)
    {
        // Check for range: a-z
        if (pos + 1 < pattern.Length && pattern[pos] == '-' && pattern[pos + 1] != ']')
        {
            pos++; // skip '-'
            char hi;
            if (pattern[pos] == '\\')
            {
                pos++;
                if (pos >= pattern.Length)
                    throw new ArgumentException("Unterminated escape in character class range.");
                var escaped = ParseEscapedCharOrClass(pattern, ref pos);
                if (escaped.ranges != null)
                    throw new ArgumentException("Cannot use character class shorthand (e.g., \\d) as range endpoint.");
                hi = escaped.ch;
            }
            else
            {
                hi = pattern[pos++];
            }

            if (lo > hi)
                throw new ArgumentException($"Invalid character range: '{lo}'-'{hi}'.");
            ranges.Add(new CharRange(lo, hi));
        }
        else
        {
            ranges.Add(new CharRange(lo, lo));
        }
    }

    private static RegexNode ParseEscape(string pattern, ref int pos)
    {
        pos++; // skip '\\'
        if (pos >= pattern.Length)
            throw new ArgumentException("Unterminated escape sequence.");

        var escaped = ParseEscapedCharOrClass(pattern, ref pos);
        if (escaped.ranges != null)
            return new RegexNode.CharClass(escaped.ranges, Negated: false);
        return new RegexNode.Literal(escaped.ch);
    }

    /// <summary>
    /// Parses an escaped character or character class shorthand after the backslash.
    /// Returns either a single char or an array of ranges (for \d, \w, \s and their negations).
    /// </summary>
    private static (char ch, CharRange[]? ranges) ParseEscapedCharOrClass(string pattern, ref int pos)
    {
        char c = pattern[pos++];
        return c switch
        {
            'd' => ('\0', [new CharRange('0', '9')]),
            'D' => ('\0', [new CharRange('\0', '/'), new CharRange(':', '\uffff')]),
            'w' => ('\0', [new CharRange('0', '9'), new CharRange('A', 'Z'), new CharRange('a', 'z'), new CharRange('_', '_')]),
            'W' => ('\0', [new CharRange('\0', '/'), new CharRange(':', '@'), new CharRange('[', '^'), new CharRange('`', '`'), new CharRange('{', '\uffff')]),
            's' => ('\0', [new CharRange(' ', ' '), new CharRange('\t', '\t'), new CharRange('\n', '\n'), new CharRange('\r', '\r'), new CharRange('\f', '\f')]),
            'S' => ('\0', [new CharRange('\0', '\b'), new CharRange('\u000e', '\u001f'), new CharRange('!', '\uffff')]),
            'n' => ('\n', null),
            'r' => ('\r', null),
            't' => ('\t', null),
            'f' => ('\f', null),
            'v' => ('\v', null),
            '0' => ('\0', null),
            // Backreference detection
            >= '1' and <= '9' => throw new ArgumentException(
                $"Backreferences (\\{c}) are not supported in DFA-based regex."),
            // Literal escapes
            _ => (c, null)
        };
    }
}
