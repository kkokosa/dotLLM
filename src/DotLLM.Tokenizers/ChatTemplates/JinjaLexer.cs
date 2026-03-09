namespace DotLLM.Tokenizers.ChatTemplates;

/// <summary>
/// Tokenizes a Jinja2 template string into a flat token list.
/// Two-mode lexer: text mode (outside tags) and tag mode (inside {{ }}, {% %}).
/// Handles whitespace control markers ({%- -%} {{- -}}).
/// </summary>
internal sealed class JinjaLexer
{
    private readonly string _source;
    private int _pos;
    private int _line = 1;
    private int _col = 1;
    private int _braceDepth;

    public JinjaLexer(string source)
    {
        _source = source;
    }

    public List<JinjaToken> Tokenize()
    {
        var tokens = new List<JinjaToken>();
        _pos = 0;
        _line = 1;
        _col = 1;
        _braceDepth = 0;

        while (_pos < _source.Length)
        {
            // Check for comment {# ... #}
            if (Match("{#"))
            {
                SkipComment();
                continue;
            }

            // Check for expression start {{ or {{-
            if (Match("{{"))
            {
                bool stripLeft = false;
                if (_pos < _source.Length && _source[_pos] == '-')
                {
                    stripLeft = true;
                    Advance();
                }

                if (stripLeft)
                    StripTrailingWhitespace(tokens);

                tokens.Add(new JinjaToken(JinjaTokenType.ExprStart, null, _line, _col));
                TokenizeTag(tokens, "}}");
                continue;
            }

            // Check for statement start {% or {%-
            if (Match("{%"))
            {
                bool stripLeft = false;
                if (_pos < _source.Length && _source[_pos] == '-')
                {
                    stripLeft = true;
                    Advance();
                }

                if (stripLeft)
                    StripTrailingWhitespace(tokens);

                tokens.Add(new JinjaToken(JinjaTokenType.StmtStart, null, _line, _col));
                TokenizeTag(tokens, "%}");
                continue;
            }

            // Text mode: scan until next tag
            int textStart = _pos;
            int textLine = _line;
            int textCol = _col;
            while (_pos < _source.Length && !LookAhead("{{") && !LookAhead("{%") && !LookAhead("{#"))
            {
                Advance();
            }

            if (_pos > textStart)
            {
                string text = _source[textStart.._pos];
                tokens.Add(new JinjaToken(JinjaTokenType.Text, text, textLine, textCol));
            }
        }

        tokens.Add(new JinjaToken(JinjaTokenType.Eof, null, _line, _col));
        return tokens;
    }

    private void TokenizeTag(List<JinjaToken> tokens, string endMarker)
    {
        SkipWhitespaceInTag();

        while (_pos < _source.Length)
        {
            SkipWhitespaceInTag();

            if (_pos >= _source.Length)
                throw new JinjaException("Unexpected end of template inside tag", _line, _col);

            // Only check for end markers when not inside a dict literal
            if (_braceDepth == 0)
            {
                // Check for strip-right + end marker: -}} or -%}
                if (_source[_pos] == '-' && _pos + endMarker.Length < _source.Length &&
                    _source.AsSpan(_pos + 1, endMarker.Length).SequenceEqual(endMarker))
                {
                    _pos += 1 + endMarker.Length;
                    _col += 1 + endMarker.Length;
                    var endType = endMarker == "}}" ? JinjaTokenType.ExprEnd : JinjaTokenType.StmtEnd;
                    tokens.Add(new JinjaToken(endType, null, _line, _col));
                    StripLeadingWhitespace(tokens);
                    return;
                }

                // Check for end marker without strip
                if (LookAhead(endMarker))
                {
                    _pos += endMarker.Length;
                    _col += endMarker.Length;
                    var endType = endMarker == "}}" ? JinjaTokenType.ExprEnd : JinjaTokenType.StmtEnd;
                    tokens.Add(new JinjaToken(endType, null, _line, _col));
                    return;
                }
            }

            char c = _source[_pos];

            // String literal
            if (c is '"' or '\'')
            {
                tokens.Add(ReadString());
                continue;
            }

            // Number literal
            if (char.IsDigit(c))
            {
                tokens.Add(ReadNumber());
                continue;
            }

            // Identifier or keyword
            if (char.IsLetter(c) || c == '_')
            {
                tokens.Add(ReadIdentifierOrKeyword());
                continue;
            }

            // Two-character operators
            if (_pos + 1 < _source.Length)
            {
                string twoChar = _source.Substring(_pos, 2);
                var twoCharType = twoChar switch
                {
                    "==" => JinjaTokenType.Eq,
                    "!=" => JinjaTokenType.Ne,
                    "<=" => JinjaTokenType.Le,
                    ">=" => JinjaTokenType.Ge,
                    _ => (JinjaTokenType?)null
                };

                if (twoCharType.HasValue)
                {
                    tokens.Add(new JinjaToken(twoCharType.Value, null, _line, _col));
                    Advance();
                    Advance();
                    continue;
                }
            }

            // Single-character operators/punctuation
            // Handle braces with depth tracking for dict literals
            if (c == '{')
            {
                _braceDepth++;
                tokens.Add(new JinjaToken(JinjaTokenType.LeftBrace, null, _line, _col));
                Advance();
                continue;
            }

            if (c == '}' && _braceDepth > 0)
            {
                _braceDepth--;
                tokens.Add(new JinjaToken(JinjaTokenType.RightBrace, null, _line, _col));
                Advance();
                continue;
            }

            var singleType = c switch
            {
                '.' => JinjaTokenType.Dot,
                '[' => JinjaTokenType.LeftBracket,
                ']' => JinjaTokenType.RightBracket,
                '(' => JinjaTokenType.LeftParen,
                ')' => JinjaTokenType.RightParen,
                '|' => JinjaTokenType.Pipe,
                ',' => JinjaTokenType.Comma,
                ':' => JinjaTokenType.Colon,
                '=' => JinjaTokenType.Assign,
                '<' => JinjaTokenType.Lt,
                '>' => JinjaTokenType.Gt,
                '+' => JinjaTokenType.Plus,
                '-' => JinjaTokenType.Minus,
                '*' => JinjaTokenType.Multiply,
                '/' => JinjaTokenType.Divide,
                '%' => JinjaTokenType.Modulo,
                '~' => JinjaTokenType.Tilde,
                _ => (JinjaTokenType?)null
            };

            if (singleType.HasValue)
            {
                tokens.Add(new JinjaToken(singleType.Value, null, _line, _col));
                Advance();
                continue;
            }

            throw new JinjaException($"Unexpected character '{c}' in tag", _line, _col);
        }

        throw new JinjaException("Unclosed tag", _line, _col);
    }

    private JinjaToken ReadString()
    {
        char quote = _source[_pos];
        int startLine = _line;
        int startCol = _col;
        Advance(); // skip opening quote

        var sb = new System.Text.StringBuilder();
        while (_pos < _source.Length && _source[_pos] != quote)
        {
            if (_source[_pos] == '\\' && _pos + 1 < _source.Length)
            {
                Advance(); // skip backslash
                char escaped = _source[_pos];
                sb.Append(escaped switch
                {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    _ => escaped
                });
                Advance();
            }
            else
            {
                sb.Append(_source[_pos]);
                Advance();
            }
        }

        if (_pos >= _source.Length)
            throw new JinjaException("Unterminated string literal", startLine, startCol);

        Advance(); // skip closing quote
        return new JinjaToken(JinjaTokenType.StringLiteral, sb.ToString(), startLine, startCol);
    }

    private JinjaToken ReadNumber()
    {
        int startLine = _line;
        int startCol = _col;
        int start = _pos;

        while (_pos < _source.Length && char.IsDigit(_source[_pos]))
            Advance();

        string numStr = _source[start.._pos];
        return new JinjaToken(JinjaTokenType.IntLiteral, int.Parse(numStr), startLine, startCol);
    }

    private JinjaToken ReadIdentifierOrKeyword()
    {
        int startLine = _line;
        int startCol = _col;
        int start = _pos;

        while (_pos < _source.Length && (char.IsLetterOrDigit(_source[_pos]) || _source[_pos] == '_'))
            Advance();

        string word = _source[start.._pos];

        var type = word switch
        {
            "if" => JinjaTokenType.If,
            "elif" => JinjaTokenType.Elif,
            "else" => JinjaTokenType.Else,
            "endif" => JinjaTokenType.Endif,
            "for" => JinjaTokenType.For,
            "in" => JinjaTokenType.In,
            "endfor" => JinjaTokenType.Endfor,
            "set" => JinjaTokenType.Set,
            "not" => JinjaTokenType.Not,
            "and" => JinjaTokenType.And,
            "or" => JinjaTokenType.Or,
            "is" => JinjaTokenType.Is,
            "defined" => JinjaTokenType.Defined,
            "namespace" => JinjaTokenType.Namespace,
            "true" or "True" => JinjaTokenType.BoolLiteral,
            "false" or "False" => JinjaTokenType.BoolLiteral,
            "none" or "None" => JinjaTokenType.NoneLiteral,
            _ => JinjaTokenType.Identifier
        };

        object? value = type switch
        {
            JinjaTokenType.BoolLiteral => word is "true" or "True",
            JinjaTokenType.Identifier => word,
            _ => null
        };

        return new JinjaToken(type, value, startLine, startCol);
    }

    private void SkipComment()
    {
        // Already consumed {#, now scan for #}
        while (_pos < _source.Length)
        {
            if (_source[_pos] == '#' && _pos + 1 < _source.Length && _source[_pos + 1] == '}')
            {
                Advance();
                Advance();
                return;
            }
            Advance();
        }
        throw new JinjaException("Unclosed comment", _line, _col);
    }

    private void SkipWhitespaceInTag()
    {
        while (_pos < _source.Length && _source[_pos] is ' ' or '\t' or '\r' or '\n')
            Advance();
    }

    private bool Match(string s)
    {
        if (_pos + s.Length > _source.Length)
            return false;
        if (!_source.AsSpan(_pos, s.Length).SequenceEqual(s))
            return false;
        _pos += s.Length;
        _col += s.Length;
        return true;
    }

    private bool LookAhead(string s)
    {
        if (_pos + s.Length > _source.Length)
            return false;
        return _source.AsSpan(_pos, s.Length).SequenceEqual(s);
    }

    private void Advance()
    {
        if (_pos < _source.Length)
        {
            if (_source[_pos] == '\n')
            {
                _line++;
                _col = 1;
            }
            else
            {
                _col++;
            }
            _pos++;
        }
    }

    /// <summary>
    /// Strips trailing whitespace (including newlines) from the last Text token.
    /// Used for {{- and {%- whitespace control.
    /// </summary>
    private static void StripTrailingWhitespace(List<JinjaToken> tokens)
    {
        if (tokens.Count == 0)
            return;

        int last = tokens.Count - 1;
        if (tokens[last].Type != JinjaTokenType.Text)
            return;

        string text = (string)tokens[last].Value!;
        string trimmed = text.TrimEnd(' ', '\t', '\r', '\n');
        if (trimmed.Length == 0)
            tokens.RemoveAt(last);
        else
            tokens[last] = tokens[last] with { Value = trimmed };
    }

    /// <summary>
    /// Sets a flag so the next Text token emitted has its leading whitespace stripped.
    /// Used for -}} and -%} whitespace control.
    /// Actually modifies the token list inline — the flag is implemented by modifying the next text token.
    /// We achieve this by adding a sentinel that the tokenizer main loop handles.
    /// Instead, we just mark position and strip when the next text is emitted.
    /// </summary>
    private void StripLeadingWhitespace(List<JinjaToken> tokens)
    {
        // Strip leading whitespace from the source at the current position
        // (the text that would become the next Text token)
        while (_pos < _source.Length && _source[_pos] is ' ' or '\t' or '\r' or '\n')
            Advance();
    }
}
