namespace DotLLM.Tokenizers.ChatTemplates;

internal enum JinjaTokenType
{
    Text,
    ExprStart,      // {{
    ExprEnd,        // }}
    StmtStart,      // {%
    StmtEnd,        // %}
    StringLiteral,
    IntLiteral,
    BoolLiteral,
    NoneLiteral,
    Identifier,
    If,
    Elif,
    Else,
    Endif,
    For,
    In,
    Endfor,
    Set,
    Not,
    And,
    Or,
    Is,
    Defined,
    Namespace,
    Dot,
    LeftBracket,
    RightBracket,
    LeftParen,
    RightParen,
    LeftBrace,      // { (dict literal)
    RightBrace,     // } (dict literal)
    Pipe,
    Comma,
    Colon,          // : (dict key-value separator)
    Assign,
    Eq,             // ==
    Ne,             // !=
    Lt,             // <
    Gt,             // >
    Le,             // <=
    Ge,             // >=
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Tilde,          // ~ (string concatenation)
    Eof
}

internal readonly record struct JinjaToken(JinjaTokenType Type, object? Value, int Line, int Column);
