namespace DotLLM.Tokenizers.ChatTemplates;

/// <summary>
/// Exception thrown during Jinja2 template lexing, parsing, or evaluation.
/// </summary>
public sealed class JinjaException : Exception
{
    /// <summary>Line number in the template source (1-based).</summary>
    public int Line { get; }

    /// <summary>Column number in the template source (1-based).</summary>
    public int Column { get; }

    /// <summary>Creates an exception with line/column location info.</summary>
    public JinjaException(string message, int line, int column)
        : base($"Line {line}, Col {column}: {message}")
    {
        Line = line;
        Column = column;
    }

    /// <summary>Creates an exception without location info.</summary>
    public JinjaException(string message)
        : base(message)
    {
    }
}
