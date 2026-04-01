namespace DotLLM.Engine.Constraints.Regex;

/// <summary>
/// AST node for a parsed regex pattern. Produced by <see cref="RegexParser"/>,
/// consumed by <see cref="NfaBuilder"/>.
/// </summary>
internal abstract record RegexNode
{
    /// <summary>Single literal character.</summary>
    internal sealed record Literal(char Ch) : RegexNode;

    /// <summary>Character class: set of char ranges. Negated means complement.</summary>
    internal sealed record CharClass(CharRange[] Ranges, bool Negated) : RegexNode;

    /// <summary>Concatenation of sub-expressions (left to right).</summary>
    internal sealed record Concat(RegexNode[] Children) : RegexNode;

    /// <summary>Alternation (OR) of sub-expressions.</summary>
    internal sealed record Alternation(RegexNode[] Children) : RegexNode;

    /// <summary>
    /// Repetition: <paramref name="Min"/>..<paramref name="Max"/> occurrences.
    /// <paramref name="Max"/> = <see cref="int.MaxValue"/> for unbounded.
    /// </summary>
    internal sealed record Repeat(RegexNode Child, int Min, int Max) : RegexNode;
}
