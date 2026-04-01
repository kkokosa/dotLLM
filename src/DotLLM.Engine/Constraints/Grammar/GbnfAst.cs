namespace DotLLM.Engine.Constraints.Grammar;

/// <summary>
/// Parsed GBNF grammar: a set of named rules with a designated root rule.
/// </summary>
internal sealed record GbnfGrammar(Dictionary<string, GbnfRule> Rules, string RootRuleName);

/// <summary>
/// A named rule in a GBNF grammar.
/// </summary>
internal sealed record GbnfRule(string Name, GbnfNode Body);

/// <summary>
/// AST node for a GBNF grammar element. Produced by <see cref="GbnfParser"/>,
/// consumed by <see cref="CompiledGrammar"/>.
/// </summary>
internal abstract record GbnfNode
{
    /// <summary>String literal, e.g., <c>"hello"</c>.</summary>
    internal sealed record Literal(string Text) : GbnfNode;

    /// <summary>Character class <c>[a-z]</c> or <c>[^a-z]</c>.</summary>
    internal sealed record CharClass(CharRange[] Ranges, bool Negated) : GbnfNode;

    /// <summary>Reference to another rule by name.</summary>
    internal sealed record RuleRef(string RuleName) : GbnfNode;

    /// <summary>Sequence (concatenation) of elements.</summary>
    internal sealed record Sequence(GbnfNode[] Elements) : GbnfNode;

    /// <summary>Alternation: one of the alternatives.</summary>
    internal sealed record Alternation(GbnfNode[] Alternatives) : GbnfNode;

    /// <summary>
    /// Repetition: <paramref name="Min"/>..<paramref name="Max"/> occurrences.
    /// <c>*</c> = (0, MaxValue), <c>+</c> = (1, MaxValue), <c>?</c> = (0, 1).
    /// </summary>
    internal sealed record Repeat(GbnfNode Child, int Min, int Max) : GbnfNode;
}
