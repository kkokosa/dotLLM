namespace DotLLM.Engine.Constraints.Regex;

/// <summary>
/// Builds a Thompson NFA from a regex AST. Each NFA fragment has a single
/// start state and single accept state. Epsilon transitions are represented
/// by null character ranges.
/// </summary>
internal static class NfaBuilder
{
    /// <summary>
    /// Builds an NFA from the regex AST root node.
    /// </summary>
    public static Nfa Build(RegexNode root)
    {
        var nfa = new Nfa();
        var (start, accept) = BuildFragment(nfa, root);
        nfa.StartState = start;
        nfa.AcceptState = accept;
        return nfa;
    }

    private static (int start, int accept) BuildFragment(Nfa nfa, RegexNode node)
    {
        return node switch
        {
            RegexNode.Literal lit => BuildLiteral(nfa, lit.Ch),
            RegexNode.CharClass cc => BuildCharClass(nfa, cc.Ranges, cc.Negated),
            RegexNode.Concat cat => BuildConcat(nfa, cat.Children),
            RegexNode.Alternation alt => BuildAlternation(nfa, alt.Children),
            RegexNode.Repeat rep => BuildRepeat(nfa, rep.Child, rep.Min, rep.Max),
            _ => throw new ArgumentException($"Unknown regex AST node: {node.GetType().Name}")
        };
    }

    private static (int start, int accept) BuildLiteral(Nfa nfa, char ch)
    {
        int start = nfa.AddState();
        int accept = nfa.AddState();
        nfa.AddTransition(start, accept, [new CharRange(ch, ch)], negated: false);
        return (start, accept);
    }

    private static (int start, int accept) BuildCharClass(Nfa nfa, CharRange[] ranges, bool negated)
    {
        int start = nfa.AddState();
        int accept = nfa.AddState();
        nfa.AddTransition(start, accept, ranges, negated);
        return (start, accept);
    }

    private static (int start, int accept) BuildConcat(Nfa nfa, RegexNode[] children)
    {
        if (children.Length == 0)
            throw new ArgumentException("Empty concatenation.");

        var (start, prevAccept) = BuildFragment(nfa, children[0]);
        for (int i = 1; i < children.Length; i++)
        {
            var (fragStart, fragAccept) = BuildFragment(nfa, children[i]);
            nfa.AddEpsilon(prevAccept, fragStart);
            prevAccept = fragAccept;
        }
        return (start, prevAccept);
    }

    private static (int start, int accept) BuildAlternation(Nfa nfa, RegexNode[] children)
    {
        int start = nfa.AddState();
        int accept = nfa.AddState();

        foreach (var child in children)
        {
            var (fragStart, fragAccept) = BuildFragment(nfa, child);
            nfa.AddEpsilon(start, fragStart);
            nfa.AddEpsilon(fragAccept, accept);
        }
        return (start, accept);
    }

    private static (int start, int accept) BuildRepeat(Nfa nfa, RegexNode child, int min, int max)
    {
        if (min == 0 && max == int.MaxValue)
        {
            // Kleene star: a*
            return BuildKleeneStar(nfa, child);
        }

        if (min == 1 && max == int.MaxValue)
        {
            // a+ = a a*
            var (mandatoryStart, mandatoryAccept) = BuildFragment(nfa, child);
            var (starStart, starAccept) = BuildKleeneStar(nfa, child);
            nfa.AddEpsilon(mandatoryAccept, starStart);
            return (mandatoryStart, starAccept);
        }

        if (min == 0 && max == 1)
        {
            // a? = (a | epsilon)
            var (fragStart, fragAccept) = BuildFragment(nfa, child);
            int start = nfa.AddState();
            int accept = nfa.AddState();
            nfa.AddEpsilon(start, fragStart);
            nfa.AddEpsilon(fragAccept, accept);
            nfa.AddEpsilon(start, accept); // epsilon path
            return (start, accept);
        }

        // General {min, max}: unroll min mandatory + (max-min) optional
        return BuildBoundedRepeat(nfa, child, min, max);
    }

    private static (int start, int accept) BuildKleeneStar(Nfa nfa, RegexNode child)
    {
        int start = nfa.AddState();
        int accept = nfa.AddState();
        var (fragStart, fragAccept) = BuildFragment(nfa, child);

        nfa.AddEpsilon(start, fragStart);   // enter loop
        nfa.AddEpsilon(start, accept);       // skip (zero occurrences)
        nfa.AddEpsilon(fragAccept, fragStart); // repeat
        nfa.AddEpsilon(fragAccept, accept);  // exit loop
        return (start, accept);
    }

    private static (int start, int accept) BuildBoundedRepeat(Nfa nfa, RegexNode child, int min, int max)
    {
        // Unroll: min mandatory copies concatenated, then (max-min) optional copies
        int start = nfa.AddState();
        int current = start;

        // Mandatory copies
        for (int i = 0; i < min; i++)
        {
            var (fragStart, fragAccept) = BuildFragment(nfa, child);
            nfa.AddEpsilon(current, fragStart);
            current = fragAccept;
        }

        if (max == int.MaxValue)
        {
            // {min,} = min mandatory + Kleene star
            var (starStart, starAccept) = BuildKleeneStar(nfa, child);
            nfa.AddEpsilon(current, starStart);
            return (start, starAccept);
        }

        // Optional copies: each is (child | epsilon)
        int accept = nfa.AddState();
        nfa.AddEpsilon(current, accept); // can stop after mandatory

        for (int i = min; i < max; i++)
        {
            var (fragStart, fragAccept) = BuildFragment(nfa, child);
            nfa.AddEpsilon(current, fragStart);
            nfa.AddEpsilon(fragAccept, accept); // can stop after each optional
            current = fragAccept;
        }

        return (start, accept);
    }
}

/// <summary>
/// Thompson NFA representation. States are integers. Transitions are either
/// epsilon (null ranges) or character-based (ranges + negation flag).
/// </summary>
internal sealed class Nfa
{
    private int _stateCount;
    private readonly List<List<NfaTransition>> _transitions = [];

    /// <summary>The start state of the NFA.</summary>
    public int StartState { get; set; }

    /// <summary>The single accept state of the NFA.</summary>
    public int AcceptState { get; set; }

    /// <summary>Total number of states.</summary>
    public int StateCount => _stateCount;

    /// <summary>Gets transitions from a given state.</summary>
    public List<NfaTransition> GetTransitions(int state) => _transitions[state];

    /// <summary>Allocates a new state and returns its ID.</summary>
    public int AddState()
    {
        _transitions.Add([]);
        return _stateCount++;
    }

    /// <summary>Adds an epsilon (empty) transition.</summary>
    public void AddEpsilon(int from, int to)
    {
        _transitions[from].Add(new NfaTransition(to, null, false));
    }

    /// <summary>Adds a character/char-class transition.</summary>
    public void AddTransition(int from, int to, CharRange[] ranges, bool negated)
    {
        _transitions[from].Add(new NfaTransition(to, ranges, negated));
    }
}

/// <summary>
/// A single NFA transition. Epsilon if <see cref="Ranges"/> is null.
/// </summary>
internal readonly record struct NfaTransition(int Target, CharRange[]? Ranges, bool Negated)
{
    /// <summary>True if this is an epsilon (empty) transition.</summary>
    public bool IsEpsilon => Ranges is null;

    /// <summary>
    /// Tests whether a character matches this transition.
    /// Always false for epsilon transitions.
    /// </summary>
    public bool Matches(char c)
    {
        if (Ranges is null) return false;
        bool inRange = false;
        foreach (var range in Ranges)
        {
            if (range.Contains(c))
            {
                inRange = true;
                break;
            }
        }
        return Negated ? !inRange : inRange;
    }
}
