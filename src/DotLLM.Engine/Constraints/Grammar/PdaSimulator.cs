using System.Runtime.CompilerServices;

namespace DotLLM.Engine.Constraints.Grammar;

/// <summary>
/// Zero-allocation pushdown automaton simulator for grammar-constrained decoding.
/// Value type with <see cref="InlineArrayAttribute"/>-based stacks — copies by value for cloning.
/// </summary>
/// <remarks>
/// PDA state: current position index + call return stack.
/// Maximum nesting depth: 64 (same limit as <c>JsonCharParser</c>).
/// Struct size: ~268 bytes.
/// </remarks>
internal struct PdaSimulator
{
    private const int MaxDepth = 64;
    private const int MaxEpsilonSteps = 1000; // guard against infinite loops

    private readonly CompiledGrammar _grammar;
    private int _position;
    private ReturnStack _returnStack;
    private int _stackDepth;
    private bool _accepted;

    /// <summary>Creates a simulator at the grammar root entry position.</summary>
    public PdaSimulator(CompiledGrammar grammar)
    {
        _grammar = grammar;
        _position = grammar.RootEntry;
        _stackDepth = 0;
        _accepted = false;
        _returnStack = default;

        // Resolve initial epsilon transitions (rule calls, joins) to reach
        // the first terminal or alternation position.
        ResolveEpsilons(ref _position, ref _stackDepth, ref _returnStack, ref _accepted);
    }

    /// <summary>Whether the PDA has reached or can reach acceptance (grammar fully matched).</summary>
    public readonly bool IsAccepting => _accepted || CanAccept();

    /// <summary>
    /// Returns a state key for mask caching. Combines position, stack depth,
    /// and top-of-stack return position.
    /// </summary>
    public readonly GrammarStateKey GetEffectiveStateKey()
    {
        // Hash the full stack to avoid aliasing distinct call contexts.
        // Different stack contents at the same position can allow different tokens.
        var hash = new HashCode();
        hash.Add(_position);
        hash.Add(_stackDepth);
        hash.Add(_accepted ? 1 : 0);
        for (int i = 0; i < _stackDepth; i++)
            hash.Add(_returnStack[i]);
        return new GrammarStateKey(hash.ToHashCode());
    }

    /// <summary>
    /// Attempts to advance by one character. Resolves epsilon transitions
    /// (rule calls, returns, alternations, joins) internally before matching a terminal.
    /// Returns false if no valid transition exists for the character.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryAdvance(char c)
    {
        if (_accepted)
            return false; // grammar fully matched, no more chars allowed

        // Try to match char at all reachable terminal positions
        return TryAdvanceFromPosition(c, ref _position, ref _stackDepth, ref _returnStack, ref _accepted);
    }

    /// <summary>Resets to the initial state.</summary>
    public void Reset()
    {
        _position = _grammar.RootEntry;
        _stackDepth = 0;
        _accepted = false;
        _returnStack = default;
        ResolveEpsilons(ref _position, ref _stackDepth, ref _returnStack, ref _accepted);
    }

    /// <summary>
    /// Core advance logic: resolve epsilon transitions, then try to match the character.
    /// Uses a worklist approach for alternation exploration.
    /// </summary>
    private readonly bool TryAdvanceFromPosition(
        char c,
        ref int position, ref int stackDepth,
        ref ReturnStack returnStack, ref bool accepted)
    {
        // Explore reachable terminal positions via epsilon transitions.
        // We use a simple stack-based DFS of (position, stackDepth, returnStack) tuples.
        // On finding a terminal match, we commit to that path.

        // Fast path: current position is already a terminal
        var pos = _grammar.Positions[position];
        if (pos.Kind == GrammarPositionKind.Terminal)
        {
            if (MatchesTerminal(pos, c))
            {
                position = pos.NextPosition;
                // Resolve post-advance epsilon transitions
                ResolveEpsilons(ref position, ref stackDepth, ref returnStack, ref accepted);
                return true;
            }
            return false;
        }

        // Slow path: resolve epsilons first to find all reachable terminals
        // We need to try alternation branches. Clone state for each branch.
        return TryAdvanceWithExploration(c, ref position, ref stackDepth, ref returnStack, ref accepted);
    }

    private readonly bool TryAdvanceWithExploration(
        char c,
        ref int position, ref int stackDepth,
        ref ReturnStack returnStack, ref bool accepted)
    {
        // Explore epsilon transitions to find a terminal that matches 'c'.
        // Uses InlineArray for the first 64 entries, spills to List on overflow.
        var inlineCandidates = new ExploreStack();
        List<ExploreState>? overflow = null;
        int exploreCount = 0;

        void AddCandidate(ExploreState s)
        {
            if (exploreCount < 64)
                inlineCandidates[exploreCount] = s;
            else
                (overflow ??= new List<ExploreState>()).Add(s);
            exploreCount++;
        }

        ExploreState GetCandidate(int i) =>
            i < 64 ? inlineCandidates[i] : overflow![i - 64];

        // Seed with current state
        AddCandidate(new ExploreState(position, stackDepth, returnStack));

        int steps = 0;
        int idx = 0;

        while (idx < exploreCount && steps < MaxEpsilonSteps)
        {
            steps++;
            var state = GetCandidate(idx++);

            if (state.Position < 0 || state.Position >= _grammar.Positions.Length)
                continue;

            var pos = _grammar.Positions[state.Position];

            switch (pos.Kind)
            {
                case GrammarPositionKind.Terminal:
                    if (MatchesTerminal(pos, c))
                    {
                        // Commit to this path
                        position = pos.NextPosition;
                        stackDepth = state.StackDepth;
                        returnStack = state.ReturnStack;
                        ResolveEpsilons(ref position, ref stackDepth, ref returnStack, ref accepted);
                        return true;
                    }
                    break;

                case GrammarPositionKind.RuleCall:
                    if (state.StackDepth < MaxDepth)
                    {
                        var newStack = state.ReturnStack;
                        newStack[state.StackDepth] = pos.NextPosition;
                        AddCandidate(new ExploreState(pos.CallTarget, state.StackDepth + 1, newStack));
                    }
                    break;

                case GrammarPositionKind.RuleReturn:
                    if (state.StackDepth > 0)
                    {
                        int returnPos = state.ReturnStack[state.StackDepth - 1];
                        AddCandidate(new ExploreState(returnPos, state.StackDepth - 1, state.ReturnStack));
                    }
                    break;

                case GrammarPositionKind.Alternation:
                    if (pos.Alternatives is not null)
                    {
                        foreach (int altEntry in pos.Alternatives)
                            AddCandidate(new ExploreState(altEntry, state.StackDepth, state.ReturnStack));
                    }
                    break;

                case GrammarPositionKind.Join:
                    if (pos.NextPosition >= 0)
                        AddCandidate(new ExploreState(pos.NextPosition, state.StackDepth, state.ReturnStack));
                    break;

                case GrammarPositionKind.Accept:
                    break;
            }
        }

        return false;
    }

    /// <summary>
    /// After advancing past a terminal, resolve epsilon transitions to reach
    /// the next terminal position (or acceptance).
    /// </summary>
    private readonly void ResolveEpsilons(
        ref int position, ref int stackDepth,
        ref ReturnStack returnStack, ref bool accepted)
    {
        int steps = 0;
        while (steps++ < MaxEpsilonSteps)
        {
            if (position < 0 || position >= _grammar.Positions.Length)
            {
                accepted = true; // fell off the end
                return;
            }

            var pos = _grammar.Positions[position];
            switch (pos.Kind)
            {
                case GrammarPositionKind.Terminal:
                case GrammarPositionKind.Alternation:
                    return; // reached a position that needs user input — stop resolving

                case GrammarPositionKind.RuleCall:
                    if (stackDepth >= MaxDepth) return; // overflow protection
                    returnStack[stackDepth++] = pos.NextPosition;
                    position = pos.CallTarget;
                    break;

                case GrammarPositionKind.RuleReturn:
                    if (stackDepth > 0)
                    {
                        position = returnStack[--stackDepth];
                    }
                    else
                    {
                        // Root rule completed — grammar accepted
                        accepted = true;
                        return;
                    }
                    break;

                case GrammarPositionKind.Join:
                    position = pos.NextPosition;
                    break;

                case GrammarPositionKind.Accept:
                    accepted = true;
                    return;
            }
        }
    }

    /// <summary>
    /// Checks whether this PDA *could* accept (for EOS token allowance).
    /// Explores epsilon transitions to find if any path leads to acceptance.
    /// </summary>
    public readonly bool CanAccept()
    {
        if (_accepted)
            return true;

        // Explore epsilon transitions to find if any path leads to acceptance.
        // Uses InlineArray for the first 64 entries, spills to List on overflow.
        var inlineCandidates = new ExploreStack();
        List<ExploreState>? overflow = null;
        int exploreCount = 0;

        void AddCandidate(ExploreState s)
        {
            if (exploreCount < 64)
                inlineCandidates[exploreCount] = s;
            else
                (overflow ??= new List<ExploreState>()).Add(s);
            exploreCount++;
        }

        ExploreState GetCandidate(int i) =>
            i < 64 ? inlineCandidates[i] : overflow![i - 64];

        AddCandidate(new ExploreState(_position, _stackDepth, _returnStack));

        int steps = 0;
        int idx = 0;

        while (idx < exploreCount && steps < MaxEpsilonSteps)
        {
            steps++;
            var state = GetCandidate(idx++);

            if (state.Position < 0 || state.Position >= _grammar.Positions.Length)
                return true; // fell off end = accepted

            var pos = _grammar.Positions[state.Position];
            switch (pos.Kind)
            {
                case GrammarPositionKind.Terminal:
                    break;

                case GrammarPositionKind.RuleReturn:
                    if (state.StackDepth > 0)
                    {
                        int returnPos = state.ReturnStack[state.StackDepth - 1];
                        AddCandidate(new ExploreState(returnPos, state.StackDepth - 1, state.ReturnStack));
                    }
                    else
                    {
                        return true; // root rule completed
                    }
                    break;

                case GrammarPositionKind.RuleCall:
                    if (state.StackDepth < MaxDepth)
                    {
                        var newStack = state.ReturnStack;
                        newStack[state.StackDepth] = pos.NextPosition;
                        AddCandidate(new ExploreState(pos.CallTarget, state.StackDepth + 1, newStack));
                    }
                    break;

                case GrammarPositionKind.Alternation:
                    if (pos.Alternatives is not null)
                    {
                        foreach (int altEntry in pos.Alternatives)
                            AddCandidate(new ExploreState(altEntry, state.StackDepth, state.ReturnStack));
                    }
                    break;

                case GrammarPositionKind.Join:
                    if (pos.NextPosition >= 0)
                        AddCandidate(new ExploreState(pos.NextPosition, state.StackDepth, state.ReturnStack));
                    break;

                case GrammarPositionKind.Accept:
                    return true;
            }
        }

        return false;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool MatchesTerminal(GrammarPosition pos, char c)
    {
        if (pos.CharRanges is null)
            return false;

        bool inRange = false;
        foreach (var range in pos.CharRanges)
        {
            if (range.Contains(c))
            {
                inRange = true;
                break;
            }
        }
        return pos.Negated ? !inRange : inRange;
    }
}

/// <summary>
/// Cache key for grammar constraint mask lookup.
/// Single hash of (position, stack depth, full stack contents, accepted flag).
/// </summary>
internal readonly record struct GrammarStateKey(int Hash);

/// <summary>InlineArray for PDA return-point stack.</summary>
[InlineArray(64)]
internal struct ReturnStack
{
    private int _element;
}

/// <summary>Exploration state for epsilon-transition search.</summary>
internal readonly record struct ExploreState(int Position, int StackDepth, ReturnStack ReturnStack);

/// <summary>InlineArray for exploration stack during epsilon-transition search.</summary>
[InlineArray(64)]
internal struct ExploreStack
{
    private ExploreState _element;
}
