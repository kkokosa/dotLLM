namespace DotLLM.Engine.Constraints.Grammar;

/// <summary>
/// Immutable compiled grammar. Thread-safe — shared across all <see cref="GrammarConstraint"/> clones.
/// Grammar rules are flattened into a position-indexed array of <see cref="GrammarPosition"/> values.
/// </summary>
internal sealed class CompiledGrammar
{
    /// <summary>Flat array of grammar positions.</summary>
    public GrammarPosition[] Positions { get; }

    /// <summary>Index of the root rule's entry position.</summary>
    public int RootEntry { get; }

    private CompiledGrammar(GrammarPosition[] positions, int rootEntry)
    {
        Positions = positions;
        RootEntry = rootEntry;
    }

    /// <summary>
    /// Compiles a GBNF grammar AST into a flat position array.
    /// </summary>
    public static CompiledGrammar Compile(GbnfGrammar grammar)
    {
        var compiler = new GrammarCompiler(grammar);
        compiler.CompileAll();
        return new CompiledGrammar([.. compiler.Positions], compiler.RootEntry);
    }

    /// <summary>
    /// Result of compiling a grammar node: the entry position (where execution starts)
    /// and the tail position (whose NextPosition should be patched to whatever follows).
    /// </summary>
    private readonly record struct Fragment(int Entry, int Tail);

    /// <summary>Internal compiler that flattens the AST into positions.</summary>
    private sealed class GrammarCompiler
    {
        private readonly GbnfGrammar _grammar;
        private readonly Dictionary<string, int> _ruleEntryPoints = new();

        public List<GrammarPosition> Positions { get; } = [];
        public int RootEntry { get; private set; }

        public GrammarCompiler(GbnfGrammar grammar)
        {
            _grammar = grammar;
        }

        public void CompileAll()
        {
            // Reserve entry points for all rules first (forward references)
            foreach (var rule in _grammar.Rules.Values)
                _ruleEntryPoints[rule.Name] = -1; // placeholder

            // Compile root rule first
            CompileRule(_grammar.RootRuleName);
            RootEntry = _ruleEntryPoints[_grammar.RootRuleName];

            // Compile remaining rules
            foreach (var rule in _grammar.Rules.Values)
            {
                if (rule.Name != _grammar.RootRuleName)
                    CompileRule(rule.Name);
            }
        }

        private void CompileRule(string ruleName)
        {
            if (_ruleEntryPoints[ruleName] >= 0)
                return; // already compiled

            // We don't know the entry point yet — the node compilation will tell us.
            // But we need to set it before compiling so forward references resolve.
            // Use a two-pass: reserve a placeholder, compile, then update.

            // Actually, we compile the node and it returns the entry point.
            // We just need to ensure the entry point is at a known position.
            // For simple cases (literal, charclass, sequence starting with literal),
            // the entry is just the next available position. For alternation/repeat,
            // the entry is the alternation node which we reserve first.

            int expectedEntry = Positions.Count;
            _ruleEntryPoints[ruleName] = expectedEntry;

            var rule = _grammar.Rules[ruleName];
            var frag = CompileNode(rule.Body);

            // If the entry point isn't where we expected (due to alternation/repeat
            // adding a reserved node), we need to redirect.
            if (frag.Entry != expectedEntry)
            {
                // The expected entry was already claimed by the node's internals.
                // We need to insert a redirect. But that's complex.
                // Simpler: always emit a Join at the expected entry that points to the real entry.
                // Actually, let's just set _ruleEntryPoints to the actual entry.
                _ruleEntryPoints[ruleName] = frag.Entry;
            }

            // Add RuleReturn at the end of the rule body
            int returnPos = Positions.Count;
            Positions.Add(new GrammarPosition
            {
                Kind = GrammarPositionKind.RuleReturn,
                NextPosition = -1
            });

            // Patch the tail to point to the return
            PatchNextPosition(frag.Tail, returnPos);
        }

        private Fragment CompileNode(GbnfNode node)
        {
            return node switch
            {
                GbnfNode.Literal lit => CompileLiteral(lit),
                GbnfNode.CharClass cc => CompileCharClass(cc),
                GbnfNode.RuleRef rr => CompileRuleRef(rr),
                GbnfNode.Sequence seq => CompileSequence(seq),
                GbnfNode.Alternation alt => CompileAlternation(alt),
                GbnfNode.Repeat rep => CompileRepeat(rep),
                _ => throw new ArgumentException($"Unknown grammar node: {node.GetType().Name}")
            };
        }

        private Fragment CompileLiteral(GbnfNode.Literal lit)
        {
            if (lit.Text.Length == 0)
                throw new ArgumentException("Empty literal in grammar.");

            int first = -1;
            int prev = -1;
            foreach (char c in lit.Text)
            {
                int pos = Positions.Count;
                Positions.Add(new GrammarPosition
                {
                    Kind = GrammarPositionKind.Terminal,
                    CharRanges = [new CharRange(c, c)],
                    Negated = false,
                    NextPosition = -1
                });

                if (first < 0) first = pos;
                if (prev >= 0) PatchNextPosition(prev, pos);
                prev = pos;
            }
            return new Fragment(first, prev);
        }

        private Fragment CompileCharClass(GbnfNode.CharClass cc)
        {
            int pos = Positions.Count;
            Positions.Add(new GrammarPosition
            {
                Kind = GrammarPositionKind.Terminal,
                CharRanges = cc.Ranges,
                Negated = cc.Negated,
                NextPosition = -1
            });
            return new Fragment(pos, pos);
        }

        private Fragment CompileRuleRef(GbnfNode.RuleRef rr)
        {
            // Ensure the referenced rule is compiled
            CompileRule(rr.RuleName);

            int pos = Positions.Count;
            Positions.Add(new GrammarPosition
            {
                Kind = GrammarPositionKind.RuleCall,
                CallTarget = _ruleEntryPoints[rr.RuleName],
                NextPosition = -1 // patched to point to next thing after rule returns
            });
            return new Fragment(pos, pos);
        }

        private Fragment CompileSequence(GbnfNode.Sequence seq)
        {
            if (seq.Elements.Length == 0)
                throw new ArgumentException("Empty sequence in grammar.");

            var first = CompileNode(seq.Elements[0]);
            var prev = first;
            for (int i = 1; i < seq.Elements.Length; i++)
            {
                var next = CompileNode(seq.Elements[i]);
                PatchNextPosition(prev.Tail, next.Entry);
                prev = next;
            }
            return new Fragment(first.Entry, prev.Tail);
        }

        private Fragment CompileAlternation(GbnfNode.Alternation alt)
        {
            if (alt.Alternatives.Length == 0)
                throw new ArgumentException("Empty alternation in grammar.");

            // Reserve the alternation node FIRST (so it's the entry point)
            int altPos = Positions.Count;
            Positions.Add(default); // placeholder — filled below

            // Compile each branch
            var branchEntries = new List<int>();
            var branchTails = new List<int>();

            foreach (var branch in alt.Alternatives)
            {
                var frag = CompileNode(branch);
                branchEntries.Add(frag.Entry);
                branchTails.Add(frag.Tail);
            }

            // Create join node (where all branches converge)
            int joinPos = Positions.Count;
            Positions.Add(new GrammarPosition
            {
                Kind = GrammarPositionKind.Join,
                NextPosition = -1 // patched by caller
            });

            // Patch all branch tails to point to the join
            foreach (int tail in branchTails)
                PatchNextPosition(tail, joinPos);

            // Fill in the alternation node
            Positions[altPos] = new GrammarPosition
            {
                Kind = GrammarPositionKind.Alternation,
                Alternatives = [.. branchEntries],
                NextPosition = joinPos
            };

            return new Fragment(altPos, joinPos);
        }

        private Fragment CompileRepeat(GbnfNode.Repeat rep)
        {
            if (rep.Min == 0 && rep.Max == int.MaxValue)
            {
                return CompileKleeneStar(rep.Child);
            }
            if (rep.Min == 1 && rep.Max == int.MaxValue)
            {
                // + = one mandatory then *
                var body = CompileNode(rep.Child);
                var star = CompileKleeneStar(rep.Child);
                PatchNextPosition(body.Tail, star.Entry);
                return new Fragment(body.Entry, star.Tail);
            }
            if (rep.Min == 0 && rep.Max == 1)
            {
                return CompileOptional(rep.Child);
            }

            // General {min, max}
            Fragment? first = null;
            Fragment prev = default;

            // Mandatory copies
            for (int i = 0; i < rep.Min; i++)
            {
                var body = CompileNode(rep.Child);
                if (first is null)
                    first = body;
                else
                    PatchNextPosition(prev.Tail, body.Entry);
                prev = body;
            }

            if (rep.Max == int.MaxValue)
            {
                // {min,}: mandatory + Kleene star
                var star = CompileKleeneStar(rep.Child);
                if (first is null)
                    return star;
                PatchNextPosition(prev.Tail, star.Entry);
                return new Fragment(first.Value.Entry, star.Tail);
            }

            // Optional copies
            for (int i = rep.Min; i < rep.Max; i++)
            {
                var opt = CompileOptional(rep.Child);
                if (first is null)
                    first = opt;
                else
                    PatchNextPosition(prev.Tail, opt.Entry);
                prev = opt;
            }

            return first is null
                ? throw new ArgumentException("Invalid repeat bounds.")
                : new Fragment(first.Value.Entry, prev.Tail);
        }

        private Fragment CompileKleeneStar(GbnfNode child)
        {
            // Reserve alternation node FIRST (it's the entry and the loop-back target)
            int altPos = Positions.Count;
            Positions.Add(default); // placeholder

            // Compile body
            var body = CompileNode(child);

            // Create join node (exit point for the loop)
            int joinPos = Positions.Count;
            Positions.Add(new GrammarPosition
            {
                Kind = GrammarPositionKind.Join,
                NextPosition = -1 // patched by caller
            });

            // Body loops back to the alternation
            PatchNextPosition(body.Tail, altPos);

            // Fill alternation: try body first, or skip to join
            Positions[altPos] = new GrammarPosition
            {
                Kind = GrammarPositionKind.Alternation,
                Alternatives = [body.Entry, joinPos],
                NextPosition = joinPos
            };

            return new Fragment(altPos, joinPos);
        }

        private Fragment CompileOptional(GbnfNode child)
        {
            // Reserve alternation node FIRST
            int altPos = Positions.Count;
            Positions.Add(default); // placeholder

            // Compile body
            var body = CompileNode(child);

            // Create join node
            int joinPos = Positions.Count;
            Positions.Add(new GrammarPosition
            {
                Kind = GrammarPositionKind.Join,
                NextPosition = -1
            });

            PatchNextPosition(body.Tail, joinPos);

            // Fill alternation: try body, or skip to join
            Positions[altPos] = new GrammarPosition
            {
                Kind = GrammarPositionKind.Alternation,
                Alternatives = [body.Entry, joinPos],
                NextPosition = joinPos
            };

            return new Fragment(altPos, joinPos);
        }

        private void PatchNextPosition(int posIndex, int target)
        {
            if (posIndex < 0 || posIndex >= Positions.Count) return;
            var pos = Positions[posIndex];
            if (pos.NextPosition < 0)
                Positions[posIndex] = pos with { NextPosition = target };
        }
    }
}

/// <summary>
/// A single position in the compiled grammar. The PDA simulator advances through these.
/// </summary>
internal record struct GrammarPosition
{
    /// <summary>Kind of this position.</summary>
    public GrammarPositionKind Kind { get; init; }

    /// <summary>For <see cref="GrammarPositionKind.Terminal"/>: character ranges to match.</summary>
    public CharRange[]? CharRanges { get; init; }

    /// <summary>For <see cref="GrammarPositionKind.Terminal"/>: whether the char class is negated.</summary>
    public bool Negated { get; init; }

    /// <summary>Index of the next position after this one (for sequence advance).</summary>
    public int NextPosition { get; init; }

    /// <summary>For <see cref="GrammarPositionKind.RuleCall"/>: index of the called rule's first position.</summary>
    public int CallTarget { get; init; }

    /// <summary>For <see cref="GrammarPositionKind.Alternation"/>: indices of alternative starting positions.</summary>
    public int[]? Alternatives { get; init; }
}

/// <summary>Position kind in the compiled grammar.</summary>
internal enum GrammarPositionKind : byte
{
    /// <summary>Match a character (terminal symbol).</summary>
    Terminal,
    /// <summary>Call into another rule (push return point).</summary>
    RuleCall,
    /// <summary>Return from rule (pop stack).</summary>
    RuleReturn,
    /// <summary>Branch: choose one alternative.</summary>
    Alternation,
    /// <summary>Join point: no-op pass-through to NextPosition.</summary>
    Join,
    /// <summary>Grammar fully matched.</summary>
    Accept,
}
