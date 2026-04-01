using DotLLM.Engine.Constraints.Regex;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints.Regex;

public class DfaBuilderTests
{
    private static CompiledDfa Compile(string pattern)
    {
        var ast = RegexParser.Parse(pattern);
        var nfa = NfaBuilder.Build(ast);
        return DfaBuilder.Build(nfa);
    }

    [Fact]
    public void Build_SingleLiteral_HasCorrectStates()
    {
        var dfa = Compile("a");
        // Should have 2 states: start (non-accepting) and after-a (accepting)
        Assert.True(dfa.StateCount >= 2);
        Assert.False(dfa.IsAccepting[dfa.StartState]);
    }

    [Fact]
    public void Build_SingleLiteral_AcceptsAfterMatch()
    {
        var dfa = Compile("a");
        var sim = new DfaSimulator(dfa);
        Assert.True(sim.TryAdvance('a'));
        Assert.True(sim.IsAccepting);
    }

    [Fact]
    public void Build_SingleLiteral_RejectsMismatch()
    {
        var dfa = Compile("a");
        var sim = new DfaSimulator(dfa);
        Assert.False(sim.TryAdvance('b'));
        Assert.True(sim.IsDead);
    }

    [Fact]
    public void Build_Concatenation_AcceptsFullMatch()
    {
        var dfa = Compile("abc");
        var sim = new DfaSimulator(dfa);
        Assert.True(sim.TryAdvance('a'));
        Assert.False(sim.IsAccepting);
        Assert.True(sim.TryAdvance('b'));
        Assert.False(sim.IsAccepting);
        Assert.True(sim.TryAdvance('c'));
        Assert.True(sim.IsAccepting);
    }

    [Fact]
    public void Build_Alternation_AcceptsBothBranches()
    {
        var dfa = Compile("a|b");

        var sim1 = new DfaSimulator(dfa);
        Assert.True(sim1.TryAdvance('a'));
        Assert.True(sim1.IsAccepting);

        var sim2 = new DfaSimulator(dfa);
        Assert.True(sim2.TryAdvance('b'));
        Assert.True(sim2.IsAccepting);
    }

    [Fact]
    public void Build_KleeneStar_AcceptsZeroOrMore()
    {
        var dfa = Compile("a*");
        // Zero occurrences: accepting at start
        var sim0 = new DfaSimulator(dfa);
        Assert.True(sim0.IsAccepting);

        // One occurrence
        var sim1 = new DfaSimulator(dfa);
        Assert.True(sim1.TryAdvance('a'));
        Assert.True(sim1.IsAccepting);

        // Multiple
        var sim3 = new DfaSimulator(dfa);
        Assert.True(sim3.TryAdvance('a'));
        Assert.True(sim3.TryAdvance('a'));
        Assert.True(sim3.TryAdvance('a'));
        Assert.True(sim3.IsAccepting);
    }

    [Fact]
    public void Build_Plus_RequiresAtLeastOne()
    {
        var dfa = Compile("a+");
        var sim0 = new DfaSimulator(dfa);
        Assert.False(sim0.IsAccepting); // zero not accepted

        var sim1 = new DfaSimulator(dfa);
        Assert.True(sim1.TryAdvance('a'));
        Assert.True(sim1.IsAccepting);
    }

    [Fact]
    public void Build_CharClass_MatchesRange()
    {
        var dfa = Compile("[a-z]");
        foreach (char c in "azm")
        {
            var sim = new DfaSimulator(dfa);
            Assert.True(sim.TryAdvance(c), $"Should accept '{c}'");
            Assert.True(sim.IsAccepting);
        }

        var simBad = new DfaSimulator(dfa);
        Assert.False(simBad.TryAdvance('0'));
    }

    [Fact]
    public void Build_NegatedCharClass_MatchesComplement()
    {
        var dfa = Compile("[^0-9]");
        var sim1 = new DfaSimulator(dfa);
        Assert.True(sim1.TryAdvance('a'));
        Assert.True(sim1.IsAccepting);

        var sim2 = new DfaSimulator(dfa);
        Assert.False(sim2.TryAdvance('5'));
    }

    [Fact]
    public void Build_DatePattern_AcceptsValidDate()
    {
        var dfa = Compile("\\d{4}-\\d{2}-\\d{2}");
        var sim = new DfaSimulator(dfa);
        foreach (char c in "2024-01-15")
        {
            Assert.True(sim.TryAdvance(c), $"Should accept '{c}' in date");
        }
        Assert.True(sim.IsAccepting);
    }

    [Fact]
    public void Build_DatePattern_RejectsIncomplete()
    {
        var dfa = Compile("\\d{4}-\\d{2}-\\d{2}");
        var sim = new DfaSimulator(dfa);
        foreach (char c in "2024-01")
            Assert.True(sim.TryAdvance(c));
        Assert.False(sim.IsAccepting);
    }

    [Fact]
    public void Build_WordAlternation_AcceptsBoth()
    {
        var dfa = Compile("(yes|no)");
        var sim1 = new DfaSimulator(dfa);
        foreach (char c in "yes") Assert.True(sim1.TryAdvance(c));
        Assert.True(sim1.IsAccepting);

        var sim2 = new DfaSimulator(dfa);
        foreach (char c in "no") Assert.True(sim2.TryAdvance(c));
        Assert.True(sim2.IsAccepting);
    }

    [Fact]
    public void Build_Minimization_ReducesStates()
    {
        // a* should minimize to a very small DFA
        var dfa = Compile("a*");
        Assert.True(dfa.StateCount <= 2, $"Expected ≤2 states for a*, got {dfa.StateCount}");
    }

    [Fact]
    public void Build_EquivalenceClasses_AreCompact()
    {
        // \d should produce ~2 equivalence classes (digits vs everything else)
        var dfa = Compile("\\d");
        Assert.True(dfa.ClassCount <= 5, $"Expected ≤5 equivalence classes for \\d, got {dfa.ClassCount}");
    }
}
