using DotLLM.Engine.Constraints.Grammar;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints.Grammar;

public class PdaSimulatorTests
{
    private static PdaSimulator CreateSimulator(string gbnf)
    {
        var ast = GbnfParser.Parse(gbnf);
        var compiled = CompiledGrammar.Compile(ast);
        return new PdaSimulator(compiled);
    }

    [Fact]
    public void TryAdvance_SimpleLiteral_AcceptsExactMatch()
    {
        var sim = CreateSimulator("root ::= \"hello\"");
        foreach (char c in "hello")
            Assert.True(sim.TryAdvance(c), $"Should accept '{c}'");
        Assert.True(sim.IsAccepting);
    }

    [Fact]
    public void TryAdvance_SimpleLiteral_RejectsMismatch()
    {
        var sim = CreateSimulator("root ::= \"hello\"");
        Assert.True(sim.TryAdvance('h'));
        Assert.False(sim.TryAdvance('x')); // expected 'e'
    }

    [Fact]
    public void TryAdvance_SimpleLiteral_RejectsExtraChars()
    {
        var sim = CreateSimulator("root ::= \"hi\"");
        Assert.True(sim.TryAdvance('h'));
        Assert.True(sim.TryAdvance('i'));
        Assert.True(sim.IsAccepting);
        Assert.False(sim.TryAdvance('!')); // no more chars allowed
    }

    [Fact]
    public void TryAdvance_Alternation_AcceptsBothBranches()
    {
        var gbnf = "root ::= \"yes\" | \"no\"";

        var sim1 = CreateSimulator(gbnf);
        foreach (char c in "yes") Assert.True(sim1.TryAdvance(c));
        Assert.True(sim1.IsAccepting);

        var sim2 = CreateSimulator(gbnf);
        foreach (char c in "no") Assert.True(sim2.TryAdvance(c));
        Assert.True(sim2.IsAccepting);
    }

    [Fact]
    public void TryAdvance_CharClass_AcceptsMatchingChars()
    {
        var sim = CreateSimulator("root ::= [a-z]");
        Assert.True(sim.TryAdvance('m'));
        Assert.True(sim.IsAccepting);
    }

    [Fact]
    public void TryAdvance_CharClass_RejectsNonMatchingChars()
    {
        var sim = CreateSimulator("root ::= [a-z]");
        Assert.False(sim.TryAdvance('5'));
    }

    [Fact]
    public void TryAdvance_Repetition_Plus_RequiresOne()
    {
        var sim = CreateSimulator("root ::= [a-z]+");
        // Can accept is false before any char
        Assert.False(sim.IsAccepting);

        Assert.True(sim.TryAdvance('a'));
        Assert.True(sim.IsAccepting); // after 1+ chars
        Assert.True(sim.TryAdvance('b'));
        Assert.True(sim.IsAccepting); // still accepting
    }

    [Fact]
    public void TryAdvance_Repetition_Star_AcceptsEmpty()
    {
        var sim = CreateSimulator("root ::= [a-z]*");
        // Star allows zero matches — should be able to accept immediately
        Assert.True(sim.CanAccept());
    }

    [Fact]
    public void TryAdvance_RuleReference_FollowsRule()
    {
        var sim = CreateSimulator(
            """
            root ::= greeting
            greeting ::= "hi"
            """);
        Assert.True(sim.TryAdvance('h'));
        Assert.True(sim.TryAdvance('i'));
        Assert.True(sim.IsAccepting);
    }

    [Fact]
    public void TryAdvance_NestedRules_WorksCorrectly()
    {
        var sim = CreateSimulator(
            """
            root ::= greeting " " name
            greeting ::= "hi"
            name ::= [A-Z] [a-z]+
            """);

        foreach (char c in "hi")
            Assert.True(sim.TryAdvance(c), $"Should accept '{c}' in greeting");
        Assert.True(sim.TryAdvance(' '), "Should accept space");
        Assert.True(sim.TryAdvance('J'), "Should accept uppercase");
        Assert.True(sim.TryAdvance('o'), "Should accept lowercase");
        Assert.True(sim.TryAdvance('e'), "Should accept lowercase");
        Assert.True(sim.IsAccepting);
    }

    [Fact]
    public void Reset_ReturnsToInitial()
    {
        var sim = CreateSimulator("root ::= \"ab\"");
        Assert.True(sim.TryAdvance('a'));
        sim.Reset();
        // After reset, should be back at start
        Assert.False(sim.IsAccepting);
        Assert.True(sim.TryAdvance('a'));
        Assert.True(sim.TryAdvance('b'));
        Assert.True(sim.IsAccepting);
    }

    [Fact]
    public void CanAccept_WhenNotAccepting_ReturnsFalse()
    {
        var sim = CreateSimulator("root ::= \"abc\"");
        Assert.True(sim.TryAdvance('a'));
        Assert.False(sim.CanAccept()); // not done yet
    }

    [Fact]
    public void CanAccept_WhenAccepting_ReturnsTrue()
    {
        var sim = CreateSimulator("root ::= \"ab\"");
        Assert.True(sim.TryAdvance('a'));
        Assert.True(sim.TryAdvance('b'));
        Assert.True(sim.CanAccept());
    }
}
