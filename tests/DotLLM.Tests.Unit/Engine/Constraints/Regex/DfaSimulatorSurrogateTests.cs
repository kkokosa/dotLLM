using DotLLM.Engine.Constraints.Regex;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Constraints.Regex;

/// <summary>
/// Regression tests for surrogate handling in <see cref="DfaSimulator.TryAdvance"/>.
/// The DFA equivalence-class table is indexed by <see cref="char"/> (UTF-16 code
/// unit), so supplementary code points (emoji, CJK Extension B) arrive as
/// surrogate pairs. Before the fix, each half was looked up by code unit and
/// silently matched some unrelated equivalence class — typically the catch-all
/// dot class — producing wrong matches. The simulator now routes surrogates
/// directly to dead state, making the BMP-only limitation explicit. See upstream
/// issue #107 item 7.
/// </summary>
public class DfaSimulatorSurrogateTests
{
    private static CompiledDfa Compile(string pattern)
    {
        var ast = RegexParser.Parse(pattern);
        var nfa = NfaBuilder.Build(ast);
        return DfaBuilder.Build(nfa);
    }

    /// <summary>
    /// A `.` (any-char) pattern intentionally matches any BMP character. A
    /// supplementary code point such as U+1F600 (😀) arrives as a high-surrogate
    /// + low-surrogate pair; advancing on the high surrogate must enter the
    /// dead state, not silently accept the emoji as "one of any char". The DFA
    /// only models 16-bit code units, so accepting half a surrogate pair is
    /// always wrong.
    /// </summary>
    [Fact]
    public void TryAdvance_HighSurrogate_EntersDeadState()
    {
        var dfa = Compile(".");
        var sim = new DfaSimulator(dfa);

        // U+1F600 ('😀') as UTF-16 surrogate pair.
        const string emoji = "😀";
        Assert.Equal(2, emoji.Length);
        Assert.True(char.IsHighSurrogate(emoji[0]));

        Assert.False(sim.TryAdvance(emoji[0]));
        Assert.True(sim.IsDead);
    }

    /// <summary>
    /// A stand-alone low surrogate must also be rejected (defence against the
    /// reverse-order case of advancing on the second half first).
    /// </summary>
    [Fact]
    public void TryAdvance_LowSurrogate_EntersDeadState()
    {
        var dfa = Compile(".");
        var sim = new DfaSimulator(dfa);

        const char lowSurrogate = '\uDE00';
        Assert.True(char.IsLowSurrogate(lowSurrogate));

        Assert.False(sim.TryAdvance(lowSurrogate));
        Assert.True(sim.IsDead);
    }

    /// <summary>
    /// Non-regression: ordinary BMP characters still advance through `.` and
    /// reach the accepting state.
    /// </summary>
    [Fact]
    public void TryAdvance_BmpCharacterThroughDotPattern_Accepts()
    {
        var dfa = Compile(".");
        var sim = new DfaSimulator(dfa);

        Assert.True(sim.TryAdvance('a'));
        Assert.True(sim.IsAccepting);
    }

    /// <summary>
    /// Non-regression: a high BMP character near the surrogate range (U+D7FF —
    /// just below the surrogate block) is still accepted by `.`.
    /// </summary>
    [Fact]
    public void TryAdvance_BoundaryBmpCharacter_Accepts()
    {
        var dfa = Compile(".");
        var sim = new DfaSimulator(dfa);

        const char justBelowSurrogates = '퟿';
        Assert.False(char.IsSurrogate(justBelowSurrogates));
        Assert.True(sim.TryAdvance(justBelowSurrogates));
        Assert.True(sim.IsAccepting);
    }
}
