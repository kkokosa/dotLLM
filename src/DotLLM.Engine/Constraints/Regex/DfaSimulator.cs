using System.Runtime.CompilerServices;

namespace DotLLM.Engine.Constraints.Regex;

/// <summary>
/// Zero-allocation DFA simulator. Tracks only the current DFA state.
/// Copies by value — struct copy is a single <c>int</c> field, making per-token
/// cloning during vocabulary scan essentially free.
/// </summary>
internal struct DfaSimulator
{
    private readonly CompiledDfa _dfa;
    private int _state;

    /// <summary>Creates a simulator at the DFA start state.</summary>
    public DfaSimulator(CompiledDfa dfa)
    {
        _dfa = dfa;
        _state = dfa.StartState;
    }

    /// <summary>Whether the current state is accepting (complete match).</summary>
    public readonly bool IsAccepting
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _state >= 0 && _dfa.IsAccepting[_state];
    }

    /// <summary>Whether the DFA has entered a dead state (no further transitions possible).</summary>
    public readonly bool IsDead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _state < 0;
    }

    /// <summary>Current DFA state ID. Used as the cache key for token mask lookup.</summary>
    public readonly int State
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _state;
    }

    /// <summary>
    /// Attempts to advance by one character.
    /// Returns false if no transition exists (enters dead state).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Regex constraints currently only support BMP Unicode scalar values — U+0000–U+D7FF
    /// and U+E000–U+FFFF, i.e. the BMP excluding surrogate code units. The
    /// DFA is built against equivalence classes over <see cref="char"/> (UTF-16 code
    /// units). A supplementary code point (e.g. U+1F600) arrives as a surrogate pair
    /// (high + low surrogate); without explicit handling, each half would index
    /// <see cref="CompiledDfa.CharToClass"/> at the surrogate code unit and route to
    /// whatever class happened to be assigned there — typically the catch-all class —
    /// silently mismatching the intended character.
    /// </para>
    /// <para>
    /// To make the limitation explicit, this method routes any UTF-16 surrogate code
    /// unit (<c>U+D800–U+DFFF</c>) directly to the dead state. Callers see a clean
    /// "not allowed" rather than a wrong match.
    /// </para>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryAdvance(char c)
    {
        if (_state < 0)
            return false;

        // BMP-only: surrogate code units are rejected outright. See <remarks/>.
        if (char.IsSurrogate(c))
        {
            _state = -1;
            return false;
        }

        int classId = _dfa.CharToClass[c];
        int next = _dfa.Transitions[_state * _dfa.ClassCount + classId];
        if (next < 0)
        {
            _state = -1; // dead state
            return false;
        }

        _state = next;
        return true;
    }

    /// <summary>Resets to the start state.</summary>
    public void Reset()
    {
        _state = _dfa.StartState;
    }
}
