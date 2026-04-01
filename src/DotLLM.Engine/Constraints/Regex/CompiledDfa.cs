namespace DotLLM.Engine.Constraints.Regex;

/// <summary>
/// Immutable compiled DFA with equivalence class compression.
/// Thread-safe — shared across all <see cref="RegexConstraint"/> clones.
/// </summary>
/// <remarks>
/// Instead of a <c>state × 65536</c> transition table, all characters are mapped to
/// a small number of equivalence classes (typically 10–50 for practical patterns).
/// The transition table is <c>StateCount × ClassCount</c> ints.
/// </remarks>
internal sealed class CompiledDfa
{
    /// <summary>Number of DFA states.</summary>
    public int StateCount { get; }

    /// <summary>The start state (always 0 after minimization).</summary>
    public int StartState => 0;

    /// <summary>Whether each state is accepting (output complete).</summary>
    public bool[] IsAccepting { get; }

    /// <summary>
    /// Maps char → equivalence class index. 65536 entries (one per UTF-16 code unit).
    /// All chars in the same class produce identical transitions in every DFA state.
    /// </summary>
    public byte[] CharToClass { get; }

    /// <summary>Number of distinct equivalence classes.</summary>
    public int ClassCount { get; }

    /// <summary>
    /// Flat transition table: <c>Transitions[state * ClassCount + classId]</c> = next state.
    /// -1 means dead state (no valid transition).
    /// </summary>
    public int[] Transitions { get; }

    /// <summary>
    /// For each state, the set of equivalence class IDs that have valid (non-dead) transitions.
    /// Used to quickly enumerate valid transitions without scanning all classes.
    /// </summary>
    public int[][] ValidClassesPerState { get; }

    /// <summary>
    /// Creates a compiled DFA from pre-built tables.
    /// </summary>
    internal CompiledDfa(int stateCount, bool[] isAccepting, byte[] charToClass,
                         int classCount, int[] transitions, int[][] validClassesPerState)
    {
        StateCount = stateCount;
        IsAccepting = isAccepting;
        CharToClass = charToClass;
        ClassCount = classCount;
        Transitions = transitions;
        ValidClassesPerState = validClassesPerState;
    }
}
