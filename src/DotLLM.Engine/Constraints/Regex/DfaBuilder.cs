using System.Runtime.CompilerServices;

namespace DotLLM.Engine.Constraints.Regex;

/// <summary>
/// Converts a Thompson NFA to a minimized DFA via:
/// <list type="number">
/// <item>Equivalence class extraction (partition chars by NFA transition behavior)</item>
/// <item>Subset construction (powerset: NFA-state-set → DFA state)</item>
/// <item>Hopcroft minimization (merge equivalent DFA states)</item>
/// </list>
/// </summary>
internal static class DfaBuilder
{
    /// <summary>Maximum DFA states before aborting. Guards against state explosion.</summary>
    private const int MaxDfaStates = 10_000;

    /// <summary>
    /// Builds a minimized DFA from an NFA.
    /// </summary>
    /// <param name="nfa">The Thompson NFA.</param>
    /// <returns>A compiled, immutable DFA with equivalence class compression.</returns>
    /// <exception cref="ArgumentException">If the DFA exceeds <see cref="MaxDfaStates"/> states.</exception>
    public static CompiledDfa Build(Nfa nfa)
    {
        // Step 1: Compute equivalence classes from NFA transitions
        var (charToClass, classCount, classRepresentatives) = ComputeEquivalenceClasses(nfa);

        // Step 2: Subset construction
        var (dfaTransitions, dfaAccepting, dfaStateCount) =
            SubsetConstruction(nfa, charToClass, classCount, classRepresentatives);

        // Step 3: Hopcroft minimization
        return HopcroftMinimize(dfaTransitions, dfaAccepting, dfaStateCount, charToClass, classCount);
    }

    /// <summary>
    /// Partitions all 65536 char values into equivalence classes where all chars in a class
    /// produce identical behavior across all NFA transitions.
    /// </summary>
    private static (byte[] charToClass, int classCount, char[] classRepresentatives)
        ComputeEquivalenceClasses(Nfa nfa)
    {
        // Collect all boundary points from NFA transitions.
        // Each char range [lo, hi] creates boundaries at lo and hi+1.
        var boundaries = new SortedSet<int> { 0 }; // always start a class at 0

        for (int state = 0; state < nfa.StateCount; state++)
        {
            foreach (var trans in nfa.GetTransitions(state))
            {
                if (trans.Ranges is null) continue;
                foreach (var range in trans.Ranges)
                {
                    boundaries.Add(range.Lo);
                    if (range.Hi < char.MaxValue)
                        boundaries.Add(range.Hi + 1);
                }
            }
        }

        // Assign class IDs sequentially from sorted boundaries
        var charToClass = new byte[65536];
        var sortedBoundaries = new List<int>(boundaries);
        int classCount = sortedBoundaries.Count;

        if (classCount > 255)
            throw new ArgumentException(
                $"Regex pattern produces {classCount} character equivalence classes (max 255). " +
                "The pattern has too many distinct character-range boundaries for byte-indexed DFA compression.");

        var classReps = new char[classCount];
        for (int i = 0; i < sortedBoundaries.Count; i++)
        {
            int lo = sortedBoundaries[i];
            int hi = (i + 1 < sortedBoundaries.Count) ? sortedBoundaries[i + 1] - 1 : 65535;
            classReps[i] = (char)lo;
            byte classId = (byte)i;
            for (int c = lo; c <= hi && c <= 65535; c++)
                charToClass[c] = classId;
        }

        return (charToClass, classCount, classReps);
    }

    /// <summary>
    /// Standard subset construction: each DFA state is a set of NFA states.
    /// </summary>
    private static (int[] transitions, bool[] accepting, int stateCount)
        SubsetConstruction(Nfa nfa, byte[] charToClass, int classCount, char[] classRepresentatives)
    {
        // Epsilon closure helper
        var closureStack = new Stack<int>();
        var closureSet = new HashSet<int>();

        void EpsilonClosure(HashSet<int> stateSet)
        {
            closureStack.Clear();
            foreach (int s in stateSet)
                closureStack.Push(s);

            while (closureStack.Count > 0)
            {
                int s = closureStack.Pop();
                foreach (var trans in nfa.GetTransitions(s))
                {
                    if (trans.IsEpsilon && stateSet.Add(trans.Target))
                        closureStack.Push(trans.Target);
                }
            }
        }

        // DFA state → NFA state set mapping
        var stateMap = new Dictionary<string, int>(); // serialized NFA state set → DFA state ID
        var dfaStates = new List<HashSet<int>>();
        var transitionsList = new List<int[]>(); // each entry: classCount transitions for that DFA state
        var acceptingList = new List<bool>();
        var worklist = new Queue<int>();

        string SerializeStateSet(HashSet<int> set)
        {
            var sorted = new int[set.Count];
            set.CopyTo(sorted);
            Array.Sort(sorted);
            return string.Join(',', sorted);
        }

        int GetOrCreateDfaState(HashSet<int> nfaStates)
        {
            var key = SerializeStateSet(nfaStates);
            if (stateMap.TryGetValue(key, out int existing))
                return existing;

            if (stateMap.Count >= MaxDfaStates)
                throw new ArgumentException(
                    $"DFA state limit ({MaxDfaStates}) exceeded. The regex pattern is too complex for DFA construction.");

            int id = dfaStates.Count;
            stateMap[key] = id;
            dfaStates.Add(new HashSet<int>(nfaStates));
            transitionsList.Add(new int[classCount]);
            Array.Fill(transitionsList[id], -1); // -1 = dead state
            acceptingList.Add(nfaStates.Contains(nfa.AcceptState));
            worklist.Enqueue(id);
            return id;
        }

        // Initial DFA state: epsilon closure of NFA start state
        var startSet = new HashSet<int> { nfa.StartState };
        EpsilonClosure(startSet);
        GetOrCreateDfaState(startSet);

        // Process worklist
        while (worklist.Count > 0)
        {
            int dfaState = worklist.Dequeue();
            var nfaStates = dfaStates[dfaState];

            for (int classId = 0; classId < classCount; classId++)
            {
                char rep = classRepresentatives[classId];
                var targetSet = new HashSet<int>();

                foreach (int nfaState in nfaStates)
                {
                    foreach (var trans in nfa.GetTransitions(nfaState))
                    {
                        if (!trans.IsEpsilon && trans.Matches(rep))
                            targetSet.Add(trans.Target);
                    }
                }

                if (targetSet.Count == 0)
                    continue; // stays at -1 (dead state)

                EpsilonClosure(targetSet);
                int targetDfaState = GetOrCreateDfaState(targetSet);
                transitionsList[dfaState][classId] = targetDfaState;
            }
        }

        // Flatten to arrays
        int stateCount = dfaStates.Count;
        var transitions = new int[stateCount * classCount];
        var accepting = new bool[stateCount];
        for (int s = 0; s < stateCount; s++)
        {
            accepting[s] = acceptingList[s];
            Array.Copy(transitionsList[s], 0, transitions, s * classCount, classCount);
        }

        return (transitions, accepting, stateCount);
    }

    /// <summary>
    /// Hopcroft minimization: partition DFA states into equivalence classes
    /// (states that behave identically for all inputs), then rebuild the DFA.
    /// </summary>
    private static CompiledDfa HopcroftMinimize(
        int[] transitions, bool[] accepting, int stateCount,
        byte[] charToClass, int classCount)
    {
        if (stateCount <= 1)
            return BuildCompiledDfa(transitions, accepting, stateCount, charToClass, classCount);

        // Initial partition: accepting vs non-accepting
        var partition = new int[stateCount]; // state → partition ID
        int numPartitions = 0;

        // Separate accepting and non-accepting states
        bool hasAccepting = false, hasNonAccepting = false;
        for (int s = 0; s < stateCount; s++)
        {
            if (accepting[s]) hasAccepting = true;
            else hasNonAccepting = true;
        }

        if (hasAccepting && hasNonAccepting)
        {
            for (int s = 0; s < stateCount; s++)
                partition[s] = accepting[s] ? 0 : 1;
            numPartitions = 2;
        }
        else
        {
            numPartitions = 1; // all states are the same type
        }

        // Iteratively refine partitions
        bool changed = true;
        while (changed)
        {
            changed = false;
            var newPartitionMap = new Dictionary<string, int>();
            var newPartition = new int[stateCount];
            int newNumPartitions = 0;

            for (int s = 0; s < stateCount; s++)
            {
                // Build signature: current partition + partition of target for each class
                var sig = new int[classCount + 1];
                sig[0] = partition[s];
                for (int c = 0; c < classCount; c++)
                {
                    int target = transitions[s * classCount + c];
                    sig[c + 1] = target < 0 ? -1 : partition[target];
                }

                string key = string.Join(',', sig);
                if (!newPartitionMap.TryGetValue(key, out int partId))
                {
                    partId = newNumPartitions++;
                    newPartitionMap[key] = partId;
                }
                newPartition[s] = partId;
            }

            if (newNumPartitions > numPartitions)
            {
                changed = true;
                Array.Copy(newPartition, partition, stateCount);
                numPartitions = newNumPartitions;
            }
        }

        // Build minimized DFA
        int minStateCount = numPartitions;
        var minTransitions = new int[minStateCount * classCount];
        var minAccepting = new bool[minStateCount];
        Array.Fill(minTransitions, -1);

        // Find representative for each partition and map start state to 0
        var repState = new int[minStateCount];
        Array.Fill(repState, -1);

        // Ensure the start state's partition becomes partition 0
        int startPartition = partition[0];
        if (startPartition != 0)
        {
            // Swap partition IDs so start state is in partition 0
            for (int s = 0; s < stateCount; s++)
            {
                if (partition[s] == 0) partition[s] = startPartition;
                else if (partition[s] == startPartition) partition[s] = 0;
            }
        }

        for (int s = 0; s < stateCount; s++)
        {
            int p = partition[s];
            if (repState[p] < 0)
            {
                repState[p] = s;
                minAccepting[p] = accepting[s];
                for (int c = 0; c < classCount; c++)
                {
                    int target = transitions[s * classCount + c];
                    minTransitions[p * classCount + c] = target < 0 ? -1 : partition[target];
                }
            }
        }

        return BuildCompiledDfa(minTransitions, minAccepting, minStateCount, charToClass, classCount);
    }

    private static CompiledDfa BuildCompiledDfa(
        int[] transitions, bool[] accepting, int stateCount,
        byte[] charToClass, int classCount)
    {
        // Build valid classes per state (for fast enumeration)
        var validClasses = new int[stateCount][];
        for (int s = 0; s < stateCount; s++)
        {
            var valid = new List<int>();
            for (int c = 0; c < classCount; c++)
            {
                if (transitions[s * classCount + c] >= 0)
                    valid.Add(c);
            }
            validClasses[s] = [.. valid];
        }

        return new CompiledDfa(stateCount, accepting, charToClass, classCount, transitions, validClasses);
    }
}
