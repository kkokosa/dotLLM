using System.Runtime.InteropServices;

namespace DotLLM.Cpu.Threading;

/// <summary>
/// Classification of a logical processor core.
/// </summary>
public enum CoreType : byte
{
    /// <summary>Unknown or non-hybrid architecture.</summary>
    Unknown = 0,

    /// <summary>Performance core (P-core) on Intel hybrid architectures.</summary>
    Performance = 1,

    /// <summary>Efficiency core (E-core) on Intel hybrid architectures.</summary>
    Efficiency = 2,
}

/// <summary>
/// Describes a single logical processor in the system topology.
/// </summary>
/// <param name="ProcessorId">OS-assigned logical processor index.</param>
/// <param name="NumaNode">NUMA node this processor belongs to.</param>
/// <param name="PhysicalCoreId">Physical core ID (for detecting hyper-threading siblings).</param>
/// <param name="CoreType">P-core/E-core classification on hybrid architectures.</param>
public readonly record struct LogicalProcessor(int ProcessorId, int NumaNode, int PhysicalCoreId, CoreType CoreType);

/// <summary>
/// Immutable snapshot of the system's NUMA and CPU topology.
/// Detects NUMA nodes, physical cores, and hybrid (P-core/E-core) architecture on Windows and Linux.
/// Falls back to a single-node topology on unsupported platforms.
/// </summary>
public sealed partial class NumaTopology
{
    /// <summary>All detected logical processors.</summary>
    public IReadOnlyList<LogicalProcessor> Processors { get; }

    /// <summary>Number of NUMA nodes detected.</summary>
    public int NumaNodeCount { get; }

    /// <summary>Whether the system has a hybrid P-core/E-core architecture.</summary>
    public bool IsHybrid { get; }

    /// <summary>
    /// Heuristic estimate of memory channels.
    /// Desktop: 2, workstation: 4, multi-socket: 4×N.
    /// Used as a default decode thread cap when no explicit count is configured.
    /// </summary>
    public int MemoryChannelEstimate { get; }

    /// <summary>
    /// Logical processor IDs suitable for compute work.
    /// P-cores on hybrid systems, all physical cores on non-hybrid.
    /// </summary>
    public IReadOnlyList<int> PerformanceCoreIds { get; }

    /// <summary>
    /// Logical processors grouped by NUMA node.
    /// Key = NUMA node ID, Value = list of logical processor IDs.
    /// </summary>
    public IReadOnlyDictionary<int, IReadOnlyList<int>> ProcessorsByNumaNode { get; }

    private NumaTopology(IReadOnlyList<LogicalProcessor> processors)
    {
        Processors = processors;

        var byNode = new Dictionary<int, List<int>>();
        var pCores = new List<int>();
        bool hasEfficiency = false;
        bool hasPerformance = false;

        foreach (var p in processors)
        {
            if (!byNode.TryGetValue(p.NumaNode, out var list))
            {
                list = new List<int>();
                byNode[p.NumaNode] = list;
            }
            list.Add(p.ProcessorId);

            if (p.CoreType == CoreType.Efficiency) hasEfficiency = true;
            if (p.CoreType == CoreType.Performance) hasPerformance = true;
        }

        NumaNodeCount = byNode.Count;
        IsHybrid = hasEfficiency && hasPerformance;

        // Build performance core list
        if (IsHybrid)
        {
            foreach (var p in processors)
            {
                if (p.CoreType == CoreType.Performance)
                    pCores.Add(p.ProcessorId);
            }
        }
        else
        {
            // Non-hybrid: all cores are "performance" cores
            foreach (var p in processors)
                pCores.Add(p.ProcessorId);
        }

        PerformanceCoreIds = pCores;

        ProcessorsByNumaNode = byNode.ToDictionary(
            kv => kv.Key,
            kv => (IReadOnlyList<int>)kv.Value.AsReadOnly());

        // Memory channel heuristic: 2 for desktop, 4× NUMA nodes for multi-socket
        MemoryChannelEstimate = NumaNodeCount > 1 ? 4 * NumaNodeCount : 2;
    }

    /// <summary>
    /// Detects the system's NUMA and CPU topology.
    /// Platform-dispatched: uses OS APIs on Windows/Linux, fallback on other platforms.
    /// </summary>
    public static NumaTopology Detect()
    {
        if (OperatingSystem.IsWindows())
            return DetectWindows();
        if (OperatingSystem.IsLinux())
            return DetectLinux();
        return CreateFallback();
    }

    private static NumaTopology CreateFallback()
    {
        int count = Environment.ProcessorCount;
        var processors = new LogicalProcessor[count];
        for (int i = 0; i < count; i++)
            processors[i] = new LogicalProcessor(i, 0, i, CoreType.Unknown);
        return new NumaTopology(processors);
    }

    // ── Windows detection via GetLogicalProcessorInformationEx ──

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static NumaTopology DetectWindows()
    {
        try
        {
            return DetectWindowsCore();
        }
        catch
        {
            return CreateFallback();
        }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static NumaTopology DetectWindowsCore()
    {
        int cpuCount = Environment.ProcessorCount;
        if (cpuCount > 64)
        {
            // >64 processors requires processor groups — skip pinning, just detect topology
            // Log a warning and return fallback
            return CreateFallback();
        }

        // Maps: processorId → numaNode, processorId → physicalCoreId, processorId → efficiencyClass
        var numaMap = new Dictionary<int, int>();
        var coreMap = new Dictionary<int, int>();
        var efficiencyMap = new Dictionary<int, byte>();

        // Query NUMA node info
        ParseRelation(WindowsNative.LOGICAL_PROCESSOR_RELATIONSHIP.RelationNumaNode, buffer =>
        {
            ParseEntries(buffer, (entryPtr, relationship) =>
            {
                if (relationship != WindowsNative.LOGICAL_PROCESSOR_RELATIONSHIP.RelationNumaNode) return;

                // Offset: GroupMask starts at offset 0 within the union
                // NUMA_NODE_RELATIONSHIP: DWORD NodeNumber at offset 0, then GROUP_AFFINITY at offset 4
                unsafe
                {
                    int nodeNumber = *(int*)(entryPtr + 24); // union starts at 24 in the struct
                    // GROUP_AFFINITY is at offset 28 (24 + 4): KAFFINITY Mask (8 bytes), WORD Group, WORD[3] Reserved
                    ulong mask = *(ulong*)(entryPtr + 28);

                    for (int bit = 0; bit < 64; bit++)
                    {
                        if ((mask & (1UL << bit)) != 0)
                            numaMap[bit] = nodeNumber;
                    }
                }
            });
        });

        // Query processor core info (physical cores + efficiency class)
        int physicalCoreCounter = 0;
        ParseRelation(WindowsNative.LOGICAL_PROCESSOR_RELATIONSHIP.RelationProcessorCore, buffer =>
        {
            ParseEntries(buffer, (entryPtr, relationship) =>
            {
                if (relationship != WindowsNative.LOGICAL_PROCESSOR_RELATIONSHIP.RelationProcessorCore) return;

                unsafe
                {
                    // PROCESSOR_RELATIONSHIP: BYTE Flags(0), BYTE EfficiencyClass(1), BYTE[20] Reserved, WORD GroupCount(22), GROUP_AFFINITY[](24)
                    byte efficiencyClass = *(byte*)(entryPtr + 24 + 1); // +24 for union offset, +1 for EfficiencyClass
                    // GROUP_AFFINITY starts at offset 24 + 24 = 48
                    ulong mask = *(ulong*)(entryPtr + 48);

                    int coreId = physicalCoreCounter++;

                    for (int bit = 0; bit < 64; bit++)
                    {
                        if ((mask & (1UL << bit)) != 0)
                        {
                            coreMap[bit] = coreId;
                            efficiencyMap[bit] = efficiencyClass;
                        }
                    }
                }
            });
        });

        // Build processor list
        var processors = new List<LogicalProcessor>();
        for (int i = 0; i < cpuCount; i++)
        {
            int node = numaMap.TryGetValue(i, out var n) ? n : 0;
            int core = coreMap.TryGetValue(i, out var c) ? c : i;
            byte eff = efficiencyMap.TryGetValue(i, out var e) ? e : (byte)0;

            // On Win11 + Intel 12th gen+: EfficiencyClass 0 = P-core, 1 = E-core
            // On non-hybrid: all are 0 → Unknown
            CoreType coreType;
            if (efficiencyMap.Count > 0 && efficiencyMap.Values.Distinct().Count() > 1)
            {
                // Hybrid detected: lowest efficiency class = P-core
                byte minEff = efficiencyMap.Values.Min();
                coreType = eff == minEff ? CoreType.Performance : CoreType.Efficiency;
            }
            else
            {
                coreType = CoreType.Unknown;
            }

            processors.Add(new LogicalProcessor(i, node, core, coreType));
        }

        return new NumaTopology(processors);
    }

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static void ParseRelation(
        WindowsNative.LOGICAL_PROCESSOR_RELATIONSHIP relationship,
        Action<byte[]> action)
    {
        uint returnLength = 0;
        WindowsNative.GetLogicalProcessorInformationEx(relationship, IntPtr.Zero, ref returnLength);

        if (returnLength == 0) return;

        var buffer = new byte[returnLength];
        unsafe
        {
            fixed (byte* pBuffer = buffer)
            {
                if (!WindowsNative.GetLogicalProcessorInformationEx(relationship, (IntPtr)pBuffer, ref returnLength))
                    return;
            }
        }

        action(buffer);
    }

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static unsafe void ParseEntries(byte[] buffer, Action<nint, WindowsNative.LOGICAL_PROCESSOR_RELATIONSHIP> action)
    {
        fixed (byte* pBuffer = buffer)
        {
            int offset = 0;
            while (offset < buffer.Length)
            {
                // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX layout:
                // DWORD Relationship (4 bytes), DWORD Size (4 bytes), then union
                var rel = (WindowsNative.LOGICAL_PROCESSOR_RELATIONSHIP)(*(int*)(pBuffer + offset));
                int size = *(int*)(pBuffer + offset + 4);

                if (size == 0) break;

                action((nint)(pBuffer + offset), rel);

                offset += size;
            }
        }
    }

    // ── Linux detection via sysfs ──

    private static NumaTopology DetectLinux()
    {
        try
        {
            return DetectLinuxCore();
        }
        catch
        {
            return CreateFallback();
        }
    }

    private static NumaTopology DetectLinuxCore()
    {
        int cpuCount = Environment.ProcessorCount;
        var numaMap = new Dictionary<int, int>();
        var coreMap = new Dictionary<int, int>();
        var freqMap = new Dictionary<int, long>();

        // Parse NUMA node assignments from /sys/devices/system/node/
        if (Directory.Exists("/sys/devices/system/node"))
        {
            foreach (var nodeDir in Directory.GetDirectories("/sys/devices/system/node", "node*"))
            {
                string nodeName = Path.GetFileName(nodeDir);
                if (!int.TryParse(nodeName.AsSpan(4), out int nodeId)) continue;

                string cpulistPath = Path.Combine(nodeDir, "cpulist");
                if (!File.Exists(cpulistPath)) continue;

                string cpulist = File.ReadAllText(cpulistPath).Trim();
                foreach (int cpuId in ParseCpuList(cpulist))
                    numaMap[cpuId] = nodeId;
            }
        }

        // Parse core IDs and frequencies
        for (int i = 0; i < cpuCount; i++)
        {
            string coreIdPath = $"/sys/devices/system/cpu/cpu{i}/topology/core_id";
            if (File.Exists(coreIdPath))
            {
                if (int.TryParse(File.ReadAllText(coreIdPath).Trim(), out int coreId))
                    coreMap[i] = coreId;
            }

            string freqPath = $"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_max_freq";
            if (File.Exists(freqPath))
            {
                if (long.TryParse(File.ReadAllText(freqPath).Trim(), out long freq))
                    freqMap[i] = freq;
            }
        }

        // Classify P/E cores based on max frequency heuristic
        long maxFreq = freqMap.Count > 0 ? freqMap.Values.Max() : 0;
        long pCoreThreshold = (long)(maxFreq * 0.80); // ≥80% of max = P-core
        bool isHybrid = freqMap.Count > 1 && freqMap.Values.Min() < pCoreThreshold;

        var processors = new List<LogicalProcessor>();
        for (int i = 0; i < cpuCount; i++)
        {
            int node = numaMap.GetValueOrDefault(i, 0);
            int core = coreMap.GetValueOrDefault(i, i);

            CoreType coreType = CoreType.Unknown;
            if (isHybrid && freqMap.TryGetValue(i, out long freq))
                coreType = freq >= pCoreThreshold ? CoreType.Performance : CoreType.Efficiency;

            processors.Add(new LogicalProcessor(i, node, core, coreType));
        }

        return new NumaTopology(processors);
    }

    /// <summary>Parses a Linux cpulist format string (e.g., "0-3,5,7-9") into individual CPU IDs.</summary>
    internal static IEnumerable<int> ParseCpuList(string cpulist)
    {
        foreach (var part in cpulist.Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            var trimmed = part.Trim();
            int dashIdx = trimmed.IndexOf('-');
            if (dashIdx >= 0)
            {
                if (int.TryParse(trimmed.AsSpan(0, dashIdx), out int start) &&
                    int.TryParse(trimmed.AsSpan(dashIdx + 1), out int end))
                {
                    for (int i = start; i <= end; i++)
                        yield return i;
                }
            }
            else
            {
                if (int.TryParse(trimmed, out int id))
                    yield return id;
            }
        }
    }

    // ── Windows P/Invoke ──

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static partial class WindowsNative
    {
        internal enum LOGICAL_PROCESSOR_RELATIONSHIP
        {
            RelationProcessorCore = 0,
            RelationNumaNode = 1,
            RelationCache = 2,
            RelationProcessorPackage = 3,
            RelationGroup = 4,
        }

        [LibraryImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static partial bool GetLogicalProcessorInformationEx(
            LOGICAL_PROCESSOR_RELATIONSHIP RelationshipType,
            IntPtr Buffer,
            ref uint ReturnedLength);
    }
}
