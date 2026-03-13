using System.Runtime.InteropServices;

namespace DotLLM.Cpu.Threading;

/// <summary>
/// Pins the current thread to a specific logical processor.
/// Supports Windows (<c>SetThreadAffinityMask</c>) and Linux (<c>sched_setaffinity</c>).
/// Returns false on failure or unsupported platforms — never throws.
/// </summary>
internal static partial class CpuAffinity
{
    /// <summary>
    /// Pins the current thread to the specified logical processor.
    /// </summary>
    /// <param name="logicalProcessorId">OS logical processor index (0-based).</param>
    /// <returns><c>true</c> if pinning succeeded, <c>false</c> on failure or unsupported platform.</returns>
    public static bool PinCurrentThread(int logicalProcessorId)
    {
        if (logicalProcessorId < 0)
            return false;

        if (OperatingSystem.IsWindows())
            return PinWindows(logicalProcessorId);
        if (OperatingSystem.IsLinux())
            return PinLinux(logicalProcessorId);

        // macOS: thread_policy_set with THREAD_AFFINITY_POLICY is advisory and
        // doesn't guarantee pinning. Return false to indicate unsupported.
        return false;
    }

    // ── Windows ──

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static bool PinWindows(int logicalProcessorId)
    {
        if (logicalProcessorId >= 64)
            return false; // Would need processor groups — not supported

        try
        {
            nint mask = (nint)(1UL << logicalProcessorId);
            nint result = WindowsNative.SetThreadAffinityMask(WindowsNative.GetCurrentThread(), mask);
            return result != 0;
        }
        catch
        {
            return false;
        }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private static partial class WindowsNative
    {
        [LibraryImport("kernel32.dll")]
        internal static partial nint GetCurrentThread();

        [LibraryImport("kernel32.dll")]
        internal static partial nint SetThreadAffinityMask(nint hThread, nint dwThreadAffinityMask);
    }

    // ── Linux ──

    [System.Runtime.Versioning.SupportedOSPlatform("linux")]
    private static bool PinLinux(int logicalProcessorId)
    {
        try
        {
            // cpu_set_t on x86_64 is 128 bytes (1024 bits), supporting up to 1024 CPUs.
            // We need ceil(logicalProcessorId + 1, 64) ulongs.
            int ulongCount = (logicalProcessorId / 64) + 1;
            int setSize = ulongCount * 8; // bytes

            Span<ulong> mask = stackalloc ulong[ulongCount];
            mask.Clear();
            mask[logicalProcessorId / 64] = 1UL << (logicalProcessorId % 64);

            int result;
            unsafe
            {
                fixed (ulong* pMask = mask)
                {
                    result = LinuxNative.sched_setaffinity(0, (nuint)setSize, pMask);
                }
            }
            return result == 0;
        }
        catch
        {
            return false;
        }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("linux")]
    private static unsafe partial class LinuxNative
    {
        [LibraryImport("libc")]
        internal static partial int sched_setaffinity(int pid, nuint cpusetsize, ulong* mask);
    }
}
