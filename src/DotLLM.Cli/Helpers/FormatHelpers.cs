namespace DotLLM.Cli.Helpers;

/// <summary>
/// Shared formatting utilities for CLI output.
/// </summary>
internal static class FormatHelpers
{
    /// <summary>Formats a byte count as a human-readable size (GB/MB/KB/B).</summary>
    public static string FormatSize(long bytes) => bytes switch
    {
        >= 1L << 30 => $"{bytes / (double)(1L << 30):F1} GB",
        >= 1L << 20 => $"{bytes / (double)(1L << 20):F1} MB",
        >= 1L << 10 => $"{bytes / (double)(1L << 10):F1} KB",
        _ => $"{bytes} B"
    };

    /// <summary>Formats a byte count as MiB with one decimal place.</summary>
    public static string FormatMiB(long bytes) =>
        $"{bytes / (1024.0 * 1024.0):F1} MiB";
}
