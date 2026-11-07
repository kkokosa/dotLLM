using System.Globalization;

namespace DotLLM.Ascend;

/// <summary>Device selection and option validation for the experimental Ascend backend.</summary>
public static class AscendDevice
{
    /// <summary>Parses "ascend" or "ascend:N"; returns false for other backends.</summary>
    public static bool TryParse(string device, out int ordinal)
    {
        ordinal = 0;
        if (!device.StartsWith("ascend", StringComparison.OrdinalIgnoreCase)) return false;
        if (device.Equals("ascend", StringComparison.OrdinalIgnoreCase)) return true;
        if (!device.StartsWith("ascend:", StringComparison.OrdinalIgnoreCase) ||
            !int.TryParse(device.AsSpan(7), NumberStyles.None, CultureInfo.InvariantCulture, out ordinal) ||
            ordinal < 0)
            throw new ArgumentException("Ascend device must be 'ascend' or 'ascend:N' with a nonnegative ordinal.");
        return true;
    }

    /// <summary>Rejects unsupported options before model resolution or allocation.</summary>
    public static void ValidateOptions(string device, int? gpuLayers, string keyType, string valueType,
        bool paged = false, string? speculativeModel = null, int cacheWindow = 0)
    {
        if (!TryParse(device, out _)) return;
        if (gpuLayers is not null || paged || speculativeModel is not null || cacheWindow != 0 ||
            !keyType.Equals("f32", StringComparison.OrdinalIgnoreCase) ||
            !valueType.Equals("f32", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException(
                "Ascend uses a single NPU and contiguous FP16 KV. GPU layer offload, paged/quantized KV, " +
                "cache windows and speculative decoding are unsupported. Leave cache-type flags at their defaults.");
    }
}
