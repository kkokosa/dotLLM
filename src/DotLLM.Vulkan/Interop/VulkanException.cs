namespace DotLLM.Vulkan.Interop;

/// <summary>
/// Exception thrown when a Vulkan API call returns a non-success <c>VkResult</c>.
/// </summary>
public sealed class VulkanException : Exception
{
    /// <summary>The underlying Vulkan result code (0 = <c>VK_SUCCESS</c>).</summary>
    public int ErrorCode { get; }

    /// <summary>Creates a Vulkan exception with the given error code and message.</summary>
    public VulkanException(int errorCode, string message)
        : base($"Vulkan error {errorCode} ({ResultName(errorCode)}): {message}")
    {
        ErrorCode = errorCode;
    }

    private static string ResultName(int r) => r switch
    {
        0 => "VK_SUCCESS",
        1 => "VK_NOT_READY",
        2 => "VK_TIMEOUT",
        3 => "VK_EVENT_SET",
        4 => "VK_EVENT_RESET",
        5 => "VK_INCOMPLETE",
        -1 => "VK_ERROR_OUT_OF_HOST_MEMORY",
        -2 => "VK_ERROR_OUT_OF_DEVICE_MEMORY",
        -3 => "VK_ERROR_INITIALIZATION_FAILED",
        -4 => "VK_ERROR_DEVICE_LOST",
        -5 => "VK_ERROR_MEMORY_MAP_FAILED",
        -6 => "VK_ERROR_LAYER_NOT_PRESENT",
        -7 => "VK_ERROR_EXTENSION_NOT_PRESENT",
        -8 => "VK_ERROR_FEATURE_NOT_PRESENT",
        -9 => "VK_ERROR_INCOMPATIBLE_DRIVER",
        -10 => "VK_ERROR_TOO_MANY_OBJECTS",
        -11 => "VK_ERROR_FORMAT_NOT_SUPPORTED",
        -12 => "VK_ERROR_FRAGMENTED_POOL",
        -13 => "VK_ERROR_UNKNOWN",
        _ => "VK_ERROR_UNMAPPED"
    };
}

/// <summary>
/// Extension methods for checking Vulkan return codes.
/// </summary>
internal static class VulkanErrorHelper
{
    /// <summary>
    /// Throws <see cref="VulkanException"/> if <paramref name="result"/> is a Vulkan error code
    /// (VK_SUCCESS = 0 and positive values like VK_INCOMPLETE are treated as non-errors).
    /// </summary>
    internal static void ThrowOnError(this int result, string operation)
    {
        if (result >= 0) return;
        throw new VulkanException(result, operation);
    }
}
