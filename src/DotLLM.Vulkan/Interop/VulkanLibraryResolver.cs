using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.Vulkan.Interop;

/// <summary>
/// Resolves the "vulkan-1" library name to platform-specific Vulkan loader binaries.
/// Windows: vulkan-1.dll. Linux: libvulkan.so.1. macOS: libvulkan.dylib (via MoltenVK).
/// </summary>
internal static class VulkanLibraryResolver
{
    private static int _registered;

    /// <summary>
    /// Registers the resolver. Safe to call multiple times (idempotent).
    /// </summary>
    internal static void Register()
    {
        if (Interlocked.Exchange(ref _registered, 1) != 0) return;

        NativeLibrary.SetDllImportResolver(
            typeof(VulkanLibraryResolver).Assembly,
            ResolveVulkanLibrary);
    }

    /// <summary>
    /// Platform-specific loader filenames, in probe order. Single source of truth so an
    /// availability probe can never disagree with what actually gets loaded at runtime.
    /// </summary>
    internal static string[] LoaderCandidates =>
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? ["vulkan-1.dll"]
        // MoltenVK ships as libvulkan.dylib (plus libMoltenVK.dylib).
        : RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? ["libvulkan.dylib", "libvulkan.1.dylib"]
        : ["libvulkan.so.1", "libvulkan.so"];

    /// <summary>
    /// Attempts to load the platform Vulkan loader, trying every known filename.
    /// Caller owns <paramref name="handle"/> and must <see cref="NativeLibrary.Free"/> it.
    /// </summary>
    internal static bool TryLoadLoader(out nint handle)
    {
        foreach (string candidate in LoaderCandidates)
        {
            if (NativeLibrary.TryLoad(candidate, out handle))
                return true;
        }
        handle = 0;
        return false;
    }

    private static nint ResolveVulkanLibrary(
        string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != "vulkan-1") return 0;
        return TryLoadLoader(out nint handle) ? handle : 0; // 0 falls through to default resolution
    }
}
