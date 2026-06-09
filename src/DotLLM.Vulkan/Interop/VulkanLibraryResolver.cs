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

    private static nint ResolveVulkanLibrary(
        string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != "vulkan-1") return 0;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            if (NativeLibrary.TryLoad("vulkan-1.dll", out nint h)) return h;
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            // MoltenVK ships as libvulkan.dylib (plus libMoltenVK.dylib).
            if (NativeLibrary.TryLoad("libvulkan.dylib", out nint h)) return h;
            if (NativeLibrary.TryLoad("libvulkan.1.dylib", out nint h2)) return h2;
        }
        else
        {
            if (NativeLibrary.TryLoad("libvulkan.so.1", out nint h)) return h;
            if (NativeLibrary.TryLoad("libvulkan.so", out nint h2)) return h2;
        }

        return 0; // fall through to default resolution
    }
}
