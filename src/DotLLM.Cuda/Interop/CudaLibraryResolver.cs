using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Resolves "cuda" and "cublas" library names to platform-specific paths.
/// Linux: libcuda.so.1, libcublas.so. Windows: nvcuda.dll, cublas64_*.dll.
/// </summary>
internal static class CudaLibraryResolver
{
    private static int _registered;

    /// <summary>
    /// Registers the resolver. Safe to call multiple times (idempotent).
    /// </summary>
    internal static void Register()
    {
        if (Interlocked.Exchange(ref _registered, 1) != 0) return;

        NativeLibrary.SetDllImportResolver(
            typeof(CudaLibraryResolver).Assembly,
            ResolveCudaLibrary);
    }

    private static nint ResolveCudaLibrary(
        string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "cuda")
        {
            string osLib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "nvcuda.dll"
                : "libcuda.so.1";

            if (NativeLibrary.TryLoad(osLib, out nint handle))
                return handle;
        }

        if (libraryName == "cublas")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                foreach (var ver in new[] { "13", "12", "11" })
                {
                    if (NativeLibrary.TryLoad($"cublas64_{ver}.dll", out nint h))
                        return h;
                }
            }
            else
            {
                if (NativeLibrary.TryLoad("libcublas.so", out nint h))
                    return h;
                // Try versioned names
                foreach (var ver in new[] { "13", "12", "11" })
                {
                    if (NativeLibrary.TryLoad($"libcublas.so.{ver}", out nint h2))
                        return h2;
                }
            }
        }

        return 0; // fall through to default resolution
    }
}
