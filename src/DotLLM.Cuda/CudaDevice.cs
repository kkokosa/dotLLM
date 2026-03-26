using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Queries NVIDIA GPU device properties via the CUDA Driver API.
/// </summary>
public sealed class CudaDevice
{
    /// <summary>Device ordinal (0-based).</summary>
    public int Ordinal { get; }

    /// <summary>Device name (e.g., "NVIDIA GeForce RTX 4090").</summary>
    public string Name { get; }

    /// <summary>Total device memory in bytes.</summary>
    public long TotalMemoryBytes { get; }

    /// <summary>Compute capability major version.</summary>
    public int ComputeCapabilityMajor { get; }

    /// <summary>Compute capability minor version.</summary>
    public int ComputeCapabilityMinor { get; }

    /// <summary>Number of streaming multiprocessors.</summary>
    public int MultiprocessorCount { get; }

    private CudaDevice(int ordinal, string name, long totalMem, int ccMajor, int ccMinor, int smCount)
    {
        Ordinal = ordinal;
        Name = name;
        TotalMemoryBytes = totalMem;
        ComputeCapabilityMajor = ccMajor;
        ComputeCapabilityMinor = ccMinor;
        MultiprocessorCount = smCount;
    }

    /// <summary>Total device memory formatted as a human-readable string (e.g., "24.0 GB").</summary>
    public string TotalMemoryFormatted => $"{TotalMemoryBytes / (1024.0 * 1024 * 1024):F1} GB";

    /// <summary>Compute capability as "major.minor" string (e.g., "8.9").</summary>
    public string ComputeCapability => $"{ComputeCapabilityMajor}.{ComputeCapabilityMinor}";

    /// <summary>
    /// Checks whether CUDA is available on this system (driver installed, at least one GPU).
    /// Does not throw; returns false if CUDA is unavailable.
    /// </summary>
    public static bool IsAvailable()
    {
        try
        {
            // Probe for the CUDA driver library before any P/Invoke.
            // This must happen BEFORE any reference to CudaDriverApi, because
            // the JIT resolves [LibraryImport] P/Invoke stubs when compiling
            // a method — triggering DllNotFoundException on systems without CUDA.
            string cudaLib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "nvcuda.dll" : "libcuda.so.1";
            if (!NativeLibrary.TryLoad(cudaLib, out nint handle))
                return false;
            NativeLibrary.Free(handle);

            return ProbeGpuCount();
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Isolated in <see cref="MethodImplOptions.NoInlining"/> so the JIT only resolves
    /// CudaDriverApi P/Invoke stubs when this method is actually called — after the
    /// NativeLibrary.TryLoad probe confirms libcuda is present.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool ProbeGpuCount()
    {
        CudaLibraryResolver.Register();
        CudaDriverApi.cuInit(0).ThrowOnError();
        CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
        return count > 0;
    }

    /// <summary>Returns the number of CUDA-capable GPUs.</summary>
    public static int GetDeviceCount()
    {
        CudaLibraryResolver.Register();
        CudaDriverApi.cuInit(0).ThrowOnError();
        CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
        return count;
    }

    /// <summary>Queries device properties for the given ordinal.</summary>
    public static CudaDevice GetDevice(int ordinal)
    {
        CudaLibraryResolver.Register();
        CudaDriverApi.cuInit(0).ThrowOnError();

        CudaDriverApi.cuDeviceGet(out int device, ordinal).ThrowOnError();

        // Name
        byte[] nameBuffer = new byte[256];
        CudaDriverApi.cuDeviceGetName(nameBuffer, nameBuffer.Length, device).ThrowOnError();
        int nullIdx = Array.IndexOf(nameBuffer, (byte)0);
        string name = Encoding.ASCII.GetString(nameBuffer, 0, nullIdx >= 0 ? nullIdx : nameBuffer.Length).Trim();

        // Total memory
        CudaDriverApi.cuDeviceTotalMem_v2(out nuint totalMem, device).ThrowOnError();

        // Compute capability
        CudaDriverApi.cuDeviceGetAttribute(out int ccMajor,
            CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device).ThrowOnError();
        CudaDriverApi.cuDeviceGetAttribute(out int ccMinor,
            CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device).ThrowOnError();

        // SM count
        CudaDriverApi.cuDeviceGetAttribute(out int smCount,
            CudaDriverApi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device).ThrowOnError();

        return new CudaDevice(ordinal, name, (long)totalMem, ccMajor, ccMinor, smCount);
    }

    /// <inheritdoc/>
    public override string ToString() =>
        $"GPU {Ordinal}: {Name} ({TotalMemoryFormatted}, sm_{ComputeCapabilityMajor}{ComputeCapabilityMinor}, {MultiprocessorCount} SMs)";
}
