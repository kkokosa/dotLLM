using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Minimal P/Invoke declarations against NVIDIA's CUDA Driver API.
/// libcuda.so (Linux) / nvcuda.dll (Windows) — installed with GPU driver.
/// All functions return CUresult (int): 0 = CUDA_SUCCESS, non-zero = error.
/// </summary>
internal static partial class CudaDriverApi
{
    // .NET resolves "cuda" to libcuda.so (Linux) / nvcuda.dll (Windows)
    // via CudaLibraryResolver registered at startup.
    private const string LibName = "cuda";

    // ── Initialization ──────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuInit(uint flags);

    // ── Device ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGet(out int device, int ordinal);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetCount(out int count);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetName(
        [MarshalAs(UnmanagedType.LPArray)] byte[] name, int len, int device);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceTotalMem_v2(out nuint bytes, int device);

    [LibraryImport(LibName)]
    internal static partial int cuMemGetInfo_v2(out nuint free, out nuint total);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetAttribute(
        out int value, int attribute, int device);

    // ── Context ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuCtxCreate_v2(out nint ctx, uint flags, int device);

    [LibraryImport(LibName)]
    internal static partial int cuCtxDestroy_v2(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxSetCurrent(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxGetCurrent(out nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxGetDevice(out int device);

    // ── Module (PTX loading) ────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuModuleLoadData(out nint module, nint ptxImage);

    [LibraryImport(LibName)]
    internal static partial int cuModuleLoadDataEx(
        out nint module, nint ptxImage, uint numOptions,
        nint options, nint optionValues);

    [LibraryImport(LibName)]
    internal static partial int cuModuleGetFunction(
        out nint function, nint module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    [LibraryImport(LibName)]
    internal static partial int cuModuleUnload(nint module);

    // ── Kernel launch ───────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuLaunchKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint stream,
        nint kernelParams, nint extra);

    // ── Memory ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuMemAlloc_v2(out nint devicePtr, nuint bytesize);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    internal static partial int cuMemFree_v2(nint devicePtr);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoD_v2(
        nint dstDevice, nint srcHost, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoH_v2(
        nint dstHost, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoD_v2(
        nint dstDevice, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoDAsync_v2(
        nint dstDevice, nint srcHost, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoHAsync_v2(
        nint dstHost, nint srcDevice, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoDAsync_v2(
        nint dstDevice, nint srcDevice, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemsetD8_v2(nint dstDevice, byte value, nuint n);

    // ── Streams ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuStreamCreate(out nint stream, uint flags);

    [LibraryImport(LibName)]
    internal static partial int cuStreamDestroy_v2(nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuStreamSynchronize(nint stream);

    // ── Error ───────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuGetErrorName(int error, out nint str);

    [LibraryImport(LibName)]
    internal static partial int cuGetErrorString(int error, out nint str);

    // ── Device attribute constants ──────────────────────────────────

    internal const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    internal const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
    internal const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    internal const int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
    internal const int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8;
    internal const int CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10;
}
