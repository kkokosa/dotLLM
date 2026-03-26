using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// RAII wrapper around a CUDA context. One context per device.
/// Creating a context makes it current on the calling thread.
/// </summary>
public sealed class CudaContext : IDisposable
{
    private nint _ctx;

    /// <summary>Device ordinal this context is bound to.</summary>
    public int DeviceId { get; }

    /// <summary>The native CUDA context handle.</summary>
    public nint Handle => _ctx;

    private CudaContext(nint ctx, int deviceId)
    {
        _ctx = ctx;
        DeviceId = deviceId;
    }

    /// <summary>
    /// Creates a new CUDA context on the specified device.
    /// Initializes the CUDA driver if not already initialized.
    /// </summary>
    /// <param name="deviceId">Device ordinal (0-based).</param>
    public static CudaContext Create(int deviceId)
    {
        CudaLibraryResolver.Register();
        CudaDriverApi.cuInit(0).ThrowOnError();
        CudaDriverApi.cuDeviceGet(out int device, deviceId).ThrowOnError();
        CudaDriverApi.cuCtxCreate_v2(out nint ctx, 0, device).ThrowOnError();
        return new CudaContext(ctx, deviceId);
    }

    /// <summary>Makes this context current on the calling thread.</summary>
    public void MakeCurrent()
    {
        CudaDriverApi.cuCtxSetCurrent(_ctx).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint ctx = Interlocked.Exchange(ref _ctx, 0);
        if (ctx != 0)
            CudaDriverApi.cuCtxDestroy_v2(ctx);
    }
}
