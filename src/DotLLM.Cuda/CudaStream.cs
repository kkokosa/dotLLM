using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// RAII wrapper around a CUDA stream. Provides ordered execution of GPU operations.
/// </summary>
public sealed class CudaStream : IDisposable
{
    private nint _stream;

    /// <summary>The native CUDA stream handle. Pass to kernel launches and memcpy calls.</summary>
    public nint Handle => _stream;

    private CudaStream(nint stream)
    {
        _stream = stream;
    }

    /// <summary>Creates a new CUDA stream on the current context's device.</summary>
    public static CudaStream Create()
    {
        CudaDriverApi.cuStreamCreate(out nint stream, 0).ThrowOnError();
        return new CudaStream(stream);
    }

    /// <summary>Blocks the host thread until all operations on this stream complete.</summary>
    public void Synchronize()
    {
        CudaDriverApi.cuStreamSynchronize(_stream).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint stream = Interlocked.Exchange(ref _stream, 0);
        if (stream != 0)
            CudaDriverApi.cuStreamDestroy_v2(stream);
    }
}
