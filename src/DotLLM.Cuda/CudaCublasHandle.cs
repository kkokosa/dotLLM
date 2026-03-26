using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// RAII wrapper around a cuBLAS handle. Enables Tensor Core math mode on creation.
/// </summary>
public sealed class CudaCublasHandle : IDisposable
{
    private nint _handle;

    /// <summary>The native cuBLAS handle.</summary>
    public nint Handle => _handle;

    private CudaCublasHandle(nint handle)
    {
        _handle = handle;
    }

    /// <summary>
    /// Creates a new cuBLAS handle with Tensor Core math mode enabled.
    /// </summary>
    public static CudaCublasHandle Create()
    {
        CublasApi.cublasCreate_v2(out nint handle).ThrowOnCublasError();
        CublasApi.cublasSetMathMode(handle, CublasApi.CUBLAS_TENSOR_OP_MATH).ThrowOnCublasError();
        return new CudaCublasHandle(handle);
    }

    /// <summary>Binds this cuBLAS handle to the given CUDA stream.</summary>
    public void SetStream(CudaStream stream)
    {
        CublasApi.cublasSetStream_v2(_handle, stream.Handle).ThrowOnCublasError();
    }

    /// <summary>Binds this cuBLAS handle to the given raw CUDA stream handle.</summary>
    public void SetStream(nint streamHandle)
    {
        CublasApi.cublasSetStream_v2(_handle, streamHandle).ThrowOnCublasError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint handle = Interlocked.Exchange(ref _handle, 0);
        if (handle != 0)
            CublasApi.cublasDestroy_v2(handle);
    }
}
