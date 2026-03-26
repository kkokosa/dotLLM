namespace DotLLM.Cuda.Interop;

/// <summary>
/// Exception thrown when a CUDA Driver API or cuBLAS call fails.
/// </summary>
public sealed class CudaException : Exception
{
    /// <summary>CUDA error code (CUresult or cublasStatus_t).</summary>
    public int ErrorCode { get; }

    /// <summary>Creates a CUDA exception with the given error code and message.</summary>
    public CudaException(int errorCode, string message)
        : base($"CUDA error {errorCode}: {message}")
    {
        ErrorCode = errorCode;
    }
}
