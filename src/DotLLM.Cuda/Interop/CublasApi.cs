using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Minimal cuBLAS P/Invoke. libcublas.so / cublas64_*.dll — from CUDA Toolkit.
/// </summary>
internal static partial class CublasApi
{
    // .NET resolves "cublas" via CudaLibraryResolver.
    private const string LibName = "cublas";

    [LibraryImport(LibName)]
    internal static partial int cublasCreate_v2(out nint handle);

    [LibraryImport(LibName)]
    internal static partial int cublasDestroy_v2(nint handle);

    [LibraryImport(LibName)]
    internal static partial int cublasSetStream_v2(nint handle, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cublasSetMathMode(nint handle, int mode);

    /// <summary>
    /// FP16 GEMM — C = alpha * op(A) * op(B) + beta * C, all FP16.
    /// Tensor Cores used automatically when dims are multiples of 8.
    /// Row-major trick: compute C^T = B^T @ A^T via swapped args.
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cublasHgemm(
        nint handle,
        int transa, int transb,     // cublasOperation_t: 0=N, 1=T, 2=C
        int m, int n, int k,
        in ushort alpha,            // __half passed as ushort
        nint A, int lda,
        nint B, int ldb,
        in ushort beta,
        nint C, int ldc);

    /// <summary>
    /// Mixed-precision GEMM — FP16 input, FP32 accumulate.
    /// </summary>
    [LibraryImport(LibName)]
    internal static partial int cublasGemmEx(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        nint alpha,
        nint A, int Atype, int lda,
        nint B, int Btype, int ldb,
        nint beta,
        nint C, int Ctype, int ldc,
        int computeType, int algo);

    // ── cuBLAS constants ─────────────────────────────────────────────

    /// <summary>CUBLAS_OP_N — no transpose.</summary>
    internal const int CUBLAS_OP_N = 0;

    /// <summary>CUBLAS_OP_T — transpose.</summary>
    internal const int CUBLAS_OP_T = 1;

    /// <summary>CUBLAS_TENSOR_OP_MATH — enable Tensor Core usage.</summary>
    internal const int CUBLAS_TENSOR_OP_MATH = 1;

    /// <summary>CUDA_R_16F — FP16 data type.</summary>
    internal const int CUDA_R_16F = 2;

    /// <summary>CUDA_R_32F — FP32 data type.</summary>
    internal const int CUDA_R_32F = 0;

    /// <summary>CUBLAS_COMPUTE_16F — FP16 compute.</summary>
    internal const int CUBLAS_COMPUTE_16F = 64;

    /// <summary>CUBLAS_COMPUTE_32F — FP32 compute.</summary>
    internal const int CUBLAS_COMPUTE_32F = 68;

    /// <summary>CUBLAS_GEMM_DEFAULT — default algorithm selection.</summary>
    internal const int CUBLAS_GEMM_DEFAULT = -1;
}
