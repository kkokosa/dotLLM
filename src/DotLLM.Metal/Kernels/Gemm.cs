using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// General matrix-matrix multiplication accelerated via Metal Performance Shaders.
/// <c>C = alpha · op(A) · op(B) + beta · C</c> where <c>op(X) = Xᵀ</c> if transposed.
/// </summary>
/// <remarks>
/// <para>
/// Storage layouts (row-major, when not transposed):
/// <list type="bullet">
///   <item><description><c>A</c> : <c>[m, k]</c></description></item>
///   <item><description><c>B</c> : <c>[k, n]</c></description></item>
///   <item><description><c>C</c> : <c>[m, n]</c></description></item>
/// </list>
/// </para>
/// <para>
/// Standard LLM projection <c>Y = X · Wᵀ</c> with weights stored as <c>[outputDim, inputDim]</c>:
/// pass <c>m = seqLen</c>, <c>n = outputDim</c>, <c>k = inputDim</c>,
/// <c>transposeA = false</c>, <c>transposeB = true</c>, <c>alpha = 1</c>, <c>beta = 0</c>.
/// </para>
/// </remarks>
public static class Gemm
{
    /// <summary>FP16 GEMM. Buffers reinterpret <see cref="Half"/> as <c>ushort</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteF16(
        MetalContext       ctx,
        ReadOnlySpan<Half> a,
        ReadOnlySpan<Half> b,
        Span<Half>         c,
        int                m,
        int                n,
        int                k,
        bool               transposeA = false,
        bool               transposeB = false,
        float              alpha      = 1.0f,
        float              beta       = 0.0f)
    {
        Validate(a.Length, b.Length, c.Length, m, n, k, transposeA, transposeB);

        var aU = MemoryMarshal.Cast<Half, ushort>(a);
        var bU = MemoryMarshal.Cast<Half, ushort>(b);
        var cU = MemoryMarshal.Cast<Half, ushort>(c);

        unsafe
        {
            fixed (ushort* pA = aU)
            fixed (ushort* pB = bU)
            fixed (ushort* pC = cU)
            {
                int code = MetalNative.GemmF16(
                    ctx.Handle, pA, pB, pC, m, n, k,
                    transposeA ? 1 : 0, transposeB ? 1 : 0, alpha, beta);
                if (code != 0)
                    throw new InvalidOperationException($"Metal gemm_f16 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers and does not check buffer lengths.
    /// Backed by <c>MPSMatrixMultiplication</c> for every mode.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void ExecuteF16(
        MetalContext ctx,
        nint a,
        nint b,
        nint c,
        int m,
        int n,
        int k,
        bool transposeA = false,
        bool transposeB = false,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        int code = MetalNative.GemmF16(
            ctx.Handle, (ushort*)a, (ushort*)b, (ushort*)c, m, n, k,
            transposeA ? 1 : 0, transposeB ? 1 : 0, alpha, beta);
        if (code != 0)
        {
            throw new InvalidOperationException($"Metal gemm_f16 failed with code {code}.");
        }
    }

    /// <summary>FP32 GEMM.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteF32(
        MetalContext        ctx,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float>         c,
        int                 m,
        int                 n,
        int                 k,
        bool                transposeA = false,
        bool                transposeB = false,
        float               alpha      = 1.0f,
        float               beta       = 0.0f)
    {
        Validate(a.Length, b.Length, c.Length, m, n, k, transposeA, transposeB);

        unsafe
        {
            fixed (float* pA = a)
            fixed (float* pB = b)
            fixed (float* pC = c)
            {
                int code = MetalNative.GemmF32(
                    ctx.Handle, pA, pB, pC, m, n, k,
                    transposeA ? 1 : 0, transposeB ? 1 : 0, alpha, beta);
                if (code != 0)
                    throw new InvalidOperationException($"Metal gemm_f32 failed with code {code}.");
            }
        }
    }

    private static void Validate(
        int aLen, int bLen, int cLen,
        int m, int n, int k,
        bool transposeA, bool transposeB)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));

        long expectedA = (long)m * k;
        long expectedB = (long)k * n;
        long expectedC = (long)m * n;

        if (aLen != expectedA)
            throw new ArgumentException(
                $"a.Length ({aLen}) must equal {(transposeA ? "k×m" : "m×k")} = {expectedA}.", nameof(aLen));
        if (bLen != expectedB)
            throw new ArgumentException(
                $"b.Length ({bLen}) must equal {(transposeB ? "n×k" : "k×n")} = {expectedB}.", nameof(bLen));
        if (cLen != expectedC)
            throw new ArgumentException(
                $"c.Length ({cLen}) must equal m×n = {expectedC}.", nameof(cLen));
    }
}
