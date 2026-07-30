using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Element-wise FP16 vector addition accelerated via Metal GPU.
/// Port of <c>add.cu::add_f16</c>. Uses vectorized half2 operations internally.
/// </summary>
public static class AddF16
{
    /// <summary>
    /// Adds two FP16 vectors element-wise: <c>result[i] = a[i] + b[i]</c>.
    /// Output may alias either input (in-place safe).
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="a">First input vector (FP16).</param>
    /// <param name="b">Second input vector (FP16).</param>
    /// <param name="result">Output span (FP16). Must be at least as long as <paramref name="a"/>.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when input lengths differ or <paramref name="result"/> is too short.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, ReadOnlySpan<Half> a, ReadOnlySpan<Half> b, Span<Half> result)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Input spans must have the same length.");

        if (result.Length < a.Length)
            throw new ArgumentException("Result span is too small.");

        if (a.Length == 0)
            return;

        var aU  = MemoryMarshal.Cast<Half, ushort>(a);
        var bU  = MemoryMarshal.Cast<Half, ushort>(b);
        var rU  = MemoryMarshal.Cast<Half, ushort>(result);

        unsafe
        {
            fixed (ushort* pA = aU)
            fixed (ushort* pB = bU)
            fixed (ushort* pR = rU)
            {
                int code = MetalNative.AddF16(ctx.Handle, pA, pB, pR, (uint)a.Length);
                if (code != 0)
                    throw new InvalidOperationException($"Metal add_f16 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers and does not check buffer lengths.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Execute(MetalContext ctx, nint a, nint b, nint result, uint seqLen)
    {
        int code = MetalNative.AddF16(ctx.Handle, (ushort*)a, (ushort*)b, (ushort*)result, seqLen);
        if (code != 0)
            throw new InvalidOperationException($"Metal add_f16 failed with code {code}.");

    }
}
