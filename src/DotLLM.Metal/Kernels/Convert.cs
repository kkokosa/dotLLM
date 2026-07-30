using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Type conversion kernels accelerated via Metal GPU.
/// Direct translation of convert.cu — element-wise, one thread per value.
/// </summary>
public static class Convert
{
    /// <summary>
    /// Converts <paramref name="n"/> float16 values to float32.
    /// No precision loss — float32 can represent every float16 value exactly.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="src">Source buffer of float16 values (<see cref="Half"/>).</param>
    /// <param name="dst">Destination buffer of float32 values. Must have length >= <paramref name="n"/>.</param>
    /// <param name="n">Number of elements to convert.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void F16ToF32(MetalContext ctx, ReadOnlySpan<Half> src, Span<float> dst, int n)
    {
        if (n <= 0) return;
        if (src.Length < n)
            throw new ArgumentException($"src.Length ({src.Length}) must be >= n ({n}).", nameof(src));
        if (dst.Length < n)
            throw new ArgumentException($"dst.Length ({dst.Length}) must be >= n ({n}).", nameof(dst));

        // System.Half and ushort share the same 16-bit IEEE 754 layout.
        // MemoryMarshal.Cast is a zero-copy reinterpret — no allocation.
        ReadOnlySpan<ushort> srcRaw = MemoryMarshal.Cast<Half, ushort>(src);

        unsafe
        {
            fixed (ushort* pSrc = srcRaw)
            fixed (float*  pDst = dst)
            {
                int code = MetalNative.ConvertF16ToF32(ctx.Handle, pSrc, pDst, n);
                if (code != 0)
                    throw new InvalidOperationException($"Metal convert_f16_to_f32 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers from
    /// <see cref="MetalForwardState"/> directly. No length validation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void F16ToF32(MetalContext ctx, nint src, nint dst, int n)
    {
        if (n <= 0) return;

        int code = MetalNative.ConvertF16ToF32(ctx.Handle, (ushort*)src, (float*)dst, n);
        if (code != 0)
            throw new InvalidOperationException($"Metal convert_f16_to_f32 failed with code {code}.");
    }

    /// <summary>
    /// Converts <paramref name="n"/> float32 values to float16.
    /// Values outside the float16 range (±65504) saturate to ±infinity.
    /// Precision is reduced to ~3 significant decimal digits.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="src">Source buffer of float32 values.</param>
    /// <param name="dst">Destination buffer of float16 values (<see cref="Half"/>). Must have length >= <paramref name="n"/>.</param>
    /// <param name="n">Number of elements to convert.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void F32ToF16(MetalContext ctx, ReadOnlySpan<float> src, Span<Half> dst, int n)
    {
        if (n <= 0) return;
        if (src.Length < n)
            throw new ArgumentException($"src.Length ({src.Length}) must be >= n ({n}).", nameof(src));
        if (dst.Length < n)
            throw new ArgumentException($"dst.Length ({dst.Length}) must be >= n ({n}).", nameof(dst));

        Span<ushort> dstRaw = MemoryMarshal.Cast<Half, ushort>(dst);

        unsafe
        {
            fixed (float*  pSrc = src)
            fixed (ushort* pDst = dstRaw)
            {
                int code = MetalNative.ConvertF32ToF16(ctx.Handle, pSrc, pDst, n);
                if (code != 0)
                    throw new InvalidOperationException($"Metal convert_f32_to_f16 failed with code {code}.");
            }
        }
    }
}
