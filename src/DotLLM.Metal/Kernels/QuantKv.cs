using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// KV-cache quantization kernels accelerated via Metal GPU.
/// Used to quantize FP16 KV-cache entries on eviction.
/// Direct translation of quant_kv.cu.
/// </summary>
public static class QuantKv
{
    private const int s_q80BlockSize  = 32;
    private const int s_q80BlockBytes = 34;

    private const int s_q40BlockSize  = 32;
    private const int s_q40BlockBytes = 18;

    /// <summary>
    /// Quantizes FP16 values to Q8_0 format on the GPU.
    /// <c>block_q8_0 = { half d; int8_t qs[32]; }</c>
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="src">FP16 input values. Length must equal <c>totalBlocks × 32</c>.</param>
    /// <param name="dst">Q8_0 output bytes. Length must equal <c>totalBlocks × 34</c>.</param>
    /// <param name="totalBlocks">Number of 32-element blocks to quantize.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void F16ToQ8_0(
        MetalContext       ctx,
        ReadOnlySpan<Half> src,
        Span<byte>         dst,
        int                totalBlocks)
    {
        if (totalBlocks <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalBlocks), "totalBlocks must be > 0.");
        if (src.Length != totalBlocks * s_q80BlockSize)
            throw new ArgumentException(
                $"src.Length ({src.Length}) must equal totalBlocks × {s_q80BlockSize} ({totalBlocks * s_q80BlockSize}).",
                nameof(src));
        if (dst.Length != totalBlocks * s_q80BlockBytes)
            throw new ArgumentException(
                $"dst.Length ({dst.Length}) must equal totalBlocks × {s_q80BlockBytes} ({totalBlocks * s_q80BlockBytes}).",
                nameof(dst));

        ReadOnlySpan<ushort> srcRaw = MemoryMarshal.Cast<Half, ushort>(src);
        unsafe
        {
            fixed (ushort* pSrc = srcRaw)
            fixed (byte*   pDst = dst)
            {
                int code = MetalNative.QuantF16ToQ8_0(ctx.Handle, pSrc, pDst, totalBlocks);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal quant_f16_to_q8_0 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Quantizes FP16 values to Q4_0 format on the GPU.
    /// <c>block_q4_0 = { half d; uint8_t qs[16]; }</c> — values stored with +8 offset.
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="src">FP16 input values. Length must equal <c>totalBlocks × 32</c>.</param>
    /// <param name="dst">Q4_0 output bytes. Length must equal <c>totalBlocks × 18</c>.</param>
    /// <param name="totalBlocks">Number of 32-element blocks to quantize.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void F16ToQ4_0(
        MetalContext       ctx,
        ReadOnlySpan<Half> src,
        Span<byte>         dst,
        int                totalBlocks)
    {
        if (totalBlocks <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalBlocks), "totalBlocks must be > 0.");
        if (src.Length != totalBlocks * s_q40BlockSize)
            throw new ArgumentException(
                $"src.Length ({src.Length}) must equal totalBlocks × {s_q40BlockSize} ({totalBlocks * s_q40BlockSize}).",
                nameof(src));
        if (dst.Length != totalBlocks * s_q40BlockBytes)
            throw new ArgumentException(
                $"dst.Length ({dst.Length}) must equal totalBlocks × {s_q40BlockBytes} ({totalBlocks * s_q40BlockBytes}).",
                nameof(dst));

        ReadOnlySpan<ushort> srcRaw = MemoryMarshal.Cast<Half, ushort>(src);
        unsafe
        {
            fixed (ushort* pSrc = srcRaw)
            fixed (byte*   pDst = dst)
            {
                int code = MetalNative.QuantF16ToQ4_0(ctx.Handle, pSrc, pDst, totalBlocks);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal quant_f16_to_q4_0 failed with code {code}.");
            }
        }
    }
}
