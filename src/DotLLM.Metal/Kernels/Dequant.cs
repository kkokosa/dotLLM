using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Dequantization kernels accelerated via Metal GPU.
/// Each kernel converts a packed quantized buffer to FP16 (Half).
/// </summary>
public static class Dequant
{
    // Block / superblock geometry — mirrors dequant.metal defines.
    private const int s_q80BlockSize  = 32;
    private const int s_q80BlockBytes = 34;

    private const int s_q40BlockSize  = 32;
    private const int s_q40BlockBytes = 18;

    private const int s_q50BlockSize  = 32;
    private const int s_q50BlockBytes = 22;

    private const int s_q4KSuperBlockSize  = 256;
    private const int s_q4KBlockBytes      = 144;

    private const int s_q5KSuperBlockSize  = 256;
    private const int s_q5KBlockBytes      = 176;

    private const int s_q6KSuperBlockSize  = 256;
    private const int s_q6KBlockBytes      = 210;

    // ── Q8_0 ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dequantize Q8_0 → FP16 on the GPU.
    /// Each block is 34 bytes: 2-byte half scale followed by 32 int8 weights.
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="src">Raw Q8_0 bytes. Length must equal <c>totalBlocks × 34</c>.</param>
    /// <param name="dst">Output FP16 values. Length must equal <c>totalBlocks × 32</c>.</param>
    /// <param name="totalBlocks">Number of Q8_0 blocks.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q8_0ToF16(
        MetalContext       ctx,
        ReadOnlySpan<byte> src,
        Span<Half>         dst,
        int                totalBlocks)
    {
        ValidateSimple(src.Length, dst.Length, totalBlocks, s_q80BlockBytes, s_q80BlockSize, nameof(Q8_0ToF16));
        Span<ushort> dstRaw = MemoryMarshal.Cast<Half, ushort>(dst);
        unsafe
        {
            fixed (byte*   pSrc = src)
            fixed (ushort* pDst = dstRaw)
            {
                int code = MetalNative.DequantQ8_0F16(ctx.Handle, pSrc, pDst, totalBlocks);
                if (code != 0)
                    throw new InvalidOperationException($"Metal dequant_q8_0_f16 failed with code {code}.");
            }
        }
    }

    // ── Q4_0 ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dequantize Q4_0 → FP16 on the GPU.
    /// Each block is 18 bytes: 2-byte half scale followed by 16 packed nibble bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q4_0ToF16(
        MetalContext       ctx,
        ReadOnlySpan<byte> src,
        Span<Half>         dst,
        int                totalBlocks)
    {
        ValidateSimple(src.Length, dst.Length, totalBlocks, s_q40BlockBytes, s_q40BlockSize, nameof(Q4_0ToF16));
        Span<ushort> dstRaw = MemoryMarshal.Cast<Half, ushort>(dst);
        unsafe
        {
            fixed (byte*   pSrc = src)
            fixed (ushort* pDst = dstRaw)
            {
                int code = MetalNative.DequantQ4_0F16(ctx.Handle, pSrc, pDst, totalBlocks);
                if (code != 0)
                    throw new InvalidOperationException($"Metal dequant_q4_0_f16 failed with code {code}.");
            }
        }
    }

    // ── Q5_0 ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dequantize Q5_0 → FP16 on the GPU.
    /// Each block is 22 bytes: 2-byte half scale, 4-byte high-bit mask, 16 packed nibble bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q5_0ToF16(
        MetalContext       ctx,
        ReadOnlySpan<byte> src,
        Span<Half>         dst,
        int                totalBlocks)
    {
        ValidateSimple(src.Length, dst.Length, totalBlocks, s_q50BlockBytes, s_q50BlockSize, nameof(Q5_0ToF16));
        Span<ushort> dstRaw = MemoryMarshal.Cast<Half, ushort>(dst);
        unsafe
        {
            fixed (byte*   pSrc = src)
            fixed (ushort* pDst = dstRaw)
            {
                int code = MetalNative.DequantQ5_0F16(ctx.Handle, pSrc, pDst, totalBlocks);
                if (code != 0)
                    throw new InvalidOperationException($"Metal dequant_q5_0_f16 failed with code {code}.");
            }
        }
    }

    // ── Q4_K ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dequantize Q4_K → FP16 on the GPU.
    /// Each superblock is 144 bytes and produces 256 FP16 values.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q4_KToF16(
        MetalContext       ctx,
        ReadOnlySpan<byte> src,
        Span<Half>         dst,
        int                totalSuperblocks)
    {
        ValidateSimple(src.Length, dst.Length, totalSuperblocks, s_q4KBlockBytes, s_q4KSuperBlockSize, nameof(Q4_KToF16));
        Span<ushort> dstRaw = MemoryMarshal.Cast<Half, ushort>(dst);
        unsafe
        {
            fixed (byte*   pSrc = src)
            fixed (ushort* pDst = dstRaw)
            {
                int code = MetalNative.DequantQ4_KF16(ctx.Handle, pSrc, pDst, totalSuperblocks);
                if (code != 0)
                    throw new InvalidOperationException($"Metal dequant_q4_k_f16 failed with code {code}.");
            }
        }
    }

    // ── Q5_K ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dequantize Q5_K → FP16 on the GPU.
    /// Each superblock is 176 bytes and produces 256 FP16 values.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q5_KToF16(
        MetalContext       ctx,
        ReadOnlySpan<byte> src,
        Span<Half>         dst,
        int                totalSuperblocks)
    {
        ValidateSimple(src.Length, dst.Length, totalSuperblocks, s_q5KBlockBytes, s_q5KSuperBlockSize, nameof(Q5_KToF16));
        Span<ushort> dstRaw = MemoryMarshal.Cast<Half, ushort>(dst);
        unsafe
        {
            fixed (byte*   pSrc = src)
            fixed (ushort* pDst = dstRaw)
            {
                int code = MetalNative.DequantQ5_KF16(ctx.Handle, pSrc, pDst, totalSuperblocks);
                if (code != 0)
                    throw new InvalidOperationException($"Metal dequant_q5_k_f16 failed with code {code}.");
            }
        }
    }

    // ── Q6_K ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dequantize Q6_K → FP16 on the GPU.
    /// Each superblock is 210 bytes and produces 256 FP16 values.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q6_KToF16(
        MetalContext       ctx,
        ReadOnlySpan<byte> src,
        Span<Half>         dst,
        int                totalSuperblocks)
    {
        ValidateSimple(src.Length, dst.Length, totalSuperblocks, s_q6KBlockBytes, s_q6KSuperBlockSize, nameof(Q6_KToF16));
        Span<ushort> dstRaw = MemoryMarshal.Cast<Half, ushort>(dst);
        unsafe
        {
            fixed (byte*   pSrc = src)
            fixed (ushort* pDst = dstRaw)
            {
                int code = MetalNative.DequantQ6_KF16(ctx.Handle, pSrc, pDst, totalSuperblocks);
                if (code != 0)
                    throw new InvalidOperationException($"Metal dequant_q6_k_f16 failed with code {code}.");
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void ValidateSimple(
        int srcLen, int dstLen,
        int count, int blockBytes, int blockElems,
        string caller)
    {
        if (count <= 0)
            throw new ArgumentOutOfRangeException(nameof(count),
                $"[{caller}] count must be > 0.");
        if (srcLen != count * blockBytes)
            throw new ArgumentException(
                $"[{caller}] src.Length ({srcLen}) must equal count × {blockBytes} ({count * blockBytes}).");
        if (dstLen != count * blockElems)
            throw new ArgumentException(
                $"[{caller}] dst.Length ({dstLen}) must equal count × {blockElems} ({count * blockElems}).");
    }
}
