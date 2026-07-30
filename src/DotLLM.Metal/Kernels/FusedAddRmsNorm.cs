using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Fused residual-add + RMS normalization (FP16) accelerated via Metal GPU.
/// Direct translation of fused_add_rmsnorm_f16.cu.
///
/// Eliminates one FP16 truncation at the residual junction by computing
/// sum-of-squares from the FP32 sum before storing it back as FP16:
///   Pass 1: sum = FP32(residual[i]) + FP32(x[i])
///           residual[i] = FP16(sum)      ← updated in-place
///           sum_sq += sum * sum          ← accumulated in FP32
///   Pass 2: output[i] = FP16(FP32(residual[i]) * rms_inv * FP32(weight[i]))
/// </summary>
public static class FusedAddRmsNorm
{
    /// <summary>
    /// Fused residual-add + RMS normalization over a sequence of tokens.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="residual">
    /// Residual stream, shape <c>[seqLen × n]</c> (float16).
    /// Updated in-place: <c>residual[i] ← FP16(residual[i] + x[i])</c>.
    /// </param>
    /// <param name="x">Layer output to add, shape <c>[seqLen × n]</c> (float16). Read-only.</param>
    /// <param name="weight">RMSNorm scale vector of length <c>n</c> (float16). Read-only.</param>
    /// <param name="output">Normalized output, shape <c>[seqLen × n]</c> (float16). Written by the kernel.</param>
    /// <param name="n">Hidden dimension (inner size).</param>
    /// <param name="seqLen">Number of tokens (outer size).</param>
    /// <param name="eps">Epsilon for numerical stability (e.g. <c>1e-5f</c>).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext    ctx,
        Span<Half>      residual,
        ReadOnlySpan<Half> x,
        ReadOnlySpan<Half> weight,
        Span<Half>      output,
        int             n,
        int             seqLen,
        float           eps = 1e-5f)
    {
        if (weight.Length != n)
            throw new ArgumentException(
                $"weight.Length ({weight.Length}) must equal n ({n}).", nameof(weight));
        if (residual.Length != seqLen * n)
            throw new ArgumentException(
                $"residual.Length ({residual.Length}) must equal seqLen × n ({seqLen * n}).", nameof(residual));
        if (x.Length != seqLen * n)
            throw new ArgumentException(
                $"x.Length ({x.Length}) must equal seqLen × n ({seqLen * n}).", nameof(x));
        if (output.Length != seqLen * n)
            throw new ArgumentException(
                $"output.Length ({output.Length}) must equal seqLen × n ({seqLen * n}).", nameof(output));

        if (seqLen == 0) return;

        // System.Half and ushort share the same 16-bit IEEE 754 layout — zero-copy reinterpret.
        Span<ushort>         residualRaw = MemoryMarshal.Cast<Half, ushort>(residual);
        ReadOnlySpan<ushort> xRaw       = MemoryMarshal.Cast<Half, ushort>(x);
        ReadOnlySpan<ushort> weightRaw  = MemoryMarshal.Cast<Half, ushort>(weight);
        Span<ushort>         outputRaw  = MemoryMarshal.Cast<Half, ushort>(output);

        unsafe
        {
            fixed (ushort* pRes = residualRaw)
            fixed (ushort* pX   = xRaw)
            fixed (ushort* pW   = weightRaw)
            fixed (ushort* pOut = outputRaw)
            {
                int code = MetalNative.FusedAddRmsNormF16(
                    ctx.Handle, pRes, pX, pW, pOut, n, seqLen, eps);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal fused_add_rmsnorm_f16 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers and does not check buffer lengths.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Execute(
        MetalContext ctx,
        nint residual,
        nint x,
        nint weight,
        nint output,
        int n,
        int seqLen,
        float eps = 1e-5f)
    {
        int code = MetalNative.FusedAddRmsNormF16(ctx.Handle, (ushort*)residual, (ushort*)x, (ushort*)weight, (ushort*)output, n, seqLen, eps);

        if (code != 0)
        {
            throw new InvalidOperationException($"Metal fused_add_rmsnorm_f16 failed with code {code}.");
        }
    }
}
