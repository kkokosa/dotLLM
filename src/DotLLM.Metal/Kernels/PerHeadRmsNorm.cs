using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Per-head RMS Normalization accelerated via Metal GPU.
/// Applies RMSNorm independently to each attention head within each token.
/// Used by models with QK-norm (Gemma 2, Cohere Command R).
/// Direct translation of per_head_rmsnorm_f32.cu.
/// </summary>
public static class PerHeadRmsNorm
{
    /// <summary>
    /// Applies RMS normalization per (token, head) in-place:
    /// for each head h and token t, <c>qk[t, h, i] = qk[t, h, i] / rms(qk[t, h]) * weight[i]</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="qk">
    /// Query or key tensor of shape <c>[seqLen, numHeads, headDim]</c>, modified in-place.
    /// Must have length equal to <paramref name="seqLen"/> × <paramref name="numHeads"/> × <paramref name="headDim"/>.
    /// </param>
    /// <param name="weight">Scale vector of length <paramref name="headDim"/> (shared across all heads and tokens).</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="headDim">Dimension per head.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="eps">Epsilon for numerical stability (e.g. <c>1e-5f</c>).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext        ctx,
        Span<float>         qk,
        ReadOnlySpan<float> weight,
        int                 numHeads,
        int                 headDim,
        int                 seqLen,
        float               eps = 1e-5f)
    {
        if (weight.Length != headDim)
            throw new ArgumentException(
                $"weight.Length ({weight.Length}) must equal headDim ({headDim}).", nameof(weight));
        if (qk.Length != seqLen * numHeads * headDim)
            throw new ArgumentException(
                $"qk.Length ({qk.Length}) must equal seqLen × numHeads × headDim ({seqLen * numHeads * headDim}).", nameof(qk));

        if (seqLen == 0) return;

        unsafe
        {
            fixed (float* pQK     = qk)
            fixed (float* pWeight = weight)
            {
                int code = MetalNative.PerHeadRmsNormF32(
                    ctx.Handle, pQK, pWeight, numHeads, headDim, seqLen, eps);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal per_head_rmsnorm_f32 failed with code {code}.");
            }
        }
    }
}

/// <summary>
/// Per-head RMS Normalization (FP16) accelerated via Metal GPU.
/// Applies RMSNorm independently to each attention head within each token.
/// FP16 I/O with FP32 accumulation. Port of <c>per_head_rmsnorm_f16.cu</c>.
/// </summary>
public static class PerHeadRmsNormF16
{
    /// <summary>
    /// Applies FP16 RMS normalization per (token, head) in-place:
    /// for each head h and token t, <c>qk[t, h, i] = FP16(FP32(qk[t, h, i]) / rms(qk[t, h]) * FP32(weight[i]))</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="qk">
    /// Query or key tensor of shape <c>[seqLen, numHeads, headDim]</c> (FP16), modified in-place.
    /// Must have length equal to <paramref name="seqLen"/> × <paramref name="numHeads"/> × <paramref name="headDim"/>.
    /// </param>
    /// <param name="weight">Scale vector of length <paramref name="headDim"/> (FP16, γ parameter).</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="headDim">Dimension per head.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="eps">Epsilon for numerical stability (e.g. <c>1e-5f</c>).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext       ctx,
        Span<Half>         qk,
        ReadOnlySpan<Half> weight,
        int                numHeads,
        int                headDim,
        int                seqLen,
        float              eps = 1e-5f)
    {
        if (weight.Length != headDim)
            throw new ArgumentException(
                $"weight.Length ({weight.Length}) must equal headDim ({headDim}).", nameof(weight));
        if (qk.Length != seqLen * numHeads * headDim)
            throw new ArgumentException(
                $"qk.Length ({qk.Length}) must equal seqLen × numHeads × headDim ({seqLen * numHeads * headDim}).", nameof(qk));

        if (seqLen == 0) return;

        var qkU     = MemoryMarshal.Cast<Half, ushort>(qk);
        var weightU = MemoryMarshal.Cast<Half, ushort>(weight);

        unsafe
        {
            fixed (ushort* pQK     = qkU)
            fixed (ushort* pWeight = weightU)
            {
                int code = MetalNative.PerHeadRmsNormF16(
                    ctx.Handle, pQK, pWeight, numHeads, headDim, seqLen, eps);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal per_head_rmsnorm_f16 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers and does not check buffer lengths.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Execute(
        MetalContext       ctx,
        nint               qk,
        nint               weight,
        int                numHeads,
        int                headDim,
        int                seqLen,
        float              eps = 1e-5f)
    {
        if (seqLen == 0) return;

        int code = MetalNative.PerHeadRmsNormF16(
            ctx.Handle, (ushort*)qk, (ushort*)weight, numHeads, headDim, seqLen, eps);

        if (code != 0)
        {
            throw new InvalidOperationException($"Metal per_head_rmsnorm_f16 failed with code {code}.");
        }
    }
}
