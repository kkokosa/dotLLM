using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// RMS Normalization accelerated via Metal GPU.
/// Direct translation of rmsnorm_f32.cu — one threadgroup per token,
/// threads collaborate via simd_sum() and threadgroup memory.
/// </summary>
public static class RmsNorm
{
    /// <summary>
    /// Applies RMS normalization row-wise:
    /// <c>output[t, i] = input[t, i] / rms(input[t]) * weight[i]</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="input">Input tensor of shape <c>[seqLen × n]</c> (read-only).</param>
    /// <param name="weight">Scale vector of length <c>n</c> (γ parameter).</param>
    /// <param name="output">Output tensor of shape <c>[seqLen × n]</c>, written by the kernel.</param>
    /// <param name="n">Hidden dimension (inner size, e.g. 4096 for Llama-7B).</param>
    /// <param name="seqLen">Number of tokens (outer size).</param>
    /// <param name="eps">Epsilon for numerical stability (e.g. <c>1e-5f</c>).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when buffer lengths are inconsistent with <paramref name="n"/> and <paramref name="seqLen"/>.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext        ctx,
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        Span<float>         output,
        int                 n,
        int                 seqLen,
        float               eps = 1e-5f)
    {
        if (weight.Length != n)
            throw new ArgumentException($"weight.Length ({weight.Length}) must equal n ({n}).", nameof(weight));
        if (input.Length != seqLen * n)
            throw new ArgumentException($"input.Length ({input.Length}) must equal seqLen × n ({seqLen * n}).", nameof(input));
        if (output.Length != seqLen * n)
            throw new ArgumentException($"output.Length ({output.Length}) must equal seqLen × n ({seqLen * n}).", nameof(output));

        if (seqLen == 0) return;

        unsafe
        {
            fixed (float* pIn  = input)
            fixed (float* pW   = weight)
            fixed (float* pOut = output)
            {
                int code = MetalNative.RmsNormF32(ctx.Handle, pIn, pW, pOut, n, seqLen, eps);
                if (code != 0)
                    throw new InvalidOperationException($"Metal rmsnorm_f32 failed with code {code}.");
            }
        }
    }
}

/// <summary>
/// RMS Normalization (FP16) accelerated via Metal GPU.
/// <c>output[t, i] = input[t, i] / rms(input[t]) * weight[i]</c>.
/// FP16 I/O with FP32 accumulation. Port of <c>rmsnorm_f16.cu</c>.
/// </summary>
public static class RmsNormF16
{
    /// <summary>
    /// Applies FP16 RMS normalization row-wise.
    /// </summary>
    /// <param name="ctx">The Metal context.</param>
    /// <param name="input">Input tensor <c>[seqLen × n]</c> (FP16, read-only).</param>
    /// <param name="weight">Scale vector of length <c>n</c> (FP16, γ parameter).</param>
    /// <param name="output">Output tensor <c>[seqLen × n]</c> (FP16).</param>
    /// <param name="n">Hidden dimension.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="eps">Epsilon for numerical stability.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext       ctx,
        ReadOnlySpan<Half> input,
        ReadOnlySpan<Half> weight,
        Span<Half>         output,
        int                n,
        int                seqLen,
        float              eps = 1e-5f)
    {
        if (weight.Length != n)
            throw new ArgumentException($"weight.Length ({weight.Length}) must equal n ({n}).", nameof(weight));
        if (input.Length != seqLen * n)
            throw new ArgumentException($"input.Length ({input.Length}) must equal seqLen × n ({seqLen * n}).", nameof(input));
        if (output.Length != seqLen * n)
            throw new ArgumentException($"output.Length ({output.Length}) must equal seqLen × n ({seqLen * n}).", nameof(output));

        if (seqLen == 0) return;

        var inU  = MemoryMarshal.Cast<Half, ushort>(input);
        var wU   = MemoryMarshal.Cast<Half, ushort>(weight);
        var outU = MemoryMarshal.Cast<Half, ushort>(output);

        unsafe
        {
            fixed (ushort* pIn  = inU)
            fixed (ushort* pW   = wU)
            fixed (ushort* pOut = outU)
            {
                int code = MetalNative.RmsNormF16(ctx.Handle, pIn, pW, pOut, n, seqLen, eps);
                if (code != 0)
                    throw new InvalidOperationException($"Metal rmsnorm_f16 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers and does not check buffer lengths.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Execute(
        MetalContext ctx,
        nint  input,
        nint  weight,
        nint  output,
        int   n,
        int   seqLen,
        float eps = 1e-5f)
    {
        if (seqLen == 0) return;

        int code = MetalNative.RmsNormF16(
            ctx.Handle,
            (ushort*)input,
            (ushort*)weight,
            (ushort*)output,
            n, seqLen, eps);

        if (code != 0)
            throw new InvalidOperationException($"Metal rmsnorm_f16 failed with code {code}.");
    }
}

/// <summary>
/// RMS Normalization — FP32 residual input, FP32 weight, FP16 output.
/// Used when the residual stream is kept in FP32 but the downstream GEMM needs FP16 activations.
/// Port of <c>rmsnorm_f32in.cu::rmsnorm_f32in_f16out</c>.
/// </summary>
public static class RmsNormF32InF16Out
{
    /// <summary>
    /// Applies RMS normalization row-wise with mixed precision:
    /// <c>output[t, i] = FP16(input[t, i] / rms(input[t]) * weight[i])</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="input">FP32 input tensor of shape <c>[seqLen × n]</c> (read-only).</param>
    /// <param name="weight">FP32 scale vector of length <c>n</c> (γ parameter).</param>
    /// <param name="output">FP16 output tensor of shape <c>[seqLen × n]</c>, written by the kernel.</param>
    /// <param name="n">Hidden dimension.</param>
    /// <param name="seqLen">Number of tokens.</param>
    /// <param name="eps">Epsilon for numerical stability (e.g. <c>1e-5f</c>).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext        ctx,
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        Span<Half>          output,
        int                 n,
        int                 seqLen,
        float               eps = 1e-5f)
    {
        if (weight.Length != n)
            throw new ArgumentException($"weight.Length ({weight.Length}) must equal n ({n}).", nameof(weight));
        if (input.Length != seqLen * n)
            throw new ArgumentException($"input.Length ({input.Length}) must equal seqLen × n ({seqLen * n}).", nameof(input));
        if (output.Length != seqLen * n)
            throw new ArgumentException($"output.Length ({output.Length}) must equal seqLen × n ({seqLen * n}).", nameof(output));

        if (seqLen == 0) return;

        var outU = MemoryMarshal.Cast<Half, ushort>(output);

        unsafe
        {
            fixed (float*  pIn  = input)
            fixed (float*  pW   = weight)
            fixed (ushort* pOut = outU)
            {
                int code = MetalNative.RmsNormF32InF16Out(ctx.Handle, pIn, pW, pOut, n, seqLen, eps);
                if (code != 0)
                    throw new InvalidOperationException($"Metal rmsnorm_f32in_f16out failed with code {code}.");
            }
        }
    }
}
