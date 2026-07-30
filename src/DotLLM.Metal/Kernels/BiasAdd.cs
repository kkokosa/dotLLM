using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// In-place bias addition (FP32 output, FP16 bias) accelerated via Metal GPU.
/// Port of <c>bias_add_f32.cu::bias_add_f32</c>.
/// </summary>
public static class BiasAddF32
{
    /// <summary>
    /// Adds a FP16 bias to each token in-place: <c>output[t, i] += float(bias[i])</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="output">
    /// Flat FP32 buffer of shape <c>[seqLen × dim]</c>, modified in-place.
    /// Must have length equal to <paramref name="dim"/> × <paramref name="seqLen"/>.
    /// </param>
    /// <param name="bias">FP16 bias vector of length <paramref name="dim"/>.</param>
    /// <param name="dim">Feature dimension (inner size).</param>
    /// <param name="seqLen">Number of tokens (outer size).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when buffer lengths are inconsistent with <paramref name="dim"/> and <paramref name="seqLen"/>.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, Span<float> output, ReadOnlySpan<Half> bias, int dim, int seqLen)
    {
        if (bias.Length != dim)
            throw new ArgumentException($"Bias length {bias.Length} does not match dim {dim}.");

        if (output.Length != dim * seqLen)
            throw new ArgumentException($"Output length {output.Length} must equal dim × seqLen ({dim * seqLen}).");

        if (output.IsEmpty)
            return;

        var biasU = MemoryMarshal.Cast<Half, ushort>(bias);

        unsafe
        {
            fixed (float*  pOut  = output)
            fixed (ushort* pBias = biasU)
            {
                int code = MetalNative.BiasAddF32(ctx.Handle, pOut, pBias, (uint)dim, (uint)seqLen);
                if (code != 0)
                    throw new InvalidOperationException($"Metal bias_add_f32 failed with code {code}.");
            }
        }
    }
}

/// <summary>
/// In-place bias addition (FP16 output + FP16 bias) accelerated via Metal GPU.
/// Uses vectorized half2 operations internally.
/// Port of <c>bias_add.cu::bias_add_f16</c>.
/// </summary>
public static class BiasAddF16
{
    /// <summary>
    /// Adds a FP16 bias to each token in-place: <c>output[t, i] += bias[i]</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="output">
    /// Flat FP16 buffer of shape <c>[seqLen × dim]</c>, modified in-place.
    /// Must have length equal to <paramref name="dim"/> × <paramref name="seqLen"/>.
    /// </param>
    /// <param name="bias">FP16 bias vector of length <paramref name="dim"/>.</param>
    /// <param name="dim">Feature dimension (inner size).</param>
    /// <param name="seqLen">Number of tokens (outer size).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when buffer lengths are inconsistent with <paramref name="dim"/> and <paramref name="seqLen"/>.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, Span<Half> output, ReadOnlySpan<Half> bias, int dim, int seqLen)
    {
        if (bias.Length != dim)
            throw new ArgumentException($"Bias length {bias.Length} does not match dim {dim}.");

        if (output.Length != dim * seqLen)
            throw new ArgumentException($"Output length {output.Length} must equal dim × seqLen ({dim * seqLen}).");

        if (output.IsEmpty)
            return;

        var outputU = MemoryMarshal.Cast<Half, ushort>(output);
        var biasU   = MemoryMarshal.Cast<Half, ushort>(bias);

        unsafe
        {
            fixed (ushort* pOut  = outputU)
            fixed (ushort* pBias = biasU)
            {
                int code = MetalNative.BiasAddF16(ctx.Handle, pOut, pBias, (uint)dim, (uint)seqLen);
                if (code != 0)
                    throw new InvalidOperationException($"Metal bias_add_f16 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers and does not check buffer lengths.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Execute(MetalContext ctx, nint output, nint bias, int dim, int seqLen)
    {
        int code = MetalNative.BiasAddF16(ctx.Handle, (ushort*)output, (ushort*)bias, (uint)dim, (uint)seqLen);
        if (code != 0)
            throw new InvalidOperationException($"Metal bias_add_f16 failed with code {code}.");
    }

}
