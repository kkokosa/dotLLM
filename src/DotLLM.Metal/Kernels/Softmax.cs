using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Numerically stable softmax accelerated via Metal GPU.
/// FP16 I/O, FP32 accumulation. One threadgroup per row.
/// Port of <c>softmax.cu::softmax_f16</c>.
/// </summary>
public static class SoftmaxF16
{
    /// <summary>
    /// Applies softmax independently to each row of a 2-D FP16 matrix:
    /// <c>output[row, i] = exp(input[row, i] − max(row)) / Σ exp(input[row, j] − max(row))</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="input">Row-major FP16 input of shape <c>[rows, cols]</c>.</param>
    /// <param name="output">Row-major FP16 output of shape <c>[rows, cols]</c>.</param>
    /// <param name="rows">Number of rows (tokens / batch elements).</param>
    /// <param name="cols">Number of columns (vocabulary size or feature dimension).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when span lengths are inconsistent with <paramref name="rows"/> × <paramref name="cols"/>.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx,
        ReadOnlySpan<Half> input, Span<Half> output,
        int rows, int cols)
    {
        int total = rows * cols;

        if (input.Length != total)
            throw new ArgumentException(
                $"Input length {input.Length} does not match rows × cols ({total}).");

        if (output.Length < total)
            throw new ArgumentException(
                $"Output span is too small ({output.Length} < {total}).");

        if (total == 0)
            return;

        var inputU  = MemoryMarshal.Cast<Half, ushort>(input);
        var outputU = MemoryMarshal.Cast<Half, ushort>(output);

        unsafe
        {
            fixed (ushort* pIn  = inputU)
            fixed (ushort* pOut = outputU)
            {
                int code = MetalNative.SoftmaxF16(ctx.Handle, pIn, pOut, rows, cols);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal softmax_f16 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Convenience overload for a single row (vector softmax).
    /// Equivalent to <c>Execute(ctx, input, output, rows: 1, cols: input.Length)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, ReadOnlySpan<Half> input, Span<Half> output)
        => Execute(ctx, input, output, rows: 1, cols: input.Length);
}
