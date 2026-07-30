using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Mixed-precision element-wise addition accelerated via Metal GPU:
/// <c>result_f32[i] = a_f32[i] + b_f16[i]</c>.
/// Used when adding an FP16 projection output into the FP32 residual stream.
/// Port of <c>add_f32.cu::add_f32_f16</c>.
/// </summary>
public static class AddF32F16
{
    /// <summary>
    /// Adds an FP32 vector and an FP16 vector element-wise into an FP32 output.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="a">First input vector (FP32).</param>
    /// <param name="b">Second input vector (FP16).</param>
    /// <param name="result">Output span (FP32). Must be at least as long as <paramref name="a"/>.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when input lengths differ or <paramref name="result"/> is too short.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, ReadOnlySpan<float> a, ReadOnlySpan<Half> b, Span<float> result)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Input spans must have the same length.");

        if (result.Length < a.Length)
            throw new ArgumentException("Result span is too small.");

        if (a.Length == 0)
            return;

        var bU = MemoryMarshal.Cast<Half, ushort>(b);

        unsafe
        {
            fixed (float*  pA = a)
            fixed (ushort* pB = bU)
            fixed (float*  pR = result)
            {
                int code = MetalNative.AddF32F16(ctx.Handle, pA, pB, pR, (uint)a.Length);
                if (code != 0)
                    throw new InvalidOperationException($"Metal add_f32_f16 failed with code {code}.");
            }
        }
    }
}
