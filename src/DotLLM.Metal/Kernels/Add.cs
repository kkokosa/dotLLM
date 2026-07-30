using System.Runtime.CompilerServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Element-wise FP32 vector addition accelerated via Metal GPU.
/// Port of <c>add_f32.cu::add_f32</c>.
/// </summary>
public static class AddF32
{
    /// <summary>
    /// Adds two FP32 vectors element-wise: <c>result[i] = a[i] + b[i]</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="a">First input vector (FP32).</param>
    /// <param name="b">Second input vector (FP32).</param>
    /// <param name="result">Output span (FP32). Must be at least as long as <paramref name="a"/>.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when input lengths differ or <paramref name="result"/> is too short.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Input spans must have the same length.");

        if (result.Length < a.Length)
            throw new ArgumentException("Result span is too small.");

        if (a.Length == 0)
            return;

        unsafe
        {
            fixed (float* pA = a)
            fixed (float* pB = b)
            fixed (float* pR = result)
            {
                int code = MetalNative.AddF32(ctx.Handle, pA, pB, pR, (uint)a.Length);
                if (code != 0)
                    throw new InvalidOperationException($"Metal add_f32 failed with code {code}.");
            }
        }
    }
}
