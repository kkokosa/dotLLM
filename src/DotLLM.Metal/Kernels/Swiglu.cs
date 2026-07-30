using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Fused SwiGLU activation (FP32) accelerated via Metal GPU.
/// <c>result[i] = SiLU(gate[i]) * up[i]</c> where <c>SiLU(x) = x / (1 + exp(−x))</c>.
/// Port of <c>swiglu_f32.cu::swiglu_f32</c>.
/// </summary>
public static class SwigluF32
{
    /// <summary>
    /// Applies fused SwiGLU element-wise: <c>result[i] = SiLU(gate[i]) * up[i]</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="gate">Gate projection vector (FP32).</param>
    /// <param name="up">Up projection vector (FP32). Must be the same length as <paramref name="gate"/>.</param>
    /// <param name="result">Output span (FP32). Must be at least as long as <paramref name="gate"/>.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when input lengths differ or <paramref name="result"/> is too short.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> result)
    {
        if (gate.Length != up.Length)
            throw new ArgumentException("Gate and up spans must have the same length.");

        if (result.Length < gate.Length)
            throw new ArgumentException("Result span is too small.");

        if (gate.Length == 0)
            return;

        unsafe
        {
            fixed (float* pG = gate)
            fixed (float* pU = up)
            fixed (float* pR = result)
            {
                int code = MetalNative.SwigluF32(ctx.Handle, pG, pU, pR, (uint)gate.Length);
                if (code != 0)
                    throw new InvalidOperationException($"Metal swiglu_f32 failed with code {code}.");
            }
        }
    }
}

/// <summary>
/// Fused SwiGLU activation (FP16) accelerated via Metal GPU.
/// <c>result[i] = SiLU(gate[i]) * up[i]</c> where <c>SiLU(x) = x / (1 + exp(−x))</c>.
/// Uses vectorized half2 operations internally with FP32 precision for sigmoid.
/// Port of <c>swiglu.cu::swiglu_f16</c>.
/// </summary>
public static class SwigluF16
{
    /// <summary>
    /// Applies fused SwiGLU element-wise: <c>result[i] = SiLU(gate[i]) * up[i]</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="gate">Gate projection vector (FP16).</param>
    /// <param name="up">Up projection vector (FP16). Must be the same length as <paramref name="gate"/>.</param>
    /// <param name="result">Output span (FP16). Must be at least as long as <paramref name="gate"/>.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when input lengths differ or <paramref name="result"/> is too short.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, ReadOnlySpan<Half> gate, ReadOnlySpan<Half> up, Span<Half> result)
    {
        if (gate.Length != up.Length)
            throw new ArgumentException("Gate and up spans must have the same length.");

        if (result.Length < gate.Length)
            throw new ArgumentException("Result span is too small.");

        if (gate.Length == 0)
            return;

        var gateU   = MemoryMarshal.Cast<Half, ushort>(gate);
        var upU     = MemoryMarshal.Cast<Half, ushort>(up);
        var resultU = MemoryMarshal.Cast<Half, ushort>(result);

        unsafe
        {
            fixed (ushort* pG = gateU)
            fixed (ushort* pU = upU)
            fixed (ushort* pR = resultU)
            {
                int code = MetalNative.SwigluF16(ctx.Handle, pG, pU, pR, (uint)gate.Length);
                if (code != 0)
                    throw new InvalidOperationException($"Metal swiglu_f16 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers and does not check buffer lengths.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Execute(MetalContext ctx, nint gate, nint up, nint result, int gateLength)
    {
        int code = MetalNative.SwigluF16(ctx.Handle, (ushort*)gate, (ushort*)up, (ushort*)result, (uint)gateLength);
        if (code != 0)
            throw new InvalidOperationException($"Metal swiglu_f16 failed with code {code}.");
    }
}
