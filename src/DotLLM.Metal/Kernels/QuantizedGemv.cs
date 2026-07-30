using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Quantized General Matrix-Vector multiplication accelerated via Metal GPU.
/// Direct translation of quantized_gemv_f32in.cu and quantized_gemv.cu.
/// </summary>
public static class QuantizedGemv
{
    private const int s_q80BlockSize  = 32;
    private const int s_q80BlockBytes = 34;

    /// <summary>
    /// Multiplies a Q8_0 quantized weight matrix by a float32 input vector.
    /// <c>y[i] = dot(dequant(weight[i, :]), x)</c> for each row <c>i</c>.
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="weight">
    /// Q8_0 weight matrix, row-major.
    /// Length must equal <c>n × (k/32) × 34</c> bytes.
    /// </param>
    /// <param name="x">Float32 input vector, length <c>k</c>.</param>
    /// <param name="y">Float32 output vector, length <c>n</c>.</param>
    /// <param name="n">Number of output rows.</param>
    /// <param name="k">Number of input columns. Must be a multiple of 32.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q8_0F32In(
        MetalContext       ctx,
        ReadOnlySpan<byte> weight,
        ReadOnlySpan<float> x,
        Span<float>         y,
        int n,
        int k)
    {
        if (k % s_q80BlockSize != 0)
            throw new ArgumentException(
                $"k ({k}) must be a multiple of {s_q80BlockSize} for Q8_0.", nameof(k));

        int blocksPerRow    = k / s_q80BlockSize;
        int expectedWeightLen = n * blocksPerRow * s_q80BlockBytes;
        if (weight.Length != expectedWeightLen)
            throw new ArgumentException(
                $"weight.Length ({weight.Length}) must equal n × (k/32) × {s_q80BlockBytes} = {expectedWeightLen}.",
                nameof(weight));
        if (x.Length != k)
            throw new ArgumentException(
                $"x.Length ({x.Length}) must equal k ({k}).", nameof(x));
        if (y.Length != n)
            throw new ArgumentException(
                $"y.Length ({y.Length}) must equal n ({n}).", nameof(y));

        unsafe
        {
            fixed (byte*  pW = weight)
            fixed (float* pX = x)
            fixed (float* pY = y)
            {
                int code = MetalNative.QuantizedGemvQ8_0F32In(ctx.Handle, pW, pX, pY, n, k);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal quantized_gemv_q8_0_f32in failed with code {code}.");
            }
        }
    }

    // ── FP16 I/O variants — port of quantized_gemv.cu ────────────────────────

    private static void ValidateF16(
        ReadOnlySpan<byte> weight,
        ReadOnlySpan<Half> x,
        Span<Half>         y,
        int n,
        int k,
        int blockSize,
        int blockBytes,
        string paramName)
    {
        if (k % blockSize != 0)
            throw new ArgumentException(
                $"k ({k}) must be a multiple of {blockSize} for {paramName}.", nameof(k));
        int blocksPerRow = k / blockSize;
        int expectedW = n * blocksPerRow * blockBytes;
        if (weight.Length != expectedW)
            throw new ArgumentException(
                $"weight.Length ({weight.Length}) must equal n × (k/{blockSize}) × {blockBytes} = {expectedW}.",
                nameof(weight));
        if (x.Length != k)
            throw new ArgumentException($"x.Length ({x.Length}) must equal k ({k}).", nameof(x));
        if (y.Length != n)
            throw new ArgumentException($"y.Length ({y.Length}) must equal n ({n}).", nameof(y));
    }

    /// <summary>
    /// Multiplies a Q8_0 quantized weight matrix by a FP16 input vector (FP16 output).
    /// <c>y[i] = dot(dequant(weight[i, :]), x)</c> for each row <c>i</c>.
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="weight">Q8_0 weight matrix. Length must equal <c>n × (k/32) × 34</c> bytes.</param>
    /// <param name="x">FP16 input vector, length <c>k</c>.</param>
    /// <param name="y">FP16 output vector, length <c>n</c>.</param>
    /// <param name="n">Number of output rows.</param>
    /// <param name="k">Number of input columns. Must be a multiple of 32.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q8_0(
        MetalContext       ctx,
        ReadOnlySpan<byte> weight,
        ReadOnlySpan<Half> x,
        Span<Half>         y,
        int n,
        int k)
    {
        ValidateF16(weight, x, y, n, k, 32, 34, "Q8_0");
        var xU = MemoryMarshal.Cast<Half, ushort>(x);
        var yU = MemoryMarshal.Cast<Half, ushort>(y);
        unsafe
        {
            fixed (byte*   pW = weight)
            fixed (ushort* pX = xU)
            fixed (ushort* pY = yU)
            {
                int code = MetalNative.QuantizedGemvQ8_0(ctx.Handle, pW, pX, pY, n, k);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal quantized_gemv_q8_0 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Multiplies a Q5_0 quantized weight matrix by a FP16 input vector (FP16 output).
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="weight">Q5_0 weight matrix. Length must equal <c>n × (k/32) × 22</c> bytes.</param>
    /// <param name="x">FP16 input vector, length <c>k</c>.</param>
    /// <param name="y">FP16 output vector, length <c>n</c>.</param>
    /// <param name="n">Number of output rows.</param>
    /// <param name="k">Number of input columns. Must be a multiple of 32.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q5_0(
        MetalContext       ctx,
        ReadOnlySpan<byte> weight,
        ReadOnlySpan<Half> x,
        Span<Half>         y,
        int n,
        int k)
    {
        ValidateF16(weight, x, y, n, k, 32, 22, "Q5_0");
        var xU = MemoryMarshal.Cast<Half, ushort>(x);
        var yU = MemoryMarshal.Cast<Half, ushort>(y);
        unsafe
        {
            fixed (byte*   pW = weight)
            fixed (ushort* pX = xU)
            fixed (ushort* pY = yU)
            {
                int code = MetalNative.QuantizedGemvQ5_0(ctx.Handle, pW, pX, pY, n, k);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal quantized_gemv_q5_0 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Multiplies a Q4_K quantized weight matrix by a FP16 input vector (FP16 output).
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="weight">Q4_K weight matrix. Length must equal <c>n × (k/256) × 144</c> bytes.</param>
    /// <param name="x">FP16 input vector, length <c>k</c>.</param>
    /// <param name="y">FP16 output vector, length <c>n</c>.</param>
    /// <param name="n">Number of output rows.</param>
    /// <param name="k">Number of input columns. Must be a multiple of 256.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q4_K(
        MetalContext       ctx,
        ReadOnlySpan<byte> weight,
        ReadOnlySpan<Half> x,
        Span<Half>         y,
        int n,
        int k)
    {
        ValidateF16(weight, x, y, n, k, 256, 144, "Q4_K");
        var xU = MemoryMarshal.Cast<Half, ushort>(x);
        var yU = MemoryMarshal.Cast<Half, ushort>(y);
        unsafe
        {
            fixed (byte*   pW = weight)
            fixed (ushort* pX = xU)
            fixed (ushort* pY = yU)
            {
                int code = MetalNative.QuantizedGemvQ4_K(ctx.Handle, pW, pX, pY, n, k);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal quantized_gemv_q4_k failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Multiplies a Q5_K quantized weight matrix by a FP16 input vector (FP16 output).
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="weight">Q5_K weight matrix. Length must equal <c>n × (k/256) × 176</c> bytes.</param>
    /// <param name="x">FP16 input vector, length <c>k</c>.</param>
    /// <param name="y">FP16 output vector, length <c>n</c>.</param>
    /// <param name="n">Number of output rows.</param>
    /// <param name="k">Number of input columns. Must be a multiple of 256.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q5_K(
        MetalContext       ctx,
        ReadOnlySpan<byte> weight,
        ReadOnlySpan<Half> x,
        Span<Half>         y,
        int n,
        int k)
    {
        ValidateF16(weight, x, y, n, k, 256, 176, "Q5_K");
        var xU = MemoryMarshal.Cast<Half, ushort>(x);
        var yU = MemoryMarshal.Cast<Half, ushort>(y);
        unsafe
        {
            fixed (byte*   pW = weight)
            fixed (ushort* pX = xU)
            fixed (ushort* pY = yU)
            {
                int code = MetalNative.QuantizedGemvQ5_K(ctx.Handle, pW, pX, pY, n, k);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal quantized_gemv_q5_k failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Multiplies a Q6_K quantized weight matrix by a FP16 input vector (FP16 output).
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="weight">Q6_K weight matrix. Length must equal <c>n × (k/256) × 210</c> bytes.</param>
    /// <param name="x">FP16 input vector, length <c>k</c>.</param>
    /// <param name="y">FP16 output vector, length <c>n</c>.</param>
    /// <param name="n">Number of output rows.</param>
    /// <param name="k">Number of input columns. Must be a multiple of 256.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q6_K(
        MetalContext       ctx,
        ReadOnlySpan<byte> weight,
        ReadOnlySpan<Half> x,
        Span<Half>         y,
        int n,
        int k)
    {
        ValidateF16(weight, x, y, n, k, 256, 210, "Q6_K");
        var xU = MemoryMarshal.Cast<Half, ushort>(x);
        var yU = MemoryMarshal.Cast<Half, ushort>(y);
        unsafe
        {
            fixed (byte*   pW = weight)
            fixed (ushort* pX = xU)
            fixed (ushort* pY = yU)
            {
                int code = MetalNative.QuantizedGemvQ6_K(ctx.Handle, pW, pX, pY, n, k);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal quantized_gemv_q6_k failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Returns true if the given quantization format is supported.
    /// </summary>
    /// <param name="wQuantizedFormat"></param>
    /// <returns></returns>
    public static bool Supports(QuantizationType wQuantizedFormat)
    {
        return wQuantizedFormat is
            QuantizationType.Q8_0 or
            QuantizationType.Q5_0 or
            QuantizationType.Q4_K or
            QuantizationType.Q5_K or
            QuantizationType.Q6_K;
    }

    /// <summary>
    /// Dispatches a kernel to perform a quantized matrix-vector multiplication.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="format"></param>
    /// <param name="w"></param>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <param name="n"></param>
    /// <param name="k"></param>
    /// <exception cref="NotSupportedException"></exception>
    /// <exception cref="InvalidOperationException"></exception>
    public static unsafe void Dispatch(MetalContext context, QuantizationType format,
        nint w, nint x, nint y, int n, int k)
    {
        int code;
        switch (format)
        {
            case QuantizationType.Q8_0:
                code = MetalNative.QuantizedGemvQ8_0(context.Handle, (byte*)w, (ushort*)x, (ushort*)y, n, k);
                break;
            case QuantizationType.Q5_0:
                code = MetalNative.QuantizedGemvQ5_0(context.Handle, (byte*)w, (ushort*)x, (ushort*)y, n, k);
                break;
            case QuantizationType.Q4_K:
                code = MetalNative.QuantizedGemvQ4_K(context.Handle, (byte*)w, (ushort*)x, (ushort*)y, n, k);
                break;
            case QuantizationType.Q5_K:
                code = MetalNative.QuantizedGemvQ5_K(context.Handle, (byte*)w, (ushort*)x, (ushort*)y, n, k);
                break;
            case QuantizationType.Q6_K:
                code = MetalNative.QuantizedGemvQ6_K(context.Handle, (byte*)w, (ushort*)x, (ushort*)y, n, k);
                break;
            default:
                throw new NotSupportedException($"Quantized GEMV not supported for {format}.");
        }

        if (code != 0)
        {
            throw new InvalidOperationException($"Metal quantized_gemv_q8_0 failed with code {code}.");
        }
    }
}
