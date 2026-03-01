using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Dequantization kernels that convert quantized tensor data to float32.
/// Supports FP16, Q8_0, and F32 (passthrough). Used at model-load time to convert
/// memory-mapped GGUF tensor data into compute-ready float buffers.
/// </summary>
public static unsafe class Dequantize
{
    /// <summary>Q8_0 block size in bytes: 2 (Half scale) + 32 (sbyte quantized values).</summary>
    private const int Q8_0BlockBytes = 34;

    /// <summary>Number of elements per Q8_0 block.</summary>
    private const int Q8_0GroupSize = 32;

    /// <summary>
    /// Converts quantized tensor data at <paramref name="src"/> to float32 in <paramref name="dest"/>.
    /// </summary>
    /// <param name="src">Pointer to the source tensor data (memory-mapped or allocated).</param>
    /// <param name="elementCount">Number of logical elements to dequantize.</param>
    /// <param name="quantType">Storage format of the source data.</param>
    /// <param name="dest">Destination span for float32 output. Must have length &gt;= <paramref name="elementCount"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException">Unsupported quantization type.</exception>
    /// <exception cref="ArgumentException"><paramref name="dest"/> is too small.</exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ToFloat32(nint src, long elementCount, QuantizationType quantType, Span<float> dest)
    {
        if (dest.Length < elementCount)
            throw new ArgumentException($"Destination span too small: {dest.Length} < {elementCount}", nameof(dest));

        switch (quantType)
        {
            case QuantizationType.F32:
                DequantizeF32(src, elementCount, dest);
                break;
            case QuantizationType.F16:
                DequantizeFp16(src, elementCount, dest);
                break;
            case QuantizationType.Q8_0:
                DequantizeQ8_0(src, elementCount, dest);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(quantType), quantType,
                    $"Unsupported quantization type: {quantType}");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DequantizeF32(nint src, long elementCount, Span<float> dest)
    {
        new ReadOnlySpan<float>((void*)src, (int)elementCount).CopyTo(dest);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DequantizeFp16(nint src, long elementCount, Span<float> dest)
    {
        TensorPrimitives.ConvertToSingle(
            new ReadOnlySpan<Half>((void*)src, (int)elementCount),
            dest);
    }

    [SkipLocalsInit]
    private static void DequantizeQ8_0(nint src, long elementCount, Span<float> dest)
    {
        Debug.Assert(elementCount % Q8_0GroupSize == 0,
            $"Q8_0 element count must be a multiple of {Q8_0GroupSize}, got {elementCount}");

        if (Avx2.IsSupported)
        {
            DequantizeQ8_0Avx2(src, elementCount, dest);
        }
        else
        {
            DequantizeQ8_0Scalar(src, elementCount, dest);
        }
    }

    /// <summary>
    /// Scalar Q8_0 dequantization. Always available as fallback and correctness reference.
    /// Each block: 2-byte Half scale + 32 sbyte quantized values → 32 floats.
    /// Formula: output[i] = (float)scale * qs[i]
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ8_0Scalar(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / Q8_0GroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;

        for (long b = 0; b < blockCount; b++)
        {
            float scale = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            sbyte* qs = (sbyte*)(blockBase + 2);

            for (int i = 0; i < Q8_0GroupSize; i++)
            {
                dest[outIdx++] = scale * qs[i];
            }

            blockBase += Q8_0BlockBytes;
        }
    }

    /// <summary>
    /// AVX2-accelerated Q8_0 dequantization. Processes one 32-element block per iteration
    /// using SIMD widen (sbyte → short → int → float) and broadcast multiply.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ8_0Avx2(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / Q8_0GroupSize;
        byte* blockBase = (byte*)src;

        fixed (float* destPtr = dest)
        {
            float* outPtr = destPtr;

            for (long b = 0; b < blockCount; b++)
            {
                // Read the Half scale and broadcast to all 8 lanes.
                float scale = (float)Unsafe.ReadUnaligned<Half>(blockBase);
                Vector256<float> vScale = Vector256.Create(scale);

                // Load 32 sbytes (quantized values).
                Vector256<sbyte> bytes = Unsafe.ReadUnaligned<Vector256<sbyte>>(blockBase + 2);

                // Widen sbyte → short: lower 16 and upper 16.
                Vector128<sbyte> bytesLo = bytes.GetLower();
                Vector128<sbyte> bytesHi = bytes.GetUpper();

                Vector256<short> shortsLo = Avx2.ConvertToVector256Int16(bytesLo);
                Vector256<short> shortsHi = Avx2.ConvertToVector256Int16(bytesHi);

                // Widen short → int (4 groups of 8).
                Vector256<int> ints0 = Avx2.ConvertToVector256Int32(shortsLo.GetLower());
                Vector256<int> ints1 = Avx2.ConvertToVector256Int32(shortsLo.GetUpper());
                Vector256<int> ints2 = Avx2.ConvertToVector256Int32(shortsHi.GetLower());
                Vector256<int> ints3 = Avx2.ConvertToVector256Int32(shortsHi.GetUpper());

                // Convert int → float and multiply by scale.
                Vector256<float> f0 = Avx.Multiply(Avx.ConvertToVector256Single(ints0), vScale);
                Vector256<float> f1 = Avx.Multiply(Avx.ConvertToVector256Single(ints1), vScale);
                Vector256<float> f2 = Avx.Multiply(Avx.ConvertToVector256Single(ints2), vScale);
                Vector256<float> f3 = Avx.Multiply(Avx.ConvertToVector256Single(ints3), vScale);

                // Store 4×8 = 32 floats.
                Avx.Store(outPtr, f0);
                Avx.Store(outPtr + 8, f1);
                Avx.Store(outPtr + 16, f2);
                Avx.Store(outPtr + 24, f3);

                outPtr += Q8_0GroupSize;
                blockBase += Q8_0BlockBytes;
            }
        }
    }
}
