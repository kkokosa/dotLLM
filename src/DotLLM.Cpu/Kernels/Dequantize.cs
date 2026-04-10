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
public static unsafe partial class Dequantize
{
    /// <summary>Q4_0 block size in bytes: 2 (Half scale) + 16 (packed nibble bytes).</summary>
    private const int Q4_0BlockBytes = 18;

    /// <summary>Q8_0 block size in bytes: 2 (Half scale) + 32 (sbyte quantized values).</summary>
    private const int Q8_0BlockBytes = 34;

    /// <summary>Number of elements per Q8_0 block.</summary>
    private const int Q8_0GroupSize = 32;

    /// <summary>Q5_0 block size in bytes: 2 (Half d) + 4 (qh) + 16 (qs) = 22.</summary>
    private const int Q5_0BlockBytes = 22;

    /// <summary>Number of elements per Q5_0 block.</summary>
    private const int Q5_0GroupSize = 32;

    /// <summary>
    /// Returns the byte size of one row of <paramref name="elementCount"/> elements in the given quantization format.
    /// Useful for computing row strides when iterating weight matrices.
    /// </summary>
    public static long RowByteSize(long elementCount, QuantizationType quantType) => quantType switch
    {
        QuantizationType.F32 => elementCount * 4,
        QuantizationType.F16 => elementCount * 2,
        QuantizationType.Q4_0 => elementCount / Q8_0GroupSize * Q4_0BlockBytes,
        QuantizationType.Q8_0 => elementCount / Q8_0GroupSize * Q8_0BlockBytes,
        QuantizationType.Q5_0 => elementCount / Q5_0GroupSize * Q5_0BlockBytes,
        QuantizationType.Q4_K => elementCount / KQuantGroupSize * Q4_K_BlockBytes,
        QuantizationType.Q5_K => elementCount / KQuantGroupSize * Q5_K_BlockBytes,
        QuantizationType.Q6_K => elementCount / KQuantGroupSize * Q6_K_BlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(quantType), quantType,
            $"Unknown quantization type: {quantType}")
    };

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
            case QuantizationType.Q5_0:
                DequantizeQ5_0(src, elementCount, dest);
                break;
            case QuantizationType.Q4_K:
                DequantizeQ4_K(src, elementCount, dest);
                break;
            case QuantizationType.Q5_K:
                DequantizeQ5_K(src, elementCount, dest);
                break;
            case QuantizationType.Q6_K:
                DequantizeQ6_K(src, elementCount, dest);
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
        if (elementCount % Q8_0GroupSize != 0)
            throw new ArgumentException(
                $"Q8_0 element count must be a multiple of {Q8_0GroupSize}, got {elementCount}",
                nameof(elementCount));

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

    // ──────────────────── Q5_0 ────────────────────

    [SkipLocalsInit]
    private static void DequantizeQ5_0(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % Q5_0GroupSize != 0)
            throw new ArgumentException(
                $"Q5_0 element count must be a multiple of {Q5_0GroupSize}, got {elementCount}",
                nameof(elementCount));

        if (Avx2.IsSupported)
        {
            DequantizeQ5_0Avx2(src, elementCount, dest);
        }
        else
        {
            DequantizeQ5_0Scalar(src, elementCount, dest);
        }
    }

    /// <summary>
    /// Scalar Q5_0 dequantization. Block layout (22 bytes, 32 elements):
    /// <c>d(Half@0), qh[4]@2, qs[16]@6</c>.
    /// Formula: <c>value = d * (((qh_bit &lt;&lt; 4) | lo_nibble) - 16)</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ5_0Scalar(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / Q5_0GroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;

        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            uint qh = Unsafe.ReadUnaligned<uint>(blockBase + 2);
            byte* qs = blockBase + 6;

            for (int j = 0; j < 16; j++)
            {
                byte qsByte = qs[j];
                int lo = qsByte & 0xF;
                int hi = (qsByte >> 4) & 0xF;

                int bit5Lo = (int)((qh >> j) & 1);
                int bit5Hi = (int)((qh >> (j + 16)) & 1);

                // Low nibbles → elements 0..15, high nibbles → elements 16..31
                // (matches ggml's dequantize_row_q5_0 output ordering)
                dest[outIdx + j] = d * ((lo | (bit5Lo << 4)) - 16);
                dest[outIdx + j + 16] = d * ((hi | (bit5Hi << 4)) - 16);
            }

            outIdx += Q5_0GroupSize;
            blockBase += Q5_0BlockBytes;
        }
    }

    // ──────────────────── Q5_0 AVX2 ────────────────────

    /// <summary>
    /// AVX2-accelerated Q5_0 dequantization. Processes one 32-element block per iteration:
    /// unpacks low/high nibbles into a 256-bit vector, ORs in the 5th bit from <c>qh</c> via
    /// <see cref="MatMul.ExtractQ5HighBits"/>, subtracts 16 to recover the signed value, then
    /// widens sbyte→short→int→float and multiplies by the broadcast scale.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ5_0Avx2(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / Q5_0GroupSize;
        byte* blockBase = (byte*)src;

        Vector128<byte> nibbleMask = Vector128.Create((byte)0x0F);
        Vector256<sbyte> sixteen = Vector256.Create((sbyte)16);

        fixed (float* destPtr = dest)
        {
            float* outPtr = destPtr;

            for (long b = 0; b < blockCount; b++)
            {
                // Broadcast the Half scale to all 8 lanes.
                float scale = (float)Unsafe.ReadUnaligned<Half>(blockBase);
                Vector256<float> vScale = Vector256.Create(scale);

                // Load the 4-byte high-bit field and the 16-byte packed-nibble payload.
                uint qh = Unsafe.ReadUnaligned<uint>(blockBase + 2);
                Vector128<byte> qsRaw = Unsafe.ReadUnaligned<Vector128<byte>>(blockBase + 6);

                // Unpack nibbles: low 4 bits → elements 0..15, high 4 bits → elements 16..31.
                Vector128<byte> lo128 = Sse2.And(qsRaw, nibbleMask);
                Vector128<byte> hi128 = Sse2.And(
                    Sse2.ShiftRightLogical(qsRaw.AsUInt16(), 4).AsByte(),
                    nibbleMask);

                // Combine halves, OR in the 5th bit (0x10 per set bit), subtract 16 to center.
                Vector256<byte> q5vals = Avx2.Or(
                    Vector256.Create(lo128, hi128),
                    MatMul.ExtractQ5HighBits(qh));
                Vector256<sbyte> centered = Avx2.Subtract(q5vals.AsSByte(), sixteen);

                // Widen sbyte → short → int and convert to float × scale.
                Vector256<short> shortsLo = Avx2.ConvertToVector256Int16(centered.GetLower());
                Vector256<short> shortsHi = Avx2.ConvertToVector256Int16(centered.GetUpper());

                Vector256<int> ints0 = Avx2.ConvertToVector256Int32(shortsLo.GetLower());
                Vector256<int> ints1 = Avx2.ConvertToVector256Int32(shortsLo.GetUpper());
                Vector256<int> ints2 = Avx2.ConvertToVector256Int32(shortsHi.GetLower());
                Vector256<int> ints3 = Avx2.ConvertToVector256Int32(shortsHi.GetUpper());

                Vector256<float> f0 = Avx.Multiply(Avx.ConvertToVector256Single(ints0), vScale);
                Vector256<float> f1 = Avx.Multiply(Avx.ConvertToVector256Single(ints1), vScale);
                Vector256<float> f2 = Avx.Multiply(Avx.ConvertToVector256Single(ints2), vScale);
                Vector256<float> f3 = Avx.Multiply(Avx.ConvertToVector256Single(ints3), vScale);

                Avx.Store(outPtr, f0);
                Avx.Store(outPtr + 8, f1);
                Avx.Store(outPtr + 16, f2);
                Avx.Store(outPtr + 24, f3);

                outPtr += Q5_0GroupSize;
                blockBase += Q5_0BlockBytes;
            }
        }
    }

    // ──────────────────── Q8_0 AVX2 ────────────────────

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
