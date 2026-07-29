using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Quantization and dequantization kernels for KV-cache compression.
/// Supports Q8_0 (34 bytes per 32 elements) and Q4_0 (18 bytes per 32 elements).
/// </summary>
public static unsafe partial class KvQuantize
{
    /// <summary>Q8_0 block: 2 bytes (Half scale) + 32 bytes (int8 values) = 34 bytes.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Q4_0 block: 2 bytes (Half scale) + 16 bytes (packed nibbles) = 18 bytes.</summary>
    public const int Q4_0BlockBytes = 18;

    /// <summary>Elements per quantization block (both Q8_0 and Q4_0).</summary>
    public const int BlockSize = 32;

    /// <summary>
    /// Returns the byte size of one quantized row of <paramref name="elementCount"/> elements.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int QuantizedRowBytes(int elementCount, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.Q8_0 => elementCount / BlockSize * Q8_0BlockBytes,
        KvCacheDType.Q4_0 => elementCount / BlockSize * Q4_0BlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Not a quantized type")
    };

    // ──────────────────── Q8_0 ────────────────────

    /// <summary>
    /// Quantizes FP32 data to Q8_0. Delegates to <see cref="MatMul.QuantizeF32ToQ8_0"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void F32ToQ8_0(float* src, byte* dest, int elementCount)
        => MatMul.QuantizeF32ToQ8_0(src, dest, elementCount);

    /// <summary>
    /// Dequantizes Q8_0 data to FP32.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Q8_0ToF32(byte* src, float* dest, int elementCount)
    {
        if (elementCount % BlockSize != 0)
            throw new ArgumentException(
                $"elementCount must be a multiple of {BlockSize}, got {elementCount}",
                nameof(elementCount));

        if (Vector256.IsHardwareAccelerated)
            Q8_0ToF32Vector256(src, dest, elementCount);
        else
            Q8_0ToF32Scalar(src, dest, elementCount);
    }

    [SkipLocalsInit]
    internal static void Q8_0ToF32Scalar(byte* src, float* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;

        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = src + block * Q8_0BlockBytes;
            float* blockDst = dest + block * BlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            sbyte* qs = (sbyte*)(blockSrc + 2);

            for (int i = 0; i < BlockSize; i++)
                blockDst[i] = d * qs[i];
        }
    }

    [SkipLocalsInit]
    internal static void Q8_0ToF32Vector256(byte* src, float* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;

        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = src + block * Q8_0BlockBytes;
            float* blockDst = dest + block * BlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            Vector256<float> vScale = Vector256.Create(d);
            sbyte* qs = (sbyte*)(blockSrc + 2);
            ref float blockDstRef = ref Unsafe.AsRef<float>(blockDst);
            Vector256<sbyte> bytes = Vector256.LoadUnsafe(ref Unsafe.AsRef<sbyte>(qs));
            Vector256<short> ints16Lo = Vector256.WidenLower(bytes);
            Vector256<short> ints16Hi = Vector256.WidenUpper(bytes);
            Vector256<int> i0 = Vector256.WidenLower(ints16Lo);
            Vector256<int> i1 = Vector256.WidenUpper(ints16Lo);
            Vector256<int> i2 = Vector256.WidenLower(ints16Hi);
            Vector256<int> i3 = Vector256.WidenUpper(ints16Hi);

            (Vector256.ConvertToSingle(i0) * vScale).StoreUnsafe(ref blockDstRef);
            (Vector256.ConvertToSingle(i1) * vScale).StoreUnsafe(ref blockDstRef, 8);
            (Vector256.ConvertToSingle(i2) * vScale).StoreUnsafe(ref blockDstRef, 16);
            (Vector256.ConvertToSingle(i3) * vScale).StoreUnsafe(ref blockDstRef, 24);
        }
    }

    // ──────────────────── Q4_0 ────────────────────

    /// <summary>
    /// Quantizes FP32 data to Q4_0 format.
    /// Block layout: <c>Half d (2 bytes) + uint8_t qs[16] (16 bytes)</c> = 18 bytes per 32 elements.
    /// Each byte packs two 4-bit values: low nibble = even element, high nibble = odd element.
    /// Dequant convention: <c>val = d * (nibble - 8)</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void F32ToQ4_0(float* src, byte* dest, int elementCount)
    {
        if (elementCount % BlockSize != 0)
            throw new ArgumentException(
                $"elementCount must be a multiple of {BlockSize}, got {elementCount}",
                nameof(elementCount));

        if (Vector256.IsHardwareAccelerated)
            F32ToQ4_0Vector256(src, dest, elementCount);
        else
            F32ToQ4_0Scalar(src, dest, elementCount);
    }

    /// <summary>
    /// Scalar reference implementation of FP32 → Q4_0 quantization.
    /// </summary>
    [SkipLocalsInit]
    internal static void F32ToQ4_0Scalar(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * BlockSize;
            byte* blockDst = dest + block * Q4_0BlockBytes;

            // Find max absolute value.
            float maxAbs = 0;
            for (int i = 0; i < BlockSize; i++)
            {
                float abs = MathF.Abs(blockSrc[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            // Scale: maps max absolute value to ±7 (4-bit signed range centered at 8).
            float d = maxAbs / 7.0f;
            Unsafe.WriteUnaligned(blockDst, (Half)d);

            byte* qs = blockDst + 2;
            if (d == 0)
            {
                // All values are zero — pack 8|8 nibbles.
                for (int j = 0; j < 16; j++)
                    qs[j] = 0x88; // (8 << 4) | 8
            }
            else
            {
                float invD = 1.0f / d;
                for (int j = 0; j < 16; j++)
                {
                    int lo = Math.Clamp((int)MathF.Round(blockSrc[2 * j] * invD) + 8, 0, 15);
                    int hi = Math.Clamp((int)MathF.Round(blockSrc[2 * j + 1] * invD) + 8, 0, 15);
                    qs[j] = (byte)((hi << 4) | lo);
                }
            }
        }
    }

    /// <summary>
    /// Vector256 SIMD implementation of FP32 → Q4_0 quantization.
    /// Vectorizes the max-abs scan and rounding. Nibble packing is scalar.
    /// </summary>
    [SkipLocalsInit]
    internal static void F32ToQ4_0Vector256(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        Vector256<float> vEight = Vector256.Create(8.0f);
        Vector256<float> vZero = Vector256<float>.Zero;
        Vector256<float> vFifteen = Vector256.Create(15.0f);
        int* rounded = stackalloc int[BlockSize];

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * BlockSize;
            byte* blockDst = dest + block * Q4_0BlockBytes;
            ref float blockSrcRef = ref Unsafe.AsRef<float>(blockSrc);
            ref int roundedRef = ref Unsafe.AsRef<int>(rounded);

            // Max-abs scan: 4 loads of 8 floats.
            Vector256<float> v0 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef));
            Vector256<float> v1 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef, 8));
            Vector256<float> v2 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef, 16));
            Vector256<float> v3 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef, 24));

            Vector256<float> max01 = Vector256.MaxNative(v0, v1);
            Vector256<float> max23 = Vector256.MaxNative(v2, v3);
            Vector256<float> maxAll = Vector256.MaxNative(max01, max23);

            // Horizontal max via shuffles.
            float maxAbs = HorizontalMaxVector256(maxAll);

            float d = maxAbs / 7.0f;
            Unsafe.WriteUnaligned(blockDst, (Half)d);

            byte* qs = blockDst + 2;
            if (d == 0)
            {
                for (int j = 0; j < 16; j++)
                    qs[j] = 0x88;
            }
            else
            {
                Vector256<float> vInvD = Vector256.Create(1.0f / d);

                // Round and clamp to [0, 15] using SIMD, then extract scalars for nibble packing.
                // Process 8 floats at a time.
                for (int i = 0; i < BlockSize; i += 8)
                {
                    Vector256<float> vals = Vector256.LoadUnsafe(ref blockSrcRef, (nuint)i);
                    Vector256<float> shifted = vals * vInvD + vEight;
                    Vector256<float> clamped = Vector256.MinNative(
                        Vector256.MaxNative(Vector256.Round(shifted), vZero),
                        vFifteen);
                    Vector256.ConvertToInt32Native(clamped).StoreUnsafe(ref roundedRef, (nuint)i);
                }

                // Pack nibbles: lo = even element, hi = odd element.
                for (int j = 0; j < 16; j++)
                    qs[j] = (byte)((rounded[2 * j + 1] << 4) | rounded[2 * j]);
            }
        }
    }

    /// <summary>
    /// Dequantizes Q4_0 data to FP32.
    /// Block layout: <c>Half d (2 bytes) + uint8_t qs[16] (16 bytes)</c>.
    /// Convention: <c>val = d * (nibble - 8)</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Q4_0ToF32(byte* src, float* dest, int elementCount)
    {
        if (elementCount % BlockSize != 0)
            throw new ArgumentException(
                $"elementCount must be a multiple of {BlockSize}, got {elementCount}",
                nameof(elementCount));

        if (Vector256.IsHardwareAccelerated)
            Q4_0ToF32Vector256(src, dest, elementCount);
        else
            Q4_0ToF32Scalar(src, dest, elementCount);
    }

    /// <summary>
    /// Scalar reference implementation of Q4_0 → FP32 dequantization.
    /// </summary>
    [SkipLocalsInit]
    internal static void Q4_0ToF32Scalar(byte* src, float* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;

        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = src + block * Q4_0BlockBytes;
            float* blockDst = dest + block * BlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            byte* qs = blockSrc + 2;

            for (int j = 0; j < 16; j++)
            {
                byte packed = qs[j];
                int lo = (packed & 0x0F) - 8;
                int hi = (packed >> 4) - 8;
                blockDst[2 * j] = d * lo;
                blockDst[2 * j + 1] = d * hi;
            }
        }
    }

    /// <summary>
    /// Vector256 SIMD implementation of Q4_0 → FP32 dequantization.
    /// Unpacks 16 packed nibble bytes into 32 floats per block.
    /// Output order matches scalar: dest[2j] = lo nibble, dest[2j+1] = hi nibble.
    /// </summary>
    [SkipLocalsInit]
    internal static void Q4_0ToF32Vector256(byte* src, float* dest, int elementCount)
    {
        int blockCount = elementCount / BlockSize;
        Vector256<int> vEight = Vector256.Create(8);

        for (int block = 0; block < blockCount; block++)
        {
            byte* blockSrc = src + block * Q4_0BlockBytes;
            float* blockDst = dest + block * BlockSize;

            float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
            Vector256<float> vScale = Vector256.Create(d);
            byte* qs = blockSrc + 2;
            // Load 16 packed bytes and extract nibbles.
            Vector128<byte> packed = Vector128.LoadUnsafe(ref Unsafe.AsRef<byte>(qs));
            Vector128<byte> loMask = Vector128.Create((byte)0x0F);
            Vector128<byte> loNibbles = packed & loMask;
            Vector128<byte> hiNibbles = Vector128.ShiftRightLogical(packed.AsUInt16(), 4).AsByte() & loMask;

            ref float blockDstRef = ref Unsafe.AsRef<float>(blockDst);
            Vector256<byte> combinedNibbles = Vector256.Create(loNibbles, hiNibbles);
            Vector256<ushort> lo16 = Vector256.WidenLower(combinedNibbles);
            Vector256<ushort> hi16 = Vector256.WidenUpper(combinedNibbles);
            Vector256<byte> interleavedBytes = (lo16 | Vector256.ShiftLeft(hi16, 8)).AsByte();
            Vector256<sbyte> interleaved = interleavedBytes.AsSByte();
            Vector256<short> ints16Lo = Vector256.WidenLower(interleaved);
            Vector256<short> ints16Hi = Vector256.WidenUpper(interleaved);
            Vector256<int> i0 = Vector256.WidenLower(ints16Lo);
            Vector256<int> i1 = Vector256.WidenUpper(ints16Lo);
            Vector256<int> i2 = Vector256.WidenLower(ints16Hi);
            Vector256<int> i3 = Vector256.WidenUpper(ints16Hi);

            ((Vector256.ConvertToSingle(i0 - vEight)) * vScale).StoreUnsafe(ref blockDstRef);
            ((Vector256.ConvertToSingle(i1 - vEight)) * vScale).StoreUnsafe(ref blockDstRef, 8);
            ((Vector256.ConvertToSingle(i2 - vEight)) * vScale).StoreUnsafe(ref blockDstRef, 16);
            ((Vector256.ConvertToSingle(i3 - vEight)) * vScale).StoreUnsafe(ref blockDstRef, 24);
        }
    }

    /// <summary>
    /// Horizontal max of a 256-bit float vector via reduction.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMaxVector256(Vector256<float> v)
    {
        // Compare upper and lower 128-bit halves.
        Vector128<float> hi = v.GetUpper();
        Vector128<float> lo = v.GetLower();
        Vector128<float> max = Vector128.MaxNative(hi, lo);
        // Reduce 4-element 128-bit vector.
        max = Vector128.MaxNative(max, Vector128.ShuffleNative(max, Vector128.Create(2, 3, 0, 1)));
        max = Vector128.MaxNative(max, Vector128.ShuffleNative(max, Vector128.Create(1, 0, 3, 2)));
        return max.ToScalar();
    }
}
