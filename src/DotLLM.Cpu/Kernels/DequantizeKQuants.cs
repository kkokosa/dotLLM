using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// K-quant dequantization kernels for Q4_K, Q5_K, and Q6_K formats.
/// These formats use 256-element super-blocks with packed sub-block scales.
/// </summary>
public static unsafe partial class Dequantize
{
    /// <summary>Q4_K block size in bytes: 2(d) + 2(dmin) + 12(scales) + 128(qs) = 144.</summary>
    internal const int Q4_K_BlockBytes = 144;

    /// <summary>Q5_K block size in bytes: 2(d) + 2(dmin) + 12(scales) + 32(qh) + 128(qs) = 176.</summary>
    internal const int Q5_K_BlockBytes = 176;

    /// <summary>Q6_K block size in bytes: 128(ql) + 64(qh) + 16(scales) + 2(d) = 210.</summary>
    internal const int Q6_K_BlockBytes = 210;

    /// <summary>Number of elements per K-quant super-block.</summary>
    internal const int KQuantGroupSize = 256;

    // ──────────────────── Scale unpacking ────────────────────

    /// <summary>
    /// Unpacks 12 packed bytes into 8 × (6-bit scale, 6-bit min) pairs for Q4_K/Q5_K.
    /// Matches llama.cpp <c>get_scale_min_k4()</c> in ggml-common.h.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void UnpackQ4Q5Scales(byte* scales12, byte* outScales8, byte* outMins8)
    {
        // Sub-blocks 0-3: low 6 bits from bytes 0-3 (scales) and 4-7 (mins)
        for (int j = 0; j < 4; j++)
        {
            outScales8[j] = (byte)(scales12[j] & 63);
            outMins8[j] = (byte)(scales12[j + 4] & 63);
        }

        // Sub-blocks 4-7: low nibble from bytes 8-11, high 2 bits from bytes 0-7
        for (int j = 4; j < 8; j++)
        {
            outScales8[j] = (byte)((scales12[j + 4] & 0xF) | ((scales12[j - 4] >> 6) << 4));
            outMins8[j] = (byte)((scales12[j + 4] >> 4) | ((scales12[j] >> 6) << 4));
        }
    }

    // ──────────────────── Q6_K ────────────────────

    /// <summary>
    /// Dispatches Q6_K dequantization to the best available implementation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void DequantizeQ6_K(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"Q6_K element count must be a multiple of {KQuantGroupSize}, got {elementCount}",
                nameof(elementCount));

        if (Avx2.IsSupported)
            DequantizeQ6_KAvx2(src, elementCount, dest);
        else
            DequantizeQ6_KScalar(src, elementCount, dest);
    }

    /// <summary>
    /// Scalar Q6_K dequantization.
    /// Layout: ql[128]@0, qh[64]@128, scales[16](int8)@192, d(Half)@208.
    /// Formula: val = d * scales[j] * ((lo4 | hi2&lt;&lt;4) - 32)
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ6_KScalar(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / KQuantGroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;

        for (long b = 0; b < blockCount; b++)
        {
            byte* ql = blockBase;
            byte* qh = blockBase + 128;
            sbyte* scales = (sbyte*)(blockBase + 192);
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase + 208);

            // Process two 128-element halves, matching llama.cpp dequantize_row_q6_K.
            // Within each half: ql lower nibbles → first 64 elements,
            // ql upper nibbles → next 64 elements.
            // qh packs 4×2-bit values per byte for groups of 32 elements.
            for (int half = 0; half < 2; half++)
            {
                int qlOff = half * 64;
                int qhOff = half * 32;
                int scOff = half * 8;

                for (int l = 0; l < 32; l++)
                {
                    int isc = l / 16;
                    int q1 = ((ql[qlOff + l] & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[qlOff + l] >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[qlOff + l + 32] >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;

                    dest[outIdx + l]      = d * scales[scOff + isc] * q1;
                    dest[outIdx + l + 32] = d * scales[scOff + isc + 2] * q2;
                    dest[outIdx + l + 64] = d * scales[scOff + isc + 4] * q3;
                    dest[outIdx + l + 96] = d * scales[scOff + isc + 6] * q4;
                }
                outIdx += 128;
            }

            blockBase += Q6_K_BlockBytes;
        }
    }

    /// <summary>
    /// AVX2-accelerated Q6_K dequantization. Processes 32 values per iteration.
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ6_KAvx2(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / KQuantGroupSize;
        byte* blockBase = (byte*)src;

        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        Vector256<sbyte> bias32 = Vector256.Create((sbyte)32);

        fixed (float* destPtr = dest)
        {
            float* outPtr = destPtr;

            for (long b = 0; b < blockCount; b++)
            {
                byte* ql = blockBase;
                byte* qh = blockBase + 128;
                sbyte* scales = (sbyte*)(blockBase + 192);
                float d = (float)Unsafe.ReadUnaligned<Half>(blockBase + 208);

                // Process 32 values per iteration (2 sub-blocks of 16), 8 iterations for 256 values.
                // Matches llama.cpp: within each half (128 elements), ql lower nibbles produce
                // the first 64 elements and upper nibbles the next 64. qh is indexed per-byte
                // with 2-bit shifts per group of 64 elements.
                Vector256<byte> mask03 = Vector256.Create((byte)0x03);

                for (int sub = 0; sub < 16; sub += 2)
                {
                    int half = sub / 8;
                    int sh = sub % 8;
                    int qlBase = half * 64 + (sh % 4) * 16;
                    bool isUpper = sh >= 4;
                    int qhBase = half * 32;
                    int qhShift = (sh / 2) * 2;

                    // Load 32 ql bytes, extract lower or upper nibbles
                    Vector256<byte> qlRaw = Unsafe.ReadUnaligned<Vector256<byte>>(ql + qlBase);
                    Vector256<byte> nibbles;
                    if (isUpper)
                        nibbles = Avx2.And(
                            Avx2.ShiftRightLogical(qlRaw.AsUInt16(), 4).AsByte(), mask0F);
                    else
                        nibbles = Avx2.And(qlRaw, mask0F);

                    // Load 32 qh bytes, extract 2-bit field at qhShift
                    Vector256<byte> qhVec = Unsafe.ReadUnaligned<Vector256<byte>>(qh + qhBase);
                    Vector256<byte> qhBits = qhShift switch
                    {
                        0 => Avx2.And(qhVec, mask03),
                        2 => Avx2.And(Avx2.ShiftRightLogical(qhVec.AsUInt16(), 2).AsByte(), mask03),
                        4 => Avx2.And(Avx2.ShiftRightLogical(qhVec.AsUInt16(), 4).AsByte(), mask03),
                        _ => Avx2.And(Avx2.ShiftRightLogical(qhVec.AsUInt16(), 6).AsByte(), mask03),
                    };

                    // Combine: q6 = nibble | (qh2 << 4), mask to 6 bits
                    Vector256<byte> q6 = Avx2.And(
                        Avx2.Or(nibbles, Avx2.ShiftLeftLogical(qhBits.AsUInt16(), 4).AsByte()),
                        Vector256.Create((byte)0x3F));

                    Vector256<sbyte> q6signed = Avx2.Subtract(q6.AsSByte(), bias32);

                    // Two sub-blocks of 16: lower 128 bits → scales[sub], upper → scales[sub+1]
                    float s0 = d * scales[sub];
                    float s1 = d * scales[sub + 1];

                    Vector128<sbyte> q6Lo = q6signed.GetLower();
                    Vector128<sbyte> q6Hi = q6signed.GetUpper();

                    Vector256<short> shorts0 = Avx2.ConvertToVector256Int16(q6Lo);
                    Vector256<int> ints0a = Avx2.ConvertToVector256Int32(shorts0.GetLower());
                    Vector256<int> ints0b = Avx2.ConvertToVector256Int32(shorts0.GetUpper());

                    Avx.Store(outPtr, Avx.Multiply(Avx.ConvertToVector256Single(ints0a), Vector256.Create(s0)));
                    Avx.Store(outPtr + 8, Avx.Multiply(Avx.ConvertToVector256Single(ints0b), Vector256.Create(s0)));

                    Vector256<short> shorts1 = Avx2.ConvertToVector256Int16(q6Hi);
                    Vector256<int> ints1a = Avx2.ConvertToVector256Int32(shorts1.GetLower());
                    Vector256<int> ints1b = Avx2.ConvertToVector256Int32(shorts1.GetUpper());

                    Avx.Store(outPtr + 16, Avx.Multiply(Avx.ConvertToVector256Single(ints1a), Vector256.Create(s1)));
                    Avx.Store(outPtr + 24, Avx.Multiply(Avx.ConvertToVector256Single(ints1b), Vector256.Create(s1)));

                    outPtr += 32;
                }

                blockBase += Q6_K_BlockBytes;
            }
        }
    }

    // ──────────────────── Q4_K ────────────────────

    /// <summary>Dispatches Q4_K dequantization.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void DequantizeQ4_K(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"Q4_K element count must be a multiple of {KQuantGroupSize}, got {elementCount}",
                nameof(elementCount));

        if (Avx2.IsSupported)
            DequantizeQ4_KAvx2(src, elementCount, dest);
        else
            DequantizeQ4_KScalar(src, elementCount, dest);
    }

    /// <summary>
    /// Scalar Q4_K dequantization.
    /// Layout: d(Half@0), dmin(Half@2), scales[12]@4, qs[128]@16.
    /// Formula: val = d * scale_j * nibble - dmin * min_j
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ4_KScalar(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / KQuantGroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(blockBase + 2);
            UnpackQ4Q5Scales(blockBase + 4, scBuf, mnBuf);
            byte* qs = blockBase + 16;

            // 4 pairs of sub-blocks (64 elements each), matching llama.cpp.
            // Within each pair: first 32 elements use lower nibbles of 32 qs bytes,
            // next 32 elements use upper nibbles of the same 32 bytes.
            for (int j = 0; j < 8; j++)
            {
                float sc = d * scBuf[j];
                float mn = dmin * mnBuf[j];
                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                for (int i = 0; i < 32; i++)
                {
                    int qsByte = pairIdx * 32 + i;
                    int nibble = nibbleHalf == 0 ? (qs[qsByte] & 0xF) : (qs[qsByte] >> 4);
                    dest[outIdx++] = sc * nibble - mn;
                }
            }

            blockBase += Q4_K_BlockBytes;
        }
    }

    /// <summary>AVX2-accelerated Q4_K dequantization.</summary>
    [SkipLocalsInit]
    internal static void DequantizeQ4_KAvx2(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / KQuantGroupSize;
        byte* blockBase = (byte*)src;

        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        fixed (float* destPtr = dest)
        {
            float* outPtr = destPtr;

            for (long b = 0; b < blockCount; b++)
            {
                float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
                float dmin = (float)Unsafe.ReadUnaligned<Half>(blockBase + 2);
                UnpackQ4Q5Scales(blockBase + 4, scBuf, mnBuf);
                byte* qs = blockBase + 16;

                // Process 2 sub-blocks at a time (64 values from 32 bytes of qs).
                // Matching llama.cpp: lower nibbles of 32 bytes → first 32 elements (sb),
                // upper nibbles → next 32 elements (sb+1). No interleaving.
                for (int sb = 0; sb < 8; sb += 2)
                {
                    float sc0 = d * scBuf[sb];
                    float mn0 = dmin * mnBuf[sb];
                    float sc1 = d * scBuf[sb + 1];
                    float mn1 = dmin * mnBuf[sb + 1];

                    // Load 32 bytes of qs for this pair
                    Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs + (sb / 2) * 32);
                    Vector256<byte> lo = Avx2.And(raw, mask0F);
                    Vector256<byte> hi = Avx2.And(
                        Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);

                    // sb: lower nibbles (32 values), sb+1: upper nibbles (32 values)
                    EmitDequantQ4K_16(lo.GetLower(), sc0, mn0, outPtr);
                    EmitDequantQ4K_16(lo.GetUpper(), sc0, mn0, outPtr + 16);
                    EmitDequantQ4K_16(hi.GetLower(), sc1, mn1, outPtr + 32);
                    EmitDequantQ4K_16(hi.GetUpper(), sc1, mn1, outPtr + 48);

                    outPtr += 64;
                }

                blockBase += Q4_K_BlockBytes;
            }
        }
    }

    /// <summary>Converts 16 unsigned nibbles to float: val = sc * nibble - mn.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void EmitDequantQ4K_16(Vector128<byte> nibbles, float sc, float mn, float* outPtr)
    {
        Vector256<short> shorts = Avx2.ConvertToVector256Int16(nibbles);
        Vector256<int> ints0 = Avx2.ConvertToVector256Int32(shorts.GetLower());
        Vector256<int> ints1 = Avx2.ConvertToVector256Int32(shorts.GetUpper());

        Vector256<float> vSc = Vector256.Create(sc);
        Vector256<float> vMn = Vector256.Create(mn);

        Avx.Store(outPtr, Avx.Subtract(Avx.Multiply(Avx.ConvertToVector256Single(ints0), vSc), vMn));
        Avx.Store(outPtr + 8, Avx.Subtract(Avx.Multiply(Avx.ConvertToVector256Single(ints1), vSc), vMn));
    }

    // ──────────────────── Q5_K ────────────────────

    /// <summary>Dispatches Q5_K dequantization.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void DequantizeQ5_K(nint src, long elementCount, Span<float> dest)
    {
        if (elementCount % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"Q5_K element count must be a multiple of {KQuantGroupSize}, got {elementCount}",
                nameof(elementCount));

        if (Avx2.IsSupported)
            DequantizeQ5_KAvx2(src, elementCount, dest);
        else
            DequantizeQ5_KScalar(src, elementCount, dest);
    }

    /// <summary>
    /// Scalar Q5_K dequantization.
    /// Layout: d(Half@0), dmin(Half@2), scales[12]@4, qh[32]@16, qs[128]@48.
    /// Formula: val = d * scale_j * (lo4 | bit5&lt;&lt;4) - dmin * min_j
    /// </summary>
    [SkipLocalsInit]
    internal static void DequantizeQ5_KScalar(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / KQuantGroupSize;
        byte* blockBase = (byte*)src;
        int outIdx = 0;

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (long b = 0; b < blockCount; b++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(blockBase + 2);
            UnpackQ4Q5Scales(blockBase + 4, scBuf, mnBuf);
            byte* qh = blockBase + 16;
            byte* qs = blockBase + 48;

            // 4 pairs of sub-blocks (64 elements each), matching llama.cpp.
            // Within each pair: first 32 elements use lower nibbles of 32 qs bytes,
            // next 32 elements use upper nibbles of the same 32 bytes.
            // qh is indexed as qh[l] >> j where l is element within the 32-element half
            // and j is the sub-block index (0..7).
            for (int j = 0; j < 8; j++)
            {
                float sc = d * scBuf[j];
                float mn = dmin * mnBuf[j];
                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                for (int i = 0; i < 32; i++)
                {
                    int qsByte = pairIdx * 32 + i;
                    int lo4 = nibbleHalf == 0 ? (qs[qsByte] & 0xF) : (qs[qsByte] >> 4);
                    int bit5 = (qh[i] >> j) & 1;

                    dest[outIdx++] = sc * (lo4 | (bit5 << 4)) - mn;
                }
            }

            blockBase += Q5_K_BlockBytes;
        }
    }

    /// <summary>AVX2-accelerated Q5_K dequantization.</summary>
    [SkipLocalsInit]
    internal static void DequantizeQ5_KAvx2(nint src, long elementCount, Span<float> dest)
    {
        long blockCount = elementCount / KQuantGroupSize;
        byte* blockBase = (byte*)src;

        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];
        fixed (float* destPtr = dest)
        {
            float* outPtr = destPtr;

            for (long b = 0; b < blockCount; b++)
            {
                float d = (float)Unsafe.ReadUnaligned<Half>(blockBase);
                float dmin = (float)Unsafe.ReadUnaligned<Half>(blockBase + 2);
                UnpackQ4Q5Scales(blockBase + 4, scBuf, mnBuf);
                byte* qh = blockBase + 16;
                byte* qs = blockBase + 48;

                // Process 2 sub-blocks at a time (64 values), matching llama.cpp.
                // Lower nibbles → first 32 elements, upper nibbles → next 32.
                // qh: bit j from each of the 32 qh bytes (NOT a flat bitfield).
                for (int sb = 0; sb < 8; sb += 2)
                {
                    float sc0 = d * scBuf[sb];
                    float mn0 = dmin * mnBuf[sb];
                    float sc1 = d * scBuf[sb + 1];
                    float mn1 = dmin * mnBuf[sb + 1];

                    int pairIdx = sb / 2;

                    // Load 32 bytes of qs for this pair
                    Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs + pairIdx * 32);
                    Vector256<byte> lo = Avx2.And(raw, mask0F);
                    Vector256<byte> hi = Avx2.And(
                        Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);

                    // Extract 5th bits from qh: bit sb from each byte for lo, bit sb+1 for hi
                    Vector256<byte> qhVec = Unsafe.ReadUnaligned<Vector256<byte>>(qh);

                    byte bitMask0 = (byte)(1 << sb);
                    Vector256<byte> hasBit0 = Avx2.CompareEqual(
                        Avx2.And(qhVec, Vector256.Create(bitMask0)), Vector256.Create(bitMask0));
                    Vector256<byte> bit5_0 = Avx2.And(hasBit0, Vector256.Create((byte)16));

                    byte bitMask1 = (byte)(1 << (sb + 1));
                    Vector256<byte> hasBit1 = Avx2.CompareEqual(
                        Avx2.And(qhVec, Vector256.Create(bitMask1)), Vector256.Create(bitMask1));
                    Vector256<byte> bit5_1 = Avx2.And(hasBit1, Vector256.Create((byte)16));

                    // Combine nibbles with 5th bit
                    Vector256<byte> q5_sb0 = Avx2.Or(lo, bit5_0);
                    Vector256<byte> q5_sb1 = Avx2.Or(hi, bit5_1);

                    // sb: 32 values from lower nibbles + bit, sb+1: 32 values from upper nibbles + bit
                    EmitDequantQ4K_16(q5_sb0.GetLower(), sc0, mn0, outPtr);
                    EmitDequantQ4K_16(q5_sb0.GetUpper(), sc0, mn0, outPtr + 16);
                    EmitDequantQ4K_16(q5_sb1.GetLower(), sc1, mn1, outPtr + 32);
                    EmitDequantQ4K_16(q5_sb1.GetUpper(), sc1, mn1, outPtr + 48);

                    outPtr += 64;
                }

                blockBase += Q5_K_BlockBytes;
            }
        }
    }

}
