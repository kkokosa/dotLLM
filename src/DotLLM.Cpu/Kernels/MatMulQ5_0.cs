using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Q5_0 × Q8_1 matrix multiplication kernels.
/// Q5_0 block: 22 bytes = Half(d) + uint32(qh) + byte[16](qs) → 32 elements.
/// Input is quantized to Q8_1 (precomputed block sums), then fused integer dot product avoids dequant-to-float.
/// Q8_1 precomputes <c>s = d * sum(qs)</c> per block, eliminating 4 SIMD ops per block from the hot path.
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>Q5_0 block size in bytes: 2 (Half d) + 4 (uint32 qh) + 16 (byte[16] qs).</summary>
    private const int Q5_0BlockBytes = 22;

    /// <summary>Number of elements per Q5_0 block.</summary>
    private const int Q5_0GroupSize = 32;

    // ──────────────────── Q5_0 × Q8_1 Scalar ────────────────────

    /// <summary>
    /// Scalar Q5_0 × Q8_1 dot product.
    /// Layout: Q5_0: d(Half@0), qh(uint32@2), qs[16]@6.
    /// Layout: Q8_1: d(Half@0), s(Half@2), qs[32]@4.
    /// Uses precomputed <c>s = d8 * sum(q8)</c> to eliminate per-block sum computation.
    /// Formula: <c>sumf += d5 * d8 * sum(q5_unsigned * q8) - 16 * d5 * s8</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_0Q8_1Scalar(byte* q5, byte* q8, int blockCount)
    {
        float sumf = 0;
        float offsetSum = 0;

        for (int block = 0; block < blockCount; block++)
        {
            byte* q5Block = q5 + block * Q5_0BlockBytes;
            byte* q8Block = q8 + block * Q8_1BlockBytes;

            float d5 = (float)Unsafe.ReadUnaligned<Half>(q5Block);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);
            float s8 = (float)Unsafe.ReadUnaligned<Half>(q8Block + 2);

            uint qh = Unsafe.ReadUnaligned<uint>(q5Block + 2);
            byte* qs = q5Block + 6;
            sbyte* q8v = (sbyte*)(q8Block + 4);

            int sumi = 0;
            for (int j = 0; j < 16; j++)
            {
                int x0 = (qs[j] & 0xF) | (((int)((qh >> j) & 1)) << 4);
                int x1 = ((qs[j] >> 4) & 0xF) | (((int)((qh >> (j + 16)) & 1)) << 4);
                sumi += x0 * q8v[j] + x1 * q8v[j + 16];
            }

            sumf += d5 * d8 * sumi;
            offsetSum += d5 * s8;
        }

        return sumf - 16.0f * offsetSum;
    }

    // ──────────────────── Q5_0 × Q8_0 Scalar (kept for cross-verification) ────────────────────

    /// <summary>
    /// Scalar Q5_0 × Q8_0 dot product.
    /// Layout: d(Half@0), qh(uint32@2), qs[16]@6.
    /// Formula: <c>sumf += d5 * d8 * sum((q5_val - 16) * q8_val)</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_0Q8_0Scalar(byte* q5, byte* q8, int blockCount)
    {
        float sumf = 0;

        for (int block = 0; block < blockCount; block++)
        {
            byte* q5Block = q5 + block * Q5_0BlockBytes;
            byte* q8Block = q8 + block * Q8_0BlockBytes;

            float d5 = (float)Unsafe.ReadUnaligned<Half>(q5Block);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);

            uint qh = Unsafe.ReadUnaligned<uint>(q5Block + 2);
            byte* qs = q5Block + 6;
            sbyte* q8v = (sbyte*)(q8Block + 2);

            int sumi = 0;
            for (int j = 0; j < 16; j++)
            {
                int x0 = (qs[j] & 0xF) | (((int)((qh >> j) & 1)) << 4);
                int x1 = ((qs[j] >> 4) & 0xF) | (((int)((qh >> (j + 16)) & 1)) << 4);
                sumi += (x0 - 16) * q8v[j] + (x1 - 16) * q8v[j + 16];
            }

            sumf += d5 * d8 * sumi;
        }

        return sumf;
    }

    // ──────────────────── Q5_0 × Q8_1 AVX2 ────────────────────

    // Pre-computed constants for SIMD bit extraction of Q5_0 high bits.
    // vpshufb mask: replicate each qh byte to 8 positions within its 128-bit lane.
    // Lane 0: byte0×8, byte1×8 (elements 0-15). Lane 1: byte2×8, byte3×8 (elements 16-31).
    private static readonly Vector256<byte> Q5_0_QhShuffleMask = Vector256.Create(
        (byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);

    // Bit masks: [1,2,4,8,16,32,64,128] repeated 4 times to isolate each bit in the replicated byte.
    private static readonly Vector256<byte> Q5_0_BitMask = Vector256.Create(
        (byte)1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);

    /// <summary>
    /// Extracts 32 high bits from Q5_0 <paramref name="qh"/> into a 256-bit vector
    /// where each byte is 0x10 (16) if the corresponding bit was set, 0x00 otherwise.
    /// Uses vpshufb to broadcast each qh byte to 8 positions, then bit masking.
    /// Byte j (for j in 0..31) corresponds to bit j of <paramref name="qh"/>.
    /// </summary>
    /// <remarks>
    /// Shared with <c>Dequantize.DequantizeQ5_0Avx2</c> — keep <c>internal</c> so both
    /// the GEMV fused vec_dot and the dequant kernel use the same bit-extraction routine.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector256<byte> ExtractQ5HighBits(uint qh)
    {
        // Broadcast the 4 qh bytes into a vector — set1_epi32 puts same 4 bytes in every lane
        Vector256<byte> qhVec = Vector256.Create(qh).AsByte();
        // Shuffle: replicate byte0→positions 0-7, byte1→8-15, byte2→16-23, byte3→24-31
        Vector256<byte> qhBytes = Avx2.Shuffle(qhVec, Q5_0_QhShuffleMask);
        // AND with bit mask to isolate each bit, then CMPEQ to get 0xFF or 0x00
        Vector256<byte> bits = Avx2.And(qhBytes, Q5_0_BitMask);
        Vector256<byte> bitSet = Avx2.CompareEqual(bits, Q5_0_BitMask);
        // AND with 0x10 to convert 0xFF→0x10 (16), 0x00→0x00
        return Avx2.And(bitSet, Vector256.Create((byte)0x10));
    }

    /// <summary>
    /// AVX2-accelerated Q5_0 × Q8_1 dot product with 2-block loop unrolling.
    /// Uses Q8_1 precomputed <c>s = d * sum(qs)</c> to avoid 4 SIMD ops per block
    /// (vpmaddubsw + vpmaddwd + vpslld + vpsubd) that were needed for runtime q8 sum.
    /// 2-block unrolling interleaves loads and arithmetic for better ILP.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_0Q8_1Avx2(byte* q5, byte* q8, int blockCount)
    {
        Vector256<float> accA = Vector256<float>.Zero;
        Vector256<float> accB = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);
        float offsetSum = 0;

        int block = 0;

        // 2-block unrolled loop — dual accumulators break the FMA dependency chain for ILP
        for (; block + 1 < blockCount; block += 2)
        {
            // Block A
            byte* q5BlockA = q5 + block * Q5_0BlockBytes;
            byte* q8BlockA = q8 + block * Q8_1BlockBytes;

            float d5A = (float)Unsafe.ReadUnaligned<Half>(q5BlockA);
            float d8A = (float)Unsafe.ReadUnaligned<Half>(q8BlockA);
            float s8A = (float)Unsafe.ReadUnaligned<Half>(q8BlockA + 2);

            uint qhA = Unsafe.ReadUnaligned<uint>(q5BlockA + 2);
            Vector128<byte> qsRawA = Unsafe.ReadUnaligned<Vector128<byte>>(q5BlockA + 6);

            // Block B — interleave loads
            byte* q5BlockB = q5 + (block + 1) * Q5_0BlockBytes;
            byte* q8BlockB = q8 + (block + 1) * Q8_1BlockBytes;

            float d5B = (float)Unsafe.ReadUnaligned<Half>(q5BlockB);
            float d8B = (float)Unsafe.ReadUnaligned<Half>(q8BlockB);
            float s8B = (float)Unsafe.ReadUnaligned<Half>(q8BlockB + 2);

            uint qhB = Unsafe.ReadUnaligned<uint>(q5BlockB + 2);
            Vector128<byte> qsRawB = Unsafe.ReadUnaligned<Vector128<byte>>(q5BlockB + 6);

            // Block A: unpack nibbles + high bits
            Vector128<byte> lo128A = Sse2.And(qsRawA, Vector128.Create((byte)0x0F));
            Vector128<byte> hi128A = Sse2.And(
                Sse2.ShiftRightLogical(qsRawA.AsUInt16(), 4).AsByte(),
                Vector128.Create((byte)0x0F));
            Vector256<byte> q5valsA = Avx2.Or(Vector256.Create(lo128A, hi128A), ExtractQ5HighBits(qhA));

            Vector256<sbyte> q8ValsA = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8BlockA + 4);
            Vector256<short> prodA = Avx2.MultiplyAddAdjacent(q5valsA, q8ValsA);
            Vector256<int> prodSumA = Avx2.MultiplyAddAdjacent(prodA, ones);

            // Block B: unpack nibbles + high bits
            Vector128<byte> lo128B = Sse2.And(qsRawB, Vector128.Create((byte)0x0F));
            Vector128<byte> hi128B = Sse2.And(
                Sse2.ShiftRightLogical(qsRawB.AsUInt16(), 4).AsByte(),
                Vector128.Create((byte)0x0F));
            Vector256<byte> q5valsB = Avx2.Or(Vector256.Create(lo128B, hi128B), ExtractQ5HighBits(qhB));

            Vector256<sbyte> q8ValsB = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8BlockB + 4);
            Vector256<short> prodB = Avx2.MultiplyAddAdjacent(q5valsB, q8ValsB);
            Vector256<int> prodSumB = Avx2.MultiplyAddAdjacent(prodB, ones);

            // FMA into separate accumulators — block B's FMA can issue without waiting for A
            float scaleA = d5A * d8A;
            float scaleB = d5B * d8B;
            if (Fma.IsSupported)
            {
                accA = Fma.MultiplyAdd(Vector256.Create(scaleA),
                    Avx.ConvertToVector256Single(prodSumA), accA);
                accB = Fma.MultiplyAdd(Vector256.Create(scaleB),
                    Avx.ConvertToVector256Single(prodSumB), accB);
            }
            else
            {
                accA = Avx.Add(accA, Avx.Multiply(Vector256.Create(scaleA),
                    Avx.ConvertToVector256Single(prodSumA)));
                accB = Avx.Add(accB, Avx.Multiply(Vector256.Create(scaleB),
                    Avx.ConvertToVector256Single(prodSumB)));
            }

            offsetSum += d5A * s8A + d5B * s8B;
        }

        // Merge dual accumulators
        Vector256<float> acc = Avx.Add(accA, accB);

        // Single-block tail for odd block count
        if (block < blockCount)
        {
            byte* q5Block = q5 + block * Q5_0BlockBytes;
            byte* q8Block = q8 + block * Q8_1BlockBytes;

            float d5 = (float)Unsafe.ReadUnaligned<Half>(q5Block);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);
            float s8 = (float)Unsafe.ReadUnaligned<Half>(q8Block + 2);

            uint qh = Unsafe.ReadUnaligned<uint>(q5Block + 2);
            Vector128<byte> qsRaw = Unsafe.ReadUnaligned<Vector128<byte>>(q5Block + 6);
            Vector128<byte> lo128 = Sse2.And(qsRaw, Vector128.Create((byte)0x0F));
            Vector128<byte> hi128 = Sse2.And(
                Sse2.ShiftRightLogical(qsRaw.AsUInt16(), 4).AsByte(),
                Vector128.Create((byte)0x0F));
            Vector256<byte> q5vals = Avx2.Or(Vector256.Create(lo128, hi128), ExtractQ5HighBits(qh));

            Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8Block + 4);
            Vector256<short> prod = Avx2.MultiplyAddAdjacent(q5vals, q8Vals);
            Vector256<int> prodSum = Avx2.MultiplyAddAdjacent(prod, ones);

            float scale = d5 * d8;
            if (Fma.IsSupported)
                acc = Fma.MultiplyAdd(Vector256.Create(scale),
                    Avx.ConvertToVector256Single(prodSum), acc);
            else
                acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(scale),
                    Avx.ConvertToVector256Single(prodSum)));

            offsetSum += d5 * s8;
        }

        return HorizontalSumAvx2Float(acc) - 16.0f * offsetSum;
    }

    // ──────────────────── Q5_0 × Q8_0 AVX2 (kept for cross-verification) ────────────────────

    /// <summary>
    /// AVX2-accelerated Q5_0 × Q8_0 dot product (legacy — used for cross-verification with Q8_1 path).
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_0Q8_0Avx2(byte* q5, byte* q8, int blockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);

        for (int block = 0; block < blockCount; block++)
        {
            byte* q5Block = q5 + block * Q5_0BlockBytes;
            byte* q8Block = q8 + block * Q8_0BlockBytes;

            float d5 = (float)Unsafe.ReadUnaligned<Half>(q5Block);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);

            uint qh = Unsafe.ReadUnaligned<uint>(q5Block + 2);

            Vector128<byte> qsRaw = Unsafe.ReadUnaligned<Vector128<byte>>(q5Block + 6);
            Vector128<byte> lo128 = Sse2.And(qsRaw, Vector128.Create((byte)0x0F));
            Vector128<byte> hi128 = Sse2.And(
                Sse2.ShiftRightLogical(qsRaw.AsUInt16(), 4).AsByte(),
                Vector128.Create((byte)0x0F));
            Vector256<byte> nibbles = Vector256.Create(lo128, hi128);

            Vector256<byte> bit5Vec = ExtractQ5HighBits(qh);
            Vector256<byte> q5vals = Avx2.Or(nibbles, bit5Vec);

            Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8Block + 2);

            Vector256<short> prod = Avx2.MultiplyAddAdjacent(q5vals, q8Vals);
            Vector256<int> prodSum = Avx2.MultiplyAddAdjacent(prod, ones);

            Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), q8Vals);
            Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);

            Vector256<int> adjusted = Avx2.Subtract(prodSum,
                Avx2.ShiftLeftLogical(q8Sum, 4));

            float scale = d5 * d8;
            if (Fma.IsSupported)
                acc = Fma.MultiplyAdd(Vector256.Create(scale),
                    Avx.ConvertToVector256Single(adjusted), acc);
            else
                acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(scale),
                    Avx.ConvertToVector256Single(adjusted)));
        }

        return HorizontalSumAvx2Float(acc);
    }

    // ──────────────────── Q5_0 × Q8_1 4-row variant ────────────────────

    /// <summary>
    /// Unpacks a single Q5_0 block and computes the unsigned integer dot product with Q8_1 values.
    /// Returns <c>sum(q5_unsigned × q8)</c> as int32×8 — the -16 offset is handled via
    /// the precomputed Q8_1 <c>s</c> field by the caller.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Q5_0BlockDotQ8_1(byte* q5Block, Vector256<sbyte> q8Vals,
        Vector256<short> ones)
    {
        uint qh = Unsafe.ReadUnaligned<uint>(q5Block + 2);

        Vector128<byte> qsRaw = Unsafe.ReadUnaligned<Vector128<byte>>(q5Block + 6);
        Vector128<byte> lo128 = Sse2.And(qsRaw, Vector128.Create((byte)0x0F));
        Vector128<byte> hi128 = Sse2.And(
            Sse2.ShiftRightLogical(qsRaw.AsUInt16(), 4).AsByte(),
            Vector128.Create((byte)0x0F));
        Vector256<byte> q5vals = Avx2.Or(Vector256.Create(lo128, hi128), ExtractQ5HighBits(qh));

        Vector256<short> prod = Avx2.MultiplyAddAdjacent(q5vals, q8Vals);
        return Avx2.MultiplyAddAdjacent(prod, ones);
    }

    /// <summary>
    /// True 4-row Q5_0 × Q8_1 dot product with shared Q8_1 loads.
    /// Loads Q8_1 data once per block and reuses across all 4 weight rows,
    /// saving 3 loads of 32 bytes per block. Uses precomputed Q8_1 <c>s</c> field
    /// for the -16 offset, eliminating 4 SIMD ops per block per row.
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ5_0Q8_1Avx2_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* q8, int blockCount, float* results)
    {
        Vector256<float> acc0 = Vector256<float>.Zero;
        Vector256<float> acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero;
        Vector256<float> acc3 = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);
        float off0 = 0, off1 = 0, off2 = 0, off3 = 0;

        for (int block = 0; block < blockCount; block++)
        {
            byte* q8Block = q8 + block * Q8_1BlockBytes;
            float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);
            float s8 = (float)Unsafe.ReadUnaligned<Half>(q8Block + 2);
            int blockOff = block * Q5_0BlockBytes;

            // Shared Q8_1 load — reused across all 4 rows
            Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8Block + 4);

            // Row 0
            float d0 = (float)Unsafe.ReadUnaligned<Half>(w0 + blockOff);
            Vector256<int> dot0 = Q5_0BlockDotQ8_1(w0 + blockOff, q8Vals, ones);
            float s0 = d0 * d8;

            // Row 1
            float d1 = (float)Unsafe.ReadUnaligned<Half>(w1 + blockOff);
            Vector256<int> dot1 = Q5_0BlockDotQ8_1(w1 + blockOff, q8Vals, ones);
            float s1 = d1 * d8;

            // Row 2
            float d2 = (float)Unsafe.ReadUnaligned<Half>(w2 + blockOff);
            Vector256<int> dot2 = Q5_0BlockDotQ8_1(w2 + blockOff, q8Vals, ones);
            float s2 = d2 * d8;

            // Row 3
            float d3 = (float)Unsafe.ReadUnaligned<Half>(w3 + blockOff);
            Vector256<int> dot3 = Q5_0BlockDotQ8_1(w3 + blockOff, q8Vals, ones);
            float s3 = d3 * d8;

            // FMA accumulate
            if (Fma.IsSupported)
            {
                acc0 = Fma.MultiplyAdd(Vector256.Create(s0), Avx.ConvertToVector256Single(dot0), acc0);
                acc1 = Fma.MultiplyAdd(Vector256.Create(s1), Avx.ConvertToVector256Single(dot1), acc1);
                acc2 = Fma.MultiplyAdd(Vector256.Create(s2), Avx.ConvertToVector256Single(dot2), acc2);
                acc3 = Fma.MultiplyAdd(Vector256.Create(s3), Avx.ConvertToVector256Single(dot3), acc3);
            }
            else
            {
                acc0 = Avx.Add(acc0, Avx.Multiply(Vector256.Create(s0), Avx.ConvertToVector256Single(dot0)));
                acc1 = Avx.Add(acc1, Avx.Multiply(Vector256.Create(s1), Avx.ConvertToVector256Single(dot1)));
                acc2 = Avx.Add(acc2, Avx.Multiply(Vector256.Create(s2), Avx.ConvertToVector256Single(dot2)));
                acc3 = Avx.Add(acc3, Avx.Multiply(Vector256.Create(s3), Avx.ConvertToVector256Single(dot3)));
            }

            // Accumulate offset sums: d5 * s8
            off0 += d0 * s8;
            off1 += d1 * s8;
            off2 += d2 * s8;
            off3 += d3 * s8;
        }

        results[0] = HorizontalSumAvx2Float(acc0) - 16.0f * off0;
        results[1] = HorizontalSumAvx2Float(acc1) - 16.0f * off1;
        results[2] = HorizontalSumAvx2Float(acc2) - 16.0f * off2;
        results[3] = HorizontalSumAvx2Float(acc3) - 16.0f * off3;
    }

    // ──────────────────── ComputeRows for Q5_0 ────────────────────

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ5_0(byte* weights, byte* xQ8, float* result, int m, int blockCount)
    {
        int rowBytes = blockCount * Q5_0BlockBytes;

        if (Avx2.IsSupported)
        {
            int row = 0;
            for (; row + 3 < m; row += 4)
            {
                VecDotQ5_0Q8_1Avx2_4Rows(
                    weights + (long)row * rowBytes,
                    weights + (long)(row + 1) * rowBytes,
                    weights + (long)(row + 2) * rowBytes,
                    weights + (long)(row + 3) * rowBytes,
                    xQ8, blockCount, result + row);
            }
            for (; row < m; row++)
                result[row] = VecDotQ5_0Q8_1Avx2(weights + (long)row * rowBytes, xQ8, blockCount);
        }
        else
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotQ5_0Q8_1Scalar(weights + (long)row * rowBytes, xQ8, blockCount);
        }
    }

    // ──────────────────── R4 Interleaved VecDot + ComputeRows for Q5_0 ────────────────────

    /// <summary>
    /// Scalar Q5_0 × Q8_1 dot product for a single row within an R4-interleaved group.
    /// Block stride is 4 * Q5_0BlockBytes.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_0Q8_1ScalarR4(byte* groupBase, int rowInGroup, byte* q8, int blockCount)
    {
        float sumf = 0;
        float offsetSum = 0;
        const int wStride = 4 * Q5_0BlockBytes;

        for (int block = 0; block < blockCount; block++)
        {
            byte* q5Block = groupBase + block * wStride + rowInGroup * Q5_0BlockBytes;
            byte* q8Block = q8 + block * Q8_1BlockBytes;

            float d5 = (float)Unsafe.ReadUnaligned<Half>(q5Block);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);
            float s8 = (float)Unsafe.ReadUnaligned<Half>(q8Block + 2);

            uint qh = Unsafe.ReadUnaligned<uint>(q5Block + 2);
            byte* qs = q5Block + 6;
            sbyte* q8v = (sbyte*)(q8Block + 4);

            int sumi = 0;
            for (int j = 0; j < 16; j++)
            {
                int x0 = (qs[j] & 0xF) | (((int)((qh >> j) & 1)) << 4);
                int x1 = ((qs[j] >> 4) & 0xF) | (((int)((qh >> (j + 16)) & 1)) << 4);
                sumi += x0 * q8v[j] + x1 * q8v[j + 16];
            }

            sumf += d5 * d8 * sumi;
            offsetSum += d5 * s8;
        }

        return sumf - 16.0f * offsetSum;
    }

    /// <summary>
    /// AVX2 4-row Q5_0 × Q8_1 dot product for R4-interleaved layout.
    /// Block stride is 4 * Q5_0BlockBytes. Uses shared Q8_1 loads and precomputed block sums.
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ5_0Q8_1Avx2_4RowsR4(
        byte* groupBase, byte* q8, int blockCount, float* results)
    {
        Vector256<float> acc0 = Vector256<float>.Zero;
        Vector256<float> acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero;
        Vector256<float> acc3 = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);
        float off0 = 0, off1 = 0, off2 = 0, off3 = 0;
        const int wStride = 4 * Q5_0BlockBytes;

        for (int block = 0; block < blockCount; block++)
        {
            byte* q8Block = q8 + block * Q8_1BlockBytes;
            float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);
            float s8 = (float)Unsafe.ReadUnaligned<Half>(q8Block + 2);

            // Shared Q8_1 load
            Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8Block + 4);
            byte* blockBase = groupBase + block * wStride;

            // Row 0
            {
                byte* w = blockBase;
                float d5 = (float)Unsafe.ReadUnaligned<Half>(w);
                Vector256<int> dot = Q5_0BlockDotQ8_1(w, q8Vals, ones);
                if (Fma.IsSupported)
                    acc0 = Fma.MultiplyAdd(Vector256.Create(d5 * d8), Avx.ConvertToVector256Single(dot), acc0);
                else
                    acc0 = Avx.Add(acc0, Avx.Multiply(Vector256.Create(d5 * d8), Avx.ConvertToVector256Single(dot)));
                off0 += d5 * s8;
            }

            // Row 1
            {
                byte* w = blockBase + Q5_0BlockBytes;
                float d5 = (float)Unsafe.ReadUnaligned<Half>(w);
                Vector256<int> dot = Q5_0BlockDotQ8_1(w, q8Vals, ones);
                if (Fma.IsSupported)
                    acc1 = Fma.MultiplyAdd(Vector256.Create(d5 * d8), Avx.ConvertToVector256Single(dot), acc1);
                else
                    acc1 = Avx.Add(acc1, Avx.Multiply(Vector256.Create(d5 * d8), Avx.ConvertToVector256Single(dot)));
                off1 += d5 * s8;
            }

            // Row 2
            {
                byte* w = blockBase + 2 * Q5_0BlockBytes;
                float d5 = (float)Unsafe.ReadUnaligned<Half>(w);
                Vector256<int> dot = Q5_0BlockDotQ8_1(w, q8Vals, ones);
                if (Fma.IsSupported)
                    acc2 = Fma.MultiplyAdd(Vector256.Create(d5 * d8), Avx.ConvertToVector256Single(dot), acc2);
                else
                    acc2 = Avx.Add(acc2, Avx.Multiply(Vector256.Create(d5 * d8), Avx.ConvertToVector256Single(dot)));
                off2 += d5 * s8;
            }

            // Row 3
            {
                byte* w = blockBase + 3 * Q5_0BlockBytes;
                float d5 = (float)Unsafe.ReadUnaligned<Half>(w);
                Vector256<int> dot = Q5_0BlockDotQ8_1(w, q8Vals, ones);
                if (Fma.IsSupported)
                    acc3 = Fma.MultiplyAdd(Vector256.Create(d5 * d8), Avx.ConvertToVector256Single(dot), acc3);
                else
                    acc3 = Avx.Add(acc3, Avx.Multiply(Vector256.Create(d5 * d8), Avx.ConvertToVector256Single(dot)));
                off3 += d5 * s8;
            }
        }

        results[0] = HorizontalSumAvx2Float(acc0) - 16.0f * off0;
        results[1] = HorizontalSumAvx2Float(acc1) - 16.0f * off1;
        results[2] = HorizontalSumAvx2Float(acc2) - 16.0f * off2;
        results[3] = HorizontalSumAvx2Float(acc3) - 16.0f * off3;
    }

    /// <summary>
    /// Processes R4-interleaved Q5_0 weights. Uses R4-aware block stride for optimal cache access.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ5_0Interleaved(byte* repackedWeights, byte* xQ8, float* result,
        int fullGroups, int tailRows, int blockCount)
    {
        int groupBytes = 4 * blockCount * Q5_0BlockBytes;

        if (Avx2.IsSupported)
        {
            for (int g = 0; g < fullGroups; g++)
            {
                byte* groupBase = repackedWeights + (long)g * groupBytes;
                VecDotQ5_0Q8_1Avx2_4RowsR4(groupBase, xQ8, blockCount, result + g * 4);
            }
        }
        else
        {
            for (int g = 0; g < fullGroups; g++)
            {
                byte* groupBase = repackedWeights + (long)g * groupBytes;
                for (int r = 0; r < 4; r++)
                    result[g * 4 + r] = VecDotQ5_0Q8_1ScalarR4(groupBase, r, xQ8, blockCount);
            }
        }

        // Tail rows (row-major)
        if (tailRows > 0)
        {
            int rowBytes = blockCount * Q5_0BlockBytes;
            byte* tailBase = repackedWeights + (long)fullGroups * groupBytes;
            for (int r = 0; r < tailRows; r++)
                result[fullGroups * 4 + r] = Avx2.IsSupported
                    ? VecDotQ5_0Q8_1Avx2(tailBase + (long)r * rowBytes, xQ8, blockCount)
                    : VecDotQ5_0Q8_1Scalar(tailBase + (long)r * rowBytes, xQ8, blockCount);
        }
    }

    /// <summary>
    /// Parallel R4-interleaved Q5_0 ComputeRows. Partitions groups across threads.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ5_0Interleaved(byte* repackedWeights, byte* xQ8, float* result,
        int fullGroups, int tailRows, int blockCount, ComputeThreadPool? pool)
    {
        int m = fullGroups * 4 + tailRows;
        if (pool is null || m < ParallelMinRows)
        {
            ComputeRowsQ5_0Interleaved(repackedWeights, xQ8, result, fullGroups, tailRows, blockCount);
            return;
        }

        var ctx = new ComputeRowsR4Ctx
        {
            RepackedWeights = repackedWeights, XQ = xQ8, Result = result,
            M = m, FullGroups = fullGroups, TailRows = tailRows,
            BlockCount = blockCount, BlockBytes = Q5_0BlockBytes
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsQ5_0R4Worker);
    }

    // ──────────────────── GemvQ5_0 ────────────────────

    /// <summary>
    /// Q5_0 GEMV: weights[M,K] in Q5_0 × f32 input[K] → f32 output[M].
    /// Quantizes input to Q8_1, then uses fused Q5_0 × Q8_1 vec_dot.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ5_0(byte* weights, float* x, float* result, int m, int k)
    {
        if (k % Q5_0GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Q5_0GroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_1GroupSize;
        int xQ8Bytes = blockCount * Q8_1BlockBytes;

        byte[]? rented = null;
        byte* xQ8;

        if (xQ8Bytes <= StackAllocThreshold)
        {
            byte* stackBuf = stackalloc byte[xQ8Bytes];
            xQ8 = stackBuf;
        }
        else
        {
            rented = ArrayPool<byte>.Shared.Rent(xQ8Bytes);
            fixed (byte* rentedPtr = rented)
            {
                xQ8 = rentedPtr;
                QuantizeF32ToQ8_1(x, xQ8, k);
                ComputeRowsQ5_0(weights, xQ8, result, m, blockCount);
            }
            ArrayPool<byte>.Shared.Return(rented);
            return;
        }

        QuantizeF32ToQ8_1(x, xQ8, k);
        ComputeRowsQ5_0(weights, xQ8, result, m, blockCount);
    }

    // ──────────────────── GemmQ5_0 ────────────────────

    /// <summary>
    /// Q5_0 GEMM: C[N,M] = B[N,K] × A[M,K]^T where A is Q5_0 weights.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ5_0(byte* weights, float* b, float* c, int m, int k, int n,
                                byte* preQuantizedInput = null)
    {
        if (k % Q5_0GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Q5_0GroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_1GroupSize;
        int q8RowBytes = blockCount * Q8_1BlockBytes;
        int q5RowBytes = blockCount * Q5_0BlockBytes;

        if (n == 1)
        {
            if (preQuantizedInput != null)
            {
                ComputeRowsQ5_0(weights, preQuantizedInput, c, m, blockCount);
            }
            else
            {
                GemvQ5_0(weights, b, c, m, k);
            }
            return;
        }

        if (preQuantizedInput != null)
        {
            int tileM = ComputeTileM(q5RowBytes);
            for (int mStart = 0; mStart < m; mStart += tileM)
            {
                int tileRows = Math.Min(tileM, m - mStart);
                byte* tileWeights = weights + (long)mStart * q5RowBytes;
                for (int t = 0; t < n; t++)
                    ComputeRowsQ5_0(tileWeights, preQuantizedInput + t * q8RowBytes,
                        c + t * m + mStart, tileRows, blockCount);
            }
            return;
        }

        // Quantize all input rows, then tiled compute
        int totalQ8Bytes = n * q8RowBytes;
        byte[] rented = ArrayPool<byte>.Shared.Rent(totalQ8Bytes);
        fixed (byte* inputQ8 = rented)
        {
            for (int t = 0; t < n; t++)
                QuantizeF32ToQ8_1(b + t * k, inputQ8 + t * q8RowBytes, k);

            int tileM = ComputeTileM(q5RowBytes);
            for (int mStart = 0; mStart < m; mStart += tileM)
            {
                int tileRows = Math.Min(tileM, m - mStart);
                byte* tileWeights = weights + (long)mStart * q5RowBytes;
                for (int t = 0; t < n; t++)
                    ComputeRowsQ5_0(tileWeights, inputQ8 + t * q8RowBytes,
                        c + t * m + mStart, tileRows, blockCount);
            }
        }
        ArrayPool<byte>.Shared.Return(rented);
    }

    // ──────────────────── Parallel overloads for Q5_0 ────────────────────

    /// <summary>Q5_0 GEMV with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ5_0(byte* weights, float* x, float* result, int m, int k,
                                ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvQ5_0(weights, x, result, m, k);
            return;
        }

        if (k % Q5_0GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Q5_0GroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_1GroupSize;
        int xQ8Bytes = blockCount * Q8_1BlockBytes;

        byte* xQ8 = (byte*)pool.GetWorkerScratch(0, xQ8Bytes);
        QuantizeF32ToQ8_1(x, xQ8, k);

        var ctx = new ComputeRowsQ5_0Ctx
        {
            Weights = weights, XQ8 = xQ8, Result = result,
            M = m, BlockCount = blockCount
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsQ5_0Worker);
    }

    /// <summary>Q5_0 GEMM with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ5_0(byte* weights, float* b, float* c, int m, int k, int n,
                                ComputeThreadPool? pool, byte* preQuantizedInput = null)
    {
        if (pool is null)
        {
            GemmQ5_0(weights, b, c, m, k, n, preQuantizedInput);
            return;
        }

        if (k % Q5_0GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Q5_0GroupSize}, got {k}", nameof(k));

        if (n == 1)
        {
            if (preQuantizedInput != null)
            {
                int blockCount = k / Q8_1GroupSize;
                var ctx = new ComputeRowsQ5_0Ctx
                {
                    Weights = weights, XQ8 = preQuantizedInput, Result = c,
                    M = m, BlockCount = blockCount
                };
                pool.Dispatch((nint)(&ctx), &ComputeRowsQ5_0Worker);
            }
            else
            {
                GemvQ5_0(weights, b, c, m, k, pool);
            }
            return;
        }

        int q8BlockCount = k / Q8_1GroupSize;
        int q8RowBytes = q8BlockCount * Q8_1BlockBytes;
        int q5RowBytes = q8BlockCount * Q5_0BlockBytes;
        int tileM = ComputeTileM(q5RowBytes);
        int totalTiles = (m + tileM - 1) / tileM;

        if (preQuantizedInput != null)
        {
            if (totalTiles < 2)
            {
                GemmQ5_0(weights, b, c, m, k, n, preQuantizedInput);
                return;
            }

            var ctx = new GemmTiledQ5_0Ctx
            {
                Weights = weights, InputQ8 = preQuantizedInput, C = c,
                M = m, N = n, BlockCount = q8BlockCount,
                TileM = tileM, Q5RowBytes = q5RowBytes, Q8RowBytes = q8RowBytes
            };
            pool.Dispatch((nint)(&ctx), &GemmTiledQ5_0Worker);
            return;
        }

        // Quantize all input rows, then parallel tiled compute
        int totalQ8Bytes = n * q8RowBytes;
        byte[] rented = ArrayPool<byte>.Shared.Rent(totalQ8Bytes);
        fixed (byte* inputQ8 = rented)
        {
            for (int t = 0; t < n; t++)
                QuantizeF32ToQ8_1(b + t * k, inputQ8 + t * q8RowBytes, k);

            if (totalTiles < 2)
            {
                for (int mStart = 0; mStart < m; mStart += tileM)
                {
                    int tileRows = Math.Min(tileM, m - mStart);
                    byte* tileWeights = weights + (long)mStart * q5RowBytes;
                    for (int t = 0; t < n; t++)
                        ComputeRowsQ5_0(tileWeights, inputQ8 + t * q8RowBytes,
                            c + t * m + mStart, tileRows, q8BlockCount);
                }
            }
            else
            {
                var ctx = new GemmTiledQ5_0Ctx
                {
                    Weights = weights, InputQ8 = inputQ8, C = c,
                    M = m, N = n, BlockCount = q8BlockCount,
                    TileM = tileM, Q5RowBytes = q5RowBytes, Q8RowBytes = q8RowBytes
                };
                pool.Dispatch((nint)(&ctx), &GemmTiledQ5_0Worker);
            }
        }
        ArrayPool<byte>.Shared.Return(rented);
    }

    // ──────────────────── Q5_0 context structs and workers ────────────────────

    private struct ComputeRowsQ5_0Ctx
    {
        public byte* Weights;
        public byte* XQ8;
        public float* Result;
        public int M;
        public int BlockCount;
    }

    private struct GemmTiledQ5_0Ctx
    {
        public byte* Weights;
        public byte* InputQ8;
        public float* C;
        public int M;
        public int N;
        public int BlockCount;
        public int TileM;
        public int Q5RowBytes;
        public int Q8RowBytes;
    }

    private static void ComputeRowsQ5_0Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<ComputeRowsQ5_0Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        int rowBytes = ctx.BlockCount * Q5_0BlockBytes;
        ComputeRowsQ5_0(ctx.Weights + (long)start * rowBytes, ctx.XQ8,
            ctx.Result + start, count, ctx.BlockCount);
    }

    private static void ComputeRowsQ5_0R4Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<ComputeRowsR4Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;

        int groupBytes = 4 * ctx.BlockCount * Q5_0BlockBytes;
        int rowBytes = ctx.BlockCount * Q5_0BlockBytes;
        int end = start + count;

        int startGroup = start / 4;
        int endGroup = Math.Min(end / 4, ctx.FullGroups);
        for (int g = startGroup; g < endGroup; g++)
        {
            byte* groupBase = ctx.RepackedWeights + (long)g * groupBytes;
            if (Avx2.IsSupported)
                VecDotQ5_0Q8_1Avx2_4RowsR4(groupBase, ctx.XQ, ctx.BlockCount, ctx.Result + g * 4);
            else
                for (int r = 0; r < 4; r++)
                    ctx.Result[g * 4 + r] = VecDotQ5_0Q8_1ScalarR4(groupBase, r, ctx.XQ, ctx.BlockCount);
        }

        if (ctx.TailRows > 0 && end > ctx.FullGroups * 4)
        {
            int tailStart = Math.Max(start, ctx.FullGroups * 4) - ctx.FullGroups * 4;
            int tailEnd = Math.Min(end, ctx.M) - ctx.FullGroups * 4;
            byte* tailBase = ctx.RepackedWeights + (long)ctx.FullGroups * groupBytes;
            for (int r = tailStart; r < tailEnd; r++)
                ctx.Result[ctx.FullGroups * 4 + r] = Avx2.IsSupported
                    ? VecDotQ5_0Q8_1Avx2(tailBase + (long)r * rowBytes, ctx.XQ, ctx.BlockCount)
                    : VecDotQ5_0Q8_1Scalar(tailBase + (long)r * rowBytes, ctx.XQ, ctx.BlockCount);
        }
    }

    private static void GemmTiledQ5_0Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmTiledQ5_0Ctx>((void*)ctxPtr);
        int totalTiles = (ctx.M + ctx.TileM - 1) / ctx.TileM;
        int tilesPerThread = (totalTiles + threadCount - 1) / threadCount;
        int startTile = threadIdx * tilesPerThread;
        int endTile = Math.Min(startTile + tilesPerThread, totalTiles);

        for (int tile = startTile; tile < endTile; tile++)
        {
            int mStart = tile * ctx.TileM;
            int tileRows = Math.Min(ctx.TileM, ctx.M - mStart);
            byte* tileWeights = ctx.Weights + (long)mStart * ctx.Q5RowBytes;
            for (int t = 0; t < ctx.N; t++)
                ComputeRowsQ5_0(tileWeights, ctx.InputQ8 + t * ctx.Q8RowBytes,
                    ctx.C + t * ctx.M + mStart, tileRows, ctx.BlockCount);
        }
    }

    // ──────────────────── Q5_0 Outer-product GEMM (R4 layout) ────────────────────

    /// <summary>
    /// AVX2 outer-product microkernel for Q5_0 R4 layout.
    /// Processes 4 weight rows × 3 tokens with 12 YMM accumulators + 12 scalar offsets.
    /// Weight block is loaded once and Q5_0BlockDotQ8_1 reused across 3 tokens (3× cache reuse).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void OuterProductQ5_0Avx2_4x3(
        byte* groupBase, byte* x0, byte* x1, byte* x2,
        float* c, int blockCount, int cStride)
    {
        const int wStride = 4 * Q5_0BlockBytes;

        Vector256<float> acc00 = Vector256<float>.Zero, acc01 = Vector256<float>.Zero, acc02 = Vector256<float>.Zero;
        Vector256<float> acc10 = Vector256<float>.Zero, acc11 = Vector256<float>.Zero, acc12 = Vector256<float>.Zero;
        Vector256<float> acc20 = Vector256<float>.Zero, acc21 = Vector256<float>.Zero, acc22 = Vector256<float>.Zero;
        Vector256<float> acc30 = Vector256<float>.Zero, acc31 = Vector256<float>.Zero, acc32 = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);
        // Scalar offsets for -16 correction: off[row][token]
        float off00 = 0, off01 = 0, off02 = 0;
        float off10 = 0, off11 = 0, off12 = 0;
        float off20 = 0, off21 = 0, off22 = 0;
        float off30 = 0, off31 = 0, off32 = 0;

        for (int b = 0; b < blockCount; b++)
        {
            byte* blockBase = groupBase + b * wStride;

            // Load 3 Q8_1 token blocks (shared across weight rows)
            byte* q8b0 = x0 + b * Q8_1BlockBytes;
            byte* q8b1 = x1 + b * Q8_1BlockBytes;
            byte* q8b2 = x2 + b * Q8_1BlockBytes;
            float d8_0 = (float)Unsafe.ReadUnaligned<Half>(q8b0);
            float d8_1 = (float)Unsafe.ReadUnaligned<Half>(q8b1);
            float d8_2 = (float)Unsafe.ReadUnaligned<Half>(q8b2);
            float s8_0 = (float)Unsafe.ReadUnaligned<Half>(q8b0 + 2);
            float s8_1 = (float)Unsafe.ReadUnaligned<Half>(q8b1 + 2);
            float s8_2 = (float)Unsafe.ReadUnaligned<Half>(q8b2 + 2);
            Vector256<sbyte> q8v0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8b0 + 4);
            Vector256<sbyte> q8v1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8b1 + 4);
            Vector256<sbyte> q8v2 = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8b2 + 4);

            // Row 0: load weight once, compute for 3 tokens
            {
                byte* w = blockBase;
                float d5 = (float)Unsafe.ReadUnaligned<Half>(w);

                Vector256<int> dot0 = Q5_0BlockDotQ8_1(w, q8v0, ones);
                if (Fma.IsSupported)
                    acc00 = Fma.MultiplyAdd(Vector256.Create(d5 * d8_0), Avx.ConvertToVector256Single(dot0), acc00);
                else
                    acc00 += Avx.ConvertToVector256Single(dot0) * Vector256.Create(d5 * d8_0);
                off00 += d5 * s8_0;

                Vector256<int> dot1 = Q5_0BlockDotQ8_1(w, q8v1, ones);
                if (Fma.IsSupported)
                    acc01 = Fma.MultiplyAdd(Vector256.Create(d5 * d8_1), Avx.ConvertToVector256Single(dot1), acc01);
                else
                    acc01 += Avx.ConvertToVector256Single(dot1) * Vector256.Create(d5 * d8_1);
                off01 += d5 * s8_1;

                Vector256<int> dot2 = Q5_0BlockDotQ8_1(w, q8v2, ones);
                if (Fma.IsSupported)
                    acc02 = Fma.MultiplyAdd(Vector256.Create(d5 * d8_2), Avx.ConvertToVector256Single(dot2), acc02);
                else
                    acc02 += Avx.ConvertToVector256Single(dot2) * Vector256.Create(d5 * d8_2);
                off02 += d5 * s8_2;
            }

            // Row 1
            {
                byte* w = blockBase + Q5_0BlockBytes;
                float d5 = (float)Unsafe.ReadUnaligned<Half>(w);

                Vector256<int> dot0 = Q5_0BlockDotQ8_1(w, q8v0, ones);
                acc10 = Fma.IsSupported ? Fma.MultiplyAdd(Vector256.Create(d5 * d8_0), Avx.ConvertToVector256Single(dot0), acc10)
                    : acc10 + Avx.ConvertToVector256Single(dot0) * Vector256.Create(d5 * d8_0);
                off10 += d5 * s8_0;

                Vector256<int> dot1 = Q5_0BlockDotQ8_1(w, q8v1, ones);
                acc11 = Fma.IsSupported ? Fma.MultiplyAdd(Vector256.Create(d5 * d8_1), Avx.ConvertToVector256Single(dot1), acc11)
                    : acc11 + Avx.ConvertToVector256Single(dot1) * Vector256.Create(d5 * d8_1);
                off11 += d5 * s8_1;

                Vector256<int> dot2 = Q5_0BlockDotQ8_1(w, q8v2, ones);
                acc12 = Fma.IsSupported ? Fma.MultiplyAdd(Vector256.Create(d5 * d8_2), Avx.ConvertToVector256Single(dot2), acc12)
                    : acc12 + Avx.ConvertToVector256Single(dot2) * Vector256.Create(d5 * d8_2);
                off12 += d5 * s8_2;
            }

            // Row 2
            {
                byte* w = blockBase + 2 * Q5_0BlockBytes;
                float d5 = (float)Unsafe.ReadUnaligned<Half>(w);

                Vector256<int> dot0 = Q5_0BlockDotQ8_1(w, q8v0, ones);
                acc20 = Fma.IsSupported ? Fma.MultiplyAdd(Vector256.Create(d5 * d8_0), Avx.ConvertToVector256Single(dot0), acc20)
                    : acc20 + Avx.ConvertToVector256Single(dot0) * Vector256.Create(d5 * d8_0);
                off20 += d5 * s8_0;

                Vector256<int> dot1 = Q5_0BlockDotQ8_1(w, q8v1, ones);
                acc21 = Fma.IsSupported ? Fma.MultiplyAdd(Vector256.Create(d5 * d8_1), Avx.ConvertToVector256Single(dot1), acc21)
                    : acc21 + Avx.ConvertToVector256Single(dot1) * Vector256.Create(d5 * d8_1);
                off21 += d5 * s8_1;

                Vector256<int> dot2 = Q5_0BlockDotQ8_1(w, q8v2, ones);
                acc22 = Fma.IsSupported ? Fma.MultiplyAdd(Vector256.Create(d5 * d8_2), Avx.ConvertToVector256Single(dot2), acc22)
                    : acc22 + Avx.ConvertToVector256Single(dot2) * Vector256.Create(d5 * d8_2);
                off22 += d5 * s8_2;
            }

            // Row 3
            {
                byte* w = blockBase + 3 * Q5_0BlockBytes;
                float d5 = (float)Unsafe.ReadUnaligned<Half>(w);

                Vector256<int> dot0 = Q5_0BlockDotQ8_1(w, q8v0, ones);
                acc30 = Fma.IsSupported ? Fma.MultiplyAdd(Vector256.Create(d5 * d8_0), Avx.ConvertToVector256Single(dot0), acc30)
                    : acc30 + Avx.ConvertToVector256Single(dot0) * Vector256.Create(d5 * d8_0);
                off30 += d5 * s8_0;

                Vector256<int> dot1 = Q5_0BlockDotQ8_1(w, q8v1, ones);
                acc31 = Fma.IsSupported ? Fma.MultiplyAdd(Vector256.Create(d5 * d8_1), Avx.ConvertToVector256Single(dot1), acc31)
                    : acc31 + Avx.ConvertToVector256Single(dot1) * Vector256.Create(d5 * d8_1);
                off31 += d5 * s8_1;

                Vector256<int> dot2 = Q5_0BlockDotQ8_1(w, q8v2, ones);
                acc32 = Fma.IsSupported ? Fma.MultiplyAdd(Vector256.Create(d5 * d8_2), Avx.ConvertToVector256Single(dot2), acc32)
                    : acc32 + Avx.ConvertToVector256Single(dot2) * Vector256.Create(d5 * d8_2);
                off32 += d5 * s8_2;
            }
        }

        c[0 * cStride + 0] = HorizontalSumAvx2Float(acc00) - 16.0f * off00;
        c[0 * cStride + 1] = HorizontalSumAvx2Float(acc10) - 16.0f * off10;
        c[0 * cStride + 2] = HorizontalSumAvx2Float(acc20) - 16.0f * off20;
        c[0 * cStride + 3] = HorizontalSumAvx2Float(acc30) - 16.0f * off30;
        c[1 * cStride + 0] = HorizontalSumAvx2Float(acc01) - 16.0f * off01;
        c[1 * cStride + 1] = HorizontalSumAvx2Float(acc11) - 16.0f * off11;
        c[1 * cStride + 2] = HorizontalSumAvx2Float(acc21) - 16.0f * off21;
        c[1 * cStride + 3] = HorizontalSumAvx2Float(acc31) - 16.0f * off31;
        c[2 * cStride + 0] = HorizontalSumAvx2Float(acc02) - 16.0f * off02;
        c[2 * cStride + 1] = HorizontalSumAvx2Float(acc12) - 16.0f * off12;
        c[2 * cStride + 2] = HorizontalSumAvx2Float(acc22) - 16.0f * off22;
        c[2 * cStride + 3] = HorizontalSumAvx2Float(acc32) - 16.0f * off32;
    }

    /// <summary>
    /// Outer-product GEMM for Q5_0 R4-interleaved weights with Q8_1 input.
    /// Processes weight groups in steps of 4 rows and token batches of 3.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void OuterProductGemmQ5_0(byte* repackedWeights, byte* inputQ8, float* c,
        int fullGroups, int tailRows, int blockCount, int m, int n)
    {
        int q8RowBytes = blockCount * Q8_1BlockBytes;
        int groupBytes = 4 * blockCount * Q5_0BlockBytes;
        int nFull3 = (n / 3) * 3;

        for (int g = 0; g < fullGroups; g++)
        {
            byte* groupBase = repackedWeights + (long)g * groupBytes;
            int baseRow = g * 4;
            int t = 0;

            if (Avx2.IsSupported)
            {
                for (; t < nFull3; t += 3)
                {
                    OuterProductQ5_0Avx2_4x3(
                        groupBase,
                        inputQ8 + (long)t * q8RowBytes,
                        inputQ8 + (long)(t + 1) * q8RowBytes,
                        inputQ8 + (long)(t + 2) * q8RowBytes,
                        c + (long)t * m + baseRow, blockCount, m);
                }
                for (; t < n; t++)
                {
                    VecDotQ5_0Q8_1Avx2_4RowsR4(groupBase, inputQ8 + (long)t * q8RowBytes,
                        blockCount, c + (long)t * m + baseRow);
                }
            }
            else
            {
                for (; t < n; t++)
                {
                    for (int r = 0; r < 4; r++)
                        c[(long)t * m + baseRow + r] = VecDotQ5_0Q8_1ScalarR4(
                            groupBase, r, inputQ8 + (long)t * q8RowBytes, blockCount);
                }
            }
        }

        // Tail rows
        if (tailRows > 0)
        {
            int rowBytes = blockCount * Q5_0BlockBytes;
            byte* tailBase = repackedWeights + (long)fullGroups * groupBytes;
            int baseRow = fullGroups * 4;

            for (int t = 0; t < n; t++)
            {
                byte* xQ8 = inputQ8 + (long)t * q8RowBytes;
                for (int r = 0; r < tailRows; r++)
                {
                    c[(long)t * m + baseRow + r] = Avx2.IsSupported
                        ? VecDotQ5_0Q8_1Avx2(tailBase + (long)r * rowBytes, xQ8, blockCount)
                        : VecDotQ5_0Q8_1Scalar(tailBase + (long)r * rowBytes, xQ8, blockCount);
                }
            }
        }
    }

    /// <summary>Context for parallel outer-product Q5_0 GEMM dispatch.</summary>
    private struct OuterProductGemmQ5Ctx
    {
        public byte* RepackedWeights;
        public byte* InputQ8;
        public float* C;
        public int FullGroups;
        public int TailRows;
        public int BlockCount;
        public int M;
        public int N;
    }

    /// <summary>
    /// Parallel outer-product GEMM for Q5_0 R4 weights.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void OuterProductGemmQ5_0(byte* repackedWeights, byte* inputQ8, float* c,
        int fullGroups, int tailRows, int blockCount, int m, int n, ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            OuterProductGemmQ5_0(repackedWeights, inputQ8, c, fullGroups, tailRows, blockCount, m, n);
            return;
        }

        var ctx = new OuterProductGemmQ5Ctx
        {
            RepackedWeights = repackedWeights, InputQ8 = inputQ8, C = c,
            FullGroups = fullGroups, TailRows = tailRows,
            BlockCount = blockCount, M = m, N = n
        };
        pool.Dispatch((nint)(&ctx), &OuterProductGemmQ5_0Worker);
    }

    private static void OuterProductGemmQ5_0Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<OuterProductGemmQ5Ctx>((void*)ctxPtr);

        int totalGroups = ctx.FullGroups + (ctx.TailRows > 0 ? 1 : 0);
        int groupsPerThread = (totalGroups + threadCount - 1) / threadCount;
        int startGroup = threadIdx * groupsPerThread;
        int endGroup = Math.Min(startGroup + groupsPerThread, totalGroups);

        if (startGroup >= totalGroups) return;

        int q8RowBytes = ctx.BlockCount * Q8_1BlockBytes;
        int groupBytes = 4 * ctx.BlockCount * Q5_0BlockBytes;
        int nFull3 = (ctx.N / 3) * 3;

        for (int g = startGroup; g < endGroup; g++)
        {
            if (g < ctx.FullGroups)
            {
                byte* groupBase = ctx.RepackedWeights + (long)g * groupBytes;
                int baseRow = g * 4;
                int t = 0;

                if (Avx2.IsSupported)
                {
                    for (; t < nFull3; t += 3)
                    {
                        OuterProductQ5_0Avx2_4x3(
                            groupBase,
                            ctx.InputQ8 + (long)t * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 1) * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 2) * q8RowBytes,
                            ctx.C + (long)t * ctx.M + baseRow, ctx.BlockCount, ctx.M);
                    }
                    for (; t < ctx.N; t++)
                    {
                        VecDotQ5_0Q8_1Avx2_4RowsR4(groupBase, ctx.InputQ8 + (long)t * q8RowBytes,
                            ctx.BlockCount, ctx.C + (long)t * ctx.M + baseRow);
                    }
                }
                else
                {
                    for (; t < ctx.N; t++)
                    {
                        for (int r = 0; r < 4; r++)
                            ctx.C[(long)t * ctx.M + baseRow + r] = VecDotQ5_0Q8_1ScalarR4(
                                groupBase, r, ctx.InputQ8 + (long)t * q8RowBytes, ctx.BlockCount);
                    }
                }
            }
            else
            {
                byte* tailBase = ctx.RepackedWeights + (long)ctx.FullGroups * groupBytes;
                int baseRow = ctx.FullGroups * 4;
                int rowBytes = ctx.BlockCount * Q5_0BlockBytes;
                for (int t = 0; t < ctx.N; t++)
                {
                    byte* xQ8 = ctx.InputQ8 + (long)t * q8RowBytes;
                    for (int r = 0; r < ctx.TailRows; r++)
                    {
                        ctx.C[(long)t * ctx.M + baseRow + r] = Avx2.IsSupported
                            ? VecDotQ5_0Q8_1Avx2(tailBase + (long)r * rowBytes, xQ8, ctx.BlockCount)
                            : VecDotQ5_0Q8_1Scalar(tailBase + (long)r * rowBytes, xQ8, ctx.BlockCount);
                    }
                }
            }
        }
    }
}
