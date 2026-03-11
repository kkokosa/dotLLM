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
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> ExtractQ5HighBits(uint qh)
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
}
