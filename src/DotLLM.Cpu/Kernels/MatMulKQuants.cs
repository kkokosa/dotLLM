using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// K-quant matrix multiplication kernels for Q4_K, Q5_K, and Q6_K × Q8_0 dot products.
/// One K-quant super-block (256 values) corresponds to 8 consecutive Q8_0 blocks (8×32).
/// </summary>
public static unsafe partial class MatMul
{
    private const int Q4_K_BlockBytes = 144;
    private const int Q5_K_BlockBytes = 176;
    private const int Q6_K_BlockBytes = 210;
    private const int KQuantGroupSize = 256;

    // ──────────────────── Q6_K × Q8_0 scalar ────────────────────

    /// <summary>
    /// Scalar Q6_K × Q8_0 dot product.
    /// Layout: ql[128]@0, qh[64]@128, scales[16](int8)@192, d(Half)@208.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ6_K_Q8_0Scalar(byte* qk, byte* q8, int superBlockCount)
    {
        float sumf = 0;

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            byte* ql = qk;
            byte* qh = qk + 128;
            sbyte* scales = (sbyte*)(qk + 192);
            float d = (float)Unsafe.ReadUnaligned<Half>(qk + 208);

            // 16 sub-blocks of 16 values each, matching llama.cpp element ordering.
            // Within each half (128 elements): ql lower nibbles → first 64, upper → next 64.
            // qh is indexed per-byte with 2-bit shifts per group of 64.
            for (int sub = 0; sub < 16; sub++)
            {
                float sc = d * scales[sub];

                int q8BlockIdx = sub / 2;
                int q8Offset = (sub % 2) * 16;
                byte* q8Block = q8 + q8BlockIdx * Q8_0BlockBytes;
                float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);
                sbyte* q8Vals = (sbyte*)(q8Block + 2) + q8Offset;

                int half = sub / 8;
                int sh = sub % 8;
                int qlOff = half * 64 + (sh % 4) * 16;
                bool isUpper = sh >= 4;
                int qhOff = half * 32 + (sh % 2) * 16;
                int qhShift = (sh / 2) * 2;

                int sumi = 0;
                for (int i = 0; i < 16; i++)
                {
                    int lo4 = isUpper ? ((ql[qlOff + i] >> 4) & 0xF) : (ql[qlOff + i] & 0xF);
                    int hi2 = (qh[qhOff + i] >> qhShift) & 3;
                    int q6 = (lo4 | (hi2 << 4)) - 32;

                    sumi += q6 * q8Vals[i];
                }

                sumf += sc * d8 * sumi;
            }

            qk += Q6_K_BlockBytes;
            q8 += 8 * Q8_0BlockBytes; // 8 Q8_0 blocks per K-quant super-block
        }

        return sumf;
    }

    // ──────────────────── Q4_K × Q8_0 scalar ────────────────────

    /// <summary>
    /// Scalar Q4_K × Q8_0 dot product.
    /// Layout: d(Half@0), dmin(Half@2), scales[12]@4, qs[128]@16.
    /// Key optimization from llama.cpp: sum += d8 * (d * scale * prod_sum - dmin * min * q8_sum)
    /// where q8_sum is the raw sum of Q8_0 values (not the product sum).
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ4_K_Q8_0Scalar(byte* qk, byte* q8, int superBlockCount)
    {
        float sumf = 0;

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(qk);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(qk + 2);
            Dequantize.UnpackQ4Q5Scales(qk + 4, scBuf, mnBuf);
            byte* qs = qk + 16;

            // 8 sub-blocks of 32 values, matching llama.cpp non-interleaved nibble ordering.
            // Within each pair: first 32 elements use lower nibbles, next 32 use upper nibbles.
            for (int j = 0; j < 8; j++)
            {
                float sc = d * scBuf[j];
                float mn = dmin * mnBuf[j];

                byte* q8Block = q8 + j * Q8_0BlockBytes;
                float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);
                sbyte* q8Vals = (sbyte*)(q8Block + 2);

                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                int prodSum = 0;
                int q8Sum = 0;

                for (int i = 0; i < 32; i++)
                {
                    int qsByte = pairIdx * 32 + i;
                    int nibble = nibbleHalf == 0 ? (qs[qsByte] & 0xF) : (qs[qsByte] >> 4);

                    prodSum += nibble * q8Vals[i];
                    q8Sum += q8Vals[i];
                }

                sumf += d8 * (sc * prodSum - mn * q8Sum);
            }

            qk += Q4_K_BlockBytes;
            q8 += 8 * Q8_0BlockBytes;
        }

        return sumf;
    }

    // ──────────────────── Q5_K × Q8_0 scalar ────────────────────

    /// <summary>
    /// Scalar Q5_K × Q8_0 dot product.
    /// Layout: d(Half@0), dmin(Half@2), scales[12]@4, qh[32]@16, qs[128]@48.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_K_Q8_0Scalar(byte* qk, byte* q8, int superBlockCount)
    {
        float sumf = 0;

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(qk);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(qk + 2);
            Dequantize.UnpackQ4Q5Scales(qk + 4, scBuf, mnBuf);
            byte* qh = qk + 16;
            byte* qs = qk + 48;

            // 8 sub-blocks of 32 values, matching llama.cpp.
            // Nibbles: non-interleaved (lower first, then upper per pair).
            // qh: bit j from byte i (qh[i] >> j), NOT flat bitfield.
            for (int j = 0; j < 8; j++)
            {
                float sc = d * scBuf[j];
                float mn = dmin * mnBuf[j];

                byte* q8Block = q8 + j * Q8_0BlockBytes;
                float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);
                sbyte* q8Vals = (sbyte*)(q8Block + 2);

                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                int prodSum = 0;
                int q8Sum = 0;

                for (int i = 0; i < 32; i++)
                {
                    int qsByte = pairIdx * 32 + i;
                    int lo4 = nibbleHalf == 0 ? (qs[qsByte] & 0xF) : (qs[qsByte] >> 4);
                    int bit5 = (qh[i] >> j) & 1;

                    int q = lo4 | (bit5 << 4);

                    prodSum += q * q8Vals[i];
                    q8Sum += q8Vals[i];
                }

                sumf += d8 * (sc * prodSum - mn * q8Sum);
            }

            qk += Q5_K_BlockBytes;
            q8 += 8 * Q8_0BlockBytes;
        }

        return sumf;
    }

    // ──────────────────── AVX2 vec_dot kernels ────────────────────

    /// <summary>
    /// AVX2 Q4_K × Q8_0 dot product. Processes one sub-block (32 values) at a time.
    /// Uses vpmaddubsw for unsigned×signed multiply-accumulate.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ4_K_Q8_0Avx2(byte* qk, byte* q8, int superBlockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        Vector256<short> ones = Vector256.Create((short)1);

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(qk);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(qk + 2);
            Dequantize.UnpackQ4Q5Scales(qk + 4, scBuf, mnBuf);
            byte* qs = qk + 16;

            // 8 sub-blocks of 32 values, matching llama.cpp non-interleaved ordering.
            // Load 32 qs bytes per pair, use lower nibbles for even sub-block,
            // upper nibbles for odd sub-block.
            for (int j = 0; j < 8; j++)
            {
                byte* q8Block = q8 + j * Q8_0BlockBytes;
                float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);
                if (d8 == 0) continue;

                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                // Load 32 bytes of qs for this pair
                Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs + pairIdx * 32);
                Vector256<byte> nibbles;
                if (nibbleHalf == 0)
                    nibbles = Avx2.And(raw, mask0F);
                else
                    nibbles = Avx2.And(
                        Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);

                // Load 32 Q8_0 values (signed)
                Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8Block + 2);

                // Product sum: unsigned nibbles × signed q8 → int16 pairs → int32
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(nibbles, q8Vals);
                Vector256<int> prodSum = Avx2.MultiplyAddAdjacent(prod, ones);

                // Q8 raw sum
                Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(
                    Vector256.Create((byte)1), q8Vals);
                Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);

                float sc = d * scBuf[j];
                float mn = dmin * mnBuf[j];

                Vector256<float> fProd = Avx.ConvertToVector256Single(prodSum);
                Vector256<float> fQ8 = Avx.ConvertToVector256Single(q8Sum);

                Vector256<float> term = Avx.Subtract(
                    Avx.Multiply(Vector256.Create(sc), fProd),
                    Avx.Multiply(Vector256.Create(mn), fQ8));

                acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(d8), term));
            }

            qk += Q4_K_BlockBytes;
            q8 += 8 * Q8_0BlockBytes;
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>
    /// AVX2 Q6_K × Q8_0 dot product.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ6_K_Q8_0Avx2(byte* qk, byte* q8, int superBlockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        Vector256<byte> mask03 = Vector256.Create((byte)0x03);
        Vector256<short> ones = Vector256.Create((short)1);

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            byte* ql = qk;
            byte* qh = qk + 128;
            sbyte* scales = (sbyte*)(qk + 192);
            float d = (float)Unsafe.ReadUnaligned<Half>(qk + 208);

            // 16 sub-blocks of 16, processed in pairs (32 values = 1 Q8_0 block).
            // Matching llama.cpp element ordering: within each half (128 elements),
            // ql lower nibbles → first 64 elements, upper → next 64.
            for (int sub = 0; sub < 16; sub += 2)
            {
                int q8BlockIdx0 = sub / 2;
                byte* q8Block0 = q8 + q8BlockIdx0 * Q8_0BlockBytes;
                float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block0);

                float sc0 = d * scales[sub];
                float sc1 = d * scales[sub + 1];

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

                // Combine: q6_unsigned = nibble | (qh2 << 4), mask to 6 bits
                Vector256<byte> q6u = Avx2.And(
                    Avx2.Or(nibbles, Avx2.ShiftLeftLogical(qhBits.AsUInt16(), 4).AsByte()),
                    Vector256.Create((byte)0x3F));

                // Load 32 Q8_0 values
                Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8Block0 + 2);

                // Unsigned × signed MAD: q6u * q8
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(q6u, q8Vals);
                Vector256<int> prodSum = Avx2.MultiplyAddAdjacent(prod, ones);

                // Sum of q8 values (for the -32 bias correction)
                Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), q8Vals);
                Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);

                // Per-lane scale: lower 128 bits → sc0, upper → sc1
                Vector128<float> prodLo = Avx.ConvertToVector128Single(prodSum.GetLower());
                Vector128<float> prodHi = Avx.ConvertToVector128Single(prodSum.GetUpper());
                Vector128<float> q8Lo = Avx.ConvertToVector128Single(q8Sum.GetLower());
                Vector128<float> q8Hi = Avx.ConvertToVector128Single(q8Sum.GetUpper());

                // term = sc * (prod - 32 * q8sum)
                Vector128<float> bias = Vector128.Create(32f);
                Vector128<float> termLo = Sse.Multiply(Vector128.Create(sc0),
                    Sse.Subtract(prodLo, Sse.Multiply(bias, q8Lo)));
                Vector128<float> termHi = Sse.Multiply(Vector128.Create(sc1),
                    Sse.Subtract(prodHi, Sse.Multiply(bias, q8Hi)));

                Vector256<float> term = Vector256.Create(termLo, termHi);
                acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(d8), term));
            }

            qk += Q6_K_BlockBytes;
            q8 += 8 * Q8_0BlockBytes;
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>
    /// AVX2 Q5_K × Q8_0 dot product.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_K_Q8_0Avx2(byte* qk, byte* q8, int superBlockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        Vector256<short> ones = Vector256.Create((short)1);

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(qk);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(qk + 2);
            Dequantize.UnpackQ4Q5Scales(qk + 4, scBuf, mnBuf);
            byte* qh = qk + 16;
            byte* qs = qk + 48;

            // 8 sub-blocks of 32 values, matching llama.cpp.
            // Nibbles: non-interleaved (lower first, then upper per pair).
            // qh: bit j from each of the 32 qh bytes.
            for (int j = 0; j < 8; j++)
            {
                byte* q8Block = q8 + j * Q8_0BlockBytes;
                float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);
                if (d8 == 0) continue;

                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                // Load 32 bytes of qs for this pair
                Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs + pairIdx * 32);
                Vector256<byte> nibbles;
                if (nibbleHalf == 0)
                    nibbles = Avx2.And(raw, mask0F);
                else
                    nibbles = Avx2.And(
                        Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);

                // Extract 5th bits from qh: bit j from each of the 32 bytes
                Vector256<byte> qhVec = Unsafe.ReadUnaligned<Vector256<byte>>(qh);
                byte bitMask = (byte)(1 << j);
                Vector256<byte> hasBit = Avx2.CompareEqual(
                    Avx2.And(qhVec, Vector256.Create(bitMask)), Vector256.Create(bitMask));
                Vector256<byte> bit5Vec = Avx2.And(hasBit, Vector256.Create((byte)16));

                Vector256<byte> q5vals = Avx2.Or(nibbles, bit5Vec);

                // Load 32 Q8_0 values
                Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8Block + 2);

                // Product sum: unsigned q5 × signed q8
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(q5vals, q8Vals);
                Vector256<int> prodSum = Avx2.MultiplyAddAdjacent(prod, ones);

                // Q8 raw sum
                Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), q8Vals);
                Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);

                float sc = d * scBuf[j];
                float mn = dmin * mnBuf[j];

                Vector256<float> fProd = Avx.ConvertToVector256Single(prodSum);
                Vector256<float> fQ8 = Avx.ConvertToVector256Single(q8Sum);

                Vector256<float> term = Avx.Subtract(
                    Avx.Multiply(Vector256.Create(sc), fProd),
                    Avx.Multiply(Vector256.Create(mn), fQ8));

                acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(d8), term));
            }

            qk += Q5_K_BlockBytes;
            q8 += 8 * Q8_0BlockBytes;
        }

        return HorizontalSumAvx2Float(acc);
    }

    // ──────────────────── AVX2 4-row variants ────────────────────

    [SkipLocalsInit]
    internal static void VecDotQ4_K_Q8_0Avx2_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* q8, int superBlockCount, float* results)
    {
        float r0 = 0, r1 = 0, r2 = 0, r3 = 0;
        // Delegate to single-row for now — the 4-row optimization provides marginal
        // benefit for K-quants since the bottleneck is scale unpacking, not Q8 loading.
        // Can be optimized further if profiling shows this is hot.
        r0 = Avx2.IsSupported
            ? VecDotQ4_K_Q8_0Avx2(w0, q8, superBlockCount)
            : VecDotQ4_K_Q8_0Scalar(w0, q8, superBlockCount);
        r1 = Avx2.IsSupported
            ? VecDotQ4_K_Q8_0Avx2(w1, q8, superBlockCount)
            : VecDotQ4_K_Q8_0Scalar(w1, q8, superBlockCount);
        r2 = Avx2.IsSupported
            ? VecDotQ4_K_Q8_0Avx2(w2, q8, superBlockCount)
            : VecDotQ4_K_Q8_0Scalar(w2, q8, superBlockCount);
        r3 = Avx2.IsSupported
            ? VecDotQ4_K_Q8_0Avx2(w3, q8, superBlockCount)
            : VecDotQ4_K_Q8_0Scalar(w3, q8, superBlockCount);
        results[0] = r0;
        results[1] = r1;
        results[2] = r2;
        results[3] = r3;
    }

    [SkipLocalsInit]
    internal static void VecDotQ5_K_Q8_0Avx2_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* q8, int superBlockCount, float* results)
    {
        results[0] = Avx2.IsSupported
            ? VecDotQ5_K_Q8_0Avx2(w0, q8, superBlockCount)
            : VecDotQ5_K_Q8_0Scalar(w0, q8, superBlockCount);
        results[1] = Avx2.IsSupported
            ? VecDotQ5_K_Q8_0Avx2(w1, q8, superBlockCount)
            : VecDotQ5_K_Q8_0Scalar(w1, q8, superBlockCount);
        results[2] = Avx2.IsSupported
            ? VecDotQ5_K_Q8_0Avx2(w2, q8, superBlockCount)
            : VecDotQ5_K_Q8_0Scalar(w2, q8, superBlockCount);
        results[3] = Avx2.IsSupported
            ? VecDotQ5_K_Q8_0Avx2(w3, q8, superBlockCount)
            : VecDotQ5_K_Q8_0Scalar(w3, q8, superBlockCount);
    }

    [SkipLocalsInit]
    internal static void VecDotQ6_K_Q8_0Avx2_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* q8, int superBlockCount, float* results)
    {
        results[0] = Avx2.IsSupported
            ? VecDotQ6_K_Q8_0Avx2(w0, q8, superBlockCount)
            : VecDotQ6_K_Q8_0Scalar(w0, q8, superBlockCount);
        results[1] = Avx2.IsSupported
            ? VecDotQ6_K_Q8_0Avx2(w1, q8, superBlockCount)
            : VecDotQ6_K_Q8_0Scalar(w1, q8, superBlockCount);
        results[2] = Avx2.IsSupported
            ? VecDotQ6_K_Q8_0Avx2(w2, q8, superBlockCount)
            : VecDotQ6_K_Q8_0Scalar(w2, q8, superBlockCount);
        results[3] = Avx2.IsSupported
            ? VecDotQ6_K_Q8_0Avx2(w3, q8, superBlockCount)
            : VecDotQ6_K_Q8_0Scalar(w3, q8, superBlockCount);
    }

    // ──────────────────── ComputeRows for K-quants ────────────────────

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ4_K(byte* weights, byte* xQ8, float* result, int m, int superBlockCount)
    {
        int rowBytes = superBlockCount * Q4_K_BlockBytes;

        if (Avx2.IsSupported)
        {
            int row = 0;
            for (; row + 3 < m; row += 4)
            {
                VecDotQ4_K_Q8_0Avx2_4Rows(
                    weights + (long)row * rowBytes,
                    weights + (long)(row + 1) * rowBytes,
                    weights + (long)(row + 2) * rowBytes,
                    weights + (long)(row + 3) * rowBytes,
                    xQ8, superBlockCount, result + row);
            }
            for (; row < m; row++)
                result[row] = VecDotQ4_K_Q8_0Avx2(weights + (long)row * rowBytes, xQ8, superBlockCount);
        }
        else
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotQ4_K_Q8_0Scalar(weights + (long)row * rowBytes, xQ8, superBlockCount);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ5_K(byte* weights, byte* xQ8, float* result, int m, int superBlockCount)
    {
        int rowBytes = superBlockCount * Q5_K_BlockBytes;

        if (Avx2.IsSupported)
        {
            int row = 0;
            for (; row + 3 < m; row += 4)
            {
                VecDotQ5_K_Q8_0Avx2_4Rows(
                    weights + (long)row * rowBytes,
                    weights + (long)(row + 1) * rowBytes,
                    weights + (long)(row + 2) * rowBytes,
                    weights + (long)(row + 3) * rowBytes,
                    xQ8, superBlockCount, result + row);
            }
            for (; row < m; row++)
                result[row] = VecDotQ5_K_Q8_0Avx2(weights + (long)row * rowBytes, xQ8, superBlockCount);
        }
        else
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotQ5_K_Q8_0Scalar(weights + (long)row * rowBytes, xQ8, superBlockCount);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ6_K(byte* weights, byte* xQ8, float* result, int m, int superBlockCount)
    {
        int rowBytes = superBlockCount * Q6_K_BlockBytes;

        if (Avx2.IsSupported)
        {
            int row = 0;
            for (; row + 3 < m; row += 4)
            {
                VecDotQ6_K_Q8_0Avx2_4Rows(
                    weights + (long)row * rowBytes,
                    weights + (long)(row + 1) * rowBytes,
                    weights + (long)(row + 2) * rowBytes,
                    weights + (long)(row + 3) * rowBytes,
                    xQ8, superBlockCount, result + row);
            }
            for (; row < m; row++)
                result[row] = VecDotQ6_K_Q8_0Avx2(weights + (long)row * rowBytes, xQ8, superBlockCount);
        }
        else
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotQ6_K_Q8_0Scalar(weights + (long)row * rowBytes, xQ8, superBlockCount);
        }
    }

    // ──────────────────── Gemv for K-quants ────────────────────

    /// <summary>
    /// Q4_K GEMV: weights[M,K] in Q4_K × f32 input[K] → f32 output[M].
    /// Quantizes input to Q8_0, then uses fused Q4_K × Q8_0 vec_dot.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ4_K(byte* weights, float* x, float* result, int m, int k)
    {
        if (k % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {KQuantGroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_0GroupSize;      // Q8_0 blocks for quantizing input
        int superBlockCount = k / KQuantGroupSize;
        int xQ8Bytes = blockCount * Q8_0BlockBytes;

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
                QuantizeF32ToQ8_0(x, xQ8, k);
                ComputeRowsQ4_K(weights, xQ8, result, m, superBlockCount);
            }
            ArrayPool<byte>.Shared.Return(rented);
            return;
        }

        QuantizeF32ToQ8_0(x, xQ8, k);
        ComputeRowsQ4_K(weights, xQ8, result, m, superBlockCount);
    }

    /// <summary>Q5_K GEMV.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ5_K(byte* weights, float* x, float* result, int m, int k)
    {
        if (k % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {KQuantGroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_0GroupSize;
        int superBlockCount = k / KQuantGroupSize;
        int xQ8Bytes = blockCount * Q8_0BlockBytes;

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
                QuantizeF32ToQ8_0(x, xQ8, k);
                ComputeRowsQ5_K(weights, xQ8, result, m, superBlockCount);
            }
            ArrayPool<byte>.Shared.Return(rented);
            return;
        }

        QuantizeF32ToQ8_0(x, xQ8, k);
        ComputeRowsQ5_K(weights, xQ8, result, m, superBlockCount);
    }

    /// <summary>Q6_K GEMV.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ6_K(byte* weights, float* x, float* result, int m, int k)
    {
        if (k % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {KQuantGroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_0GroupSize;
        int superBlockCount = k / KQuantGroupSize;
        int xQ8Bytes = blockCount * Q8_0BlockBytes;

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
                QuantizeF32ToQ8_0(x, xQ8, k);
                ComputeRowsQ6_K(weights, xQ8, result, m, superBlockCount);
            }
            ArrayPool<byte>.Shared.Return(rented);
            return;
        }

        QuantizeF32ToQ8_0(x, xQ8, k);
        ComputeRowsQ6_K(weights, xQ8, result, m, superBlockCount);
    }

    // ──────────────────── Gemm for K-quants ────────────────────

    /// <summary>
    /// Q4_K GEMM: C[N,M] = B[N,K] × A[M,K]^T where A is Q4_K weights.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ4_K(byte* weights, float* b, float* c, int m, int k, int n,
                                byte* preQuantizedInput = null)
    {
        GemmKQuant(weights, b, c, m, k, n, Q4_K_BlockBytes,
            &ComputeRowsQ4_K, preQuantizedInput);
    }

    /// <summary>Q5_K GEMM.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ5_K(byte* weights, float* b, float* c, int m, int k, int n,
                                byte* preQuantizedInput = null)
    {
        GemmKQuant(weights, b, c, m, k, n, Q5_K_BlockBytes,
            &ComputeRowsQ5_K, preQuantizedInput);
    }

    /// <summary>Q6_K GEMM.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ6_K(byte* weights, float* b, float* c, int m, int k, int n,
                                byte* preQuantizedInput = null)
    {
        GemmKQuant(weights, b, c, m, k, n, Q6_K_BlockBytes,
            &ComputeRowsQ6_K, preQuantizedInput);
    }

    /// <summary>Shared GEMM implementation for K-quant formats.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmKQuant(byte* weights, float* b, float* c, int m, int k, int n,
                                   int kBlockBytes,
                                   delegate*<byte*, byte*, float*, int, int, void> computeRows,
                                   byte* preQuantizedInput)
    {
        if (k % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {KQuantGroupSize}, got {k}", nameof(k));

        int q8BlockCount = k / Q8_0GroupSize;
        int superBlockCount = k / KQuantGroupSize;
        int q8RowBytes = q8BlockCount * Q8_0BlockBytes;
        int kRowBytes = superBlockCount * kBlockBytes;

        if (n == 1)
        {
            if (preQuantizedInput != null)
            {
                computeRows(weights, preQuantizedInput, c, m, superBlockCount);
            }
            else
            {
                // Quantize input and compute
                int xQ8Bytes = q8BlockCount * Q8_0BlockBytes;
                byte[] rented = ArrayPool<byte>.Shared.Rent(xQ8Bytes);
                fixed (byte* xQ8 = rented)
                {
                    QuantizeF32ToQ8_0(b, xQ8, k);
                    computeRows(weights, xQ8, c, m, superBlockCount);
                }
                ArrayPool<byte>.Shared.Return(rented);
            }
            return;
        }

        if (preQuantizedInput != null)
        {
            // Tiled compute with pre-quantized input
            int tileM = ComputeTileM(kRowBytes);
            for (int mStart = 0; mStart < m; mStart += tileM)
            {
                int tileRows = Math.Min(tileM, m - mStart);
                byte* tileWeights = weights + (long)mStart * kRowBytes;
                for (int t = 0; t < n; t++)
                    computeRows(tileWeights, preQuantizedInput + t * q8RowBytes,
                        c + t * m + mStart, tileRows, superBlockCount);
            }
            return;
        }

        // Quantize all input rows, then tiled compute
        int totalQ8Bytes = n * q8RowBytes;
        byte[] rentedInput = ArrayPool<byte>.Shared.Rent(totalQ8Bytes);
        fixed (byte* inputQ8 = rentedInput)
        {
            for (int t = 0; t < n; t++)
                QuantizeF32ToQ8_0(b + t * k, inputQ8 + t * q8RowBytes, k);

            int tileM = ComputeTileM(kRowBytes);
            for (int mStart = 0; mStart < m; mStart += tileM)
            {
                int tileRows = Math.Min(tileM, m - mStart);
                byte* tileWeights = weights + (long)mStart * kRowBytes;
                for (int t = 0; t < n; t++)
                    computeRows(tileWeights, inputQ8 + t * q8RowBytes,
                        c + t * m + mStart, tileRows, superBlockCount);
            }
        }
        ArrayPool<byte>.Shared.Return(rentedInput);
    }

    // ──────────────────── Parallel overloads for K-quants ────────────────────

    /// <summary>Q4_K GEMV with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ4_K(byte* weights, float* x, float* result, int m, int k,
                                ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvQ4_K(weights, x, result, m, k);
            return;
        }

        GemvKQuantParallel(weights, x, result, m, k, Q4_K_BlockBytes,
            &ComputeRowsQ4_K, pool);
    }

    /// <summary>Q5_K GEMV with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ5_K(byte* weights, float* x, float* result, int m, int k,
                                ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvQ5_K(weights, x, result, m, k);
            return;
        }

        GemvKQuantParallel(weights, x, result, m, k, Q5_K_BlockBytes,
            &ComputeRowsQ5_K, pool);
    }

    /// <summary>Q6_K GEMV with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ6_K(byte* weights, float* x, float* result, int m, int k,
                                ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvQ6_K(weights, x, result, m, k);
            return;
        }

        GemvKQuantParallel(weights, x, result, m, k, Q6_K_BlockBytes,
            &ComputeRowsQ6_K, pool);
    }

    /// <summary>Shared parallel GEMV for K-quant formats.</summary>
    [SkipLocalsInit]
    private static void GemvKQuantParallel(
        byte* weights, float* x, float* result, int m, int k,
        int kBlockBytes,
        delegate*<byte*, byte*, float*, int, int, void> computeRows,
        ComputeThreadPool pool)
    {
        if (k % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {KQuantGroupSize}, got {k}", nameof(k));

        int q8BlockCount = k / Q8_0GroupSize;
        int superBlockCount = k / KQuantGroupSize;
        int xQ8Bytes = q8BlockCount * Q8_0BlockBytes;

        // Quantize x once into pool scratch for thread 0
        byte* xQ8 = (byte*)pool.GetWorkerScratch(0, xQ8Bytes);
        QuantizeF32ToQ8_0(x, xQ8, k);

        var ctx = new ComputeRowsKQuantCtx
        {
            Weights = weights, XQ8 = xQ8, Result = result,
            M = m, SuperBlockCount = superBlockCount,
            KBlockBytes = kBlockBytes, ComputeRows = computeRows
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsKQuantWorker);
    }

    /// <summary>Q4_K GEMM with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ4_K(byte* weights, float* b, float* c, int m, int k, int n,
                                ComputeThreadPool? pool, byte* preQuantizedInput = null)
    {
        if (pool is null)
        {
            GemmQ4_K(weights, b, c, m, k, n, preQuantizedInput);
            return;
        }

        GemmKQuantParallel(weights, b, c, m, k, n, Q4_K_BlockBytes,
            &ComputeRowsQ4_K, pool, preQuantizedInput);
    }

    /// <summary>Q5_K GEMM with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ5_K(byte* weights, float* b, float* c, int m, int k, int n,
                                ComputeThreadPool? pool, byte* preQuantizedInput = null)
    {
        if (pool is null)
        {
            GemmQ5_K(weights, b, c, m, k, n, preQuantizedInput);
            return;
        }

        GemmKQuantParallel(weights, b, c, m, k, n, Q5_K_BlockBytes,
            &ComputeRowsQ5_K, pool, preQuantizedInput);
    }

    /// <summary>Q6_K GEMM with optional parallelism.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ6_K(byte* weights, float* b, float* c, int m, int k, int n,
                                ComputeThreadPool? pool, byte* preQuantizedInput = null)
    {
        if (pool is null)
        {
            GemmQ6_K(weights, b, c, m, k, n, preQuantizedInput);
            return;
        }

        GemmKQuantParallel(weights, b, c, m, k, n, Q6_K_BlockBytes,
            &ComputeRowsQ6_K, pool, preQuantizedInput);
    }

    /// <summary>Shared parallel GEMM for K-quant formats.</summary>
    [SkipLocalsInit]
    private static void GemmKQuantParallel(
        byte* weights, float* b, float* c, int m, int k, int n,
        int kBlockBytes,
        delegate*<byte*, byte*, float*, int, int, void> computeRows,
        ComputeThreadPool pool, byte* preQuantizedInput)
    {
        if (k % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {KQuantGroupSize}, got {k}", nameof(k));

        if (n == 1)
        {
            if (preQuantizedInput != null)
            {
                int superBlockCount = k / KQuantGroupSize;
                var ctx = new ComputeRowsKQuantCtx
                {
                    Weights = weights, XQ8 = preQuantizedInput, Result = c,
                    M = m, SuperBlockCount = superBlockCount,
                    KBlockBytes = kBlockBytes, ComputeRows = computeRows
                };
                pool.Dispatch((nint)(&ctx), &ComputeRowsKQuantWorker);
            }
            else
            {
                // Use the parallel GEMV overload
                GemvKQuantParallel(weights, b, c, m, k, kBlockBytes, computeRows, pool);
            }
            return;
        }

        int q8BlockCount = k / Q8_0GroupSize;
        int superBlockCount2 = k / KQuantGroupSize;
        int q8RowBytes = q8BlockCount * Q8_0BlockBytes;
        int kRowBytes = superBlockCount2 * kBlockBytes;
        int tileM = ComputeTileM(kRowBytes);
        int totalTiles = (m + tileM - 1) / tileM;

        if (preQuantizedInput != null)
        {
            if (totalTiles < 2)
            {
                GemmKQuant(weights, b, c, m, k, n, kBlockBytes, computeRows, preQuantizedInput);
                return;
            }

            var ctx = new GemmTiledKQuantCtx
            {
                Weights = weights, InputQ8 = preQuantizedInput, C = c,
                M = m, N = n, SuperBlockCount = superBlockCount2,
                TileM = tileM, KRowBytes = kRowBytes, Q8RowBytes = q8RowBytes,
                ComputeRows = computeRows
            };
            pool.Dispatch((nint)(&ctx), &GemmTiledKQuantWorker);
            return;
        }

        // Quantize all input rows, then parallel tiled compute
        int totalQ8Bytes = n * q8RowBytes;
        byte[] rented = ArrayPool<byte>.Shared.Rent(totalQ8Bytes);
        fixed (byte* inputQ8 = rented)
        {
            for (int t = 0; t < n; t++)
                QuantizeF32ToQ8_0(b + t * k, inputQ8 + t * q8RowBytes, k);

            if (totalTiles < 2)
            {
                for (int mStart = 0; mStart < m; mStart += tileM)
                {
                    int tileRows = Math.Min(tileM, m - mStart);
                    byte* tileWeights = weights + (long)mStart * kRowBytes;
                    for (int t = 0; t < n; t++)
                        computeRows(tileWeights, inputQ8 + t * q8RowBytes,
                            c + t * m + mStart, tileRows, superBlockCount2);
                }
            }
            else
            {
                var ctx = new GemmTiledKQuantCtx
                {
                    Weights = weights, InputQ8 = inputQ8, C = c,
                    M = m, N = n, SuperBlockCount = superBlockCount2,
                    TileM = tileM, KRowBytes = kRowBytes, Q8RowBytes = q8RowBytes,
                    ComputeRows = computeRows
                };
                pool.Dispatch((nint)(&ctx), &GemmTiledKQuantWorker);
            }
        }
        ArrayPool<byte>.Shared.Return(rented);
    }

    // ──────────────────── Context structs and workers ────────────────────

    private struct ComputeRowsKQuantCtx
    {
        public byte* Weights;
        public byte* XQ8;
        public float* Result;
        public int M;
        public int SuperBlockCount;
        public int KBlockBytes;
        public delegate*<byte*, byte*, float*, int, int, void> ComputeRows;
    }

    private static void ComputeRowsKQuantWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<ComputeRowsKQuantCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        int rowBytes = ctx.SuperBlockCount * ctx.KBlockBytes;
        ctx.ComputeRows(ctx.Weights + (long)start * rowBytes, ctx.XQ8,
            ctx.Result + start, count, ctx.SuperBlockCount);
    }

    private struct GemmTiledKQuantCtx
    {
        public byte* Weights;
        public byte* InputQ8;
        public float* C;
        public int M;
        public int N;
        public int SuperBlockCount;
        public int TileM;
        public int KRowBytes;
        public int Q8RowBytes;
        public delegate*<byte*, byte*, float*, int, int, void> ComputeRows;
    }

    private static void GemmTiledKQuantWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmTiledKQuantCtx>((void*)ctxPtr);
        int totalTiles = (ctx.M + ctx.TileM - 1) / ctx.TileM;
        int tilesPerThread = (totalTiles + threadCount - 1) / threadCount;
        int startTile = threadIdx * tilesPerThread;
        int endTile = Math.Min(startTile + tilesPerThread, totalTiles);

        for (int tile = startTile; tile < endTile; tile++)
        {
            int mStart = tile * ctx.TileM;
            int tileRows = Math.Min(ctx.TileM, ctx.M - mStart);
            byte* tileWeights = ctx.Weights + (long)mStart * ctx.KRowBytes;
            for (int t = 0; t < ctx.N; t++)
                ctx.ComputeRows(tileWeights, ctx.InputQ8 + t * ctx.Q8RowBytes,
                    ctx.C + t * ctx.M + mStart, tileRows, ctx.SuperBlockCount);
        }
    }
}
