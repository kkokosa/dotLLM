using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// K-quant matrix multiplication kernels for Q4_K, Q5_K, and Q6_K × Q8_K dot products.
/// Q8_K uses float32 scale (not Half) to avoid precision loss that compounds across
/// transformer layers. One K-quant super-block (256 values) corresponds to one Q8_K block.
/// </summary>
public static unsafe partial class MatMul
{
    private const int Q4_K_BlockBytes = 144;
    private const int Q5_K_BlockBytes = 176;
    private const int Q6_K_BlockBytes = 210;
    private const int KQuantGroupSize = 256;

    /// <summary>Q8_K block: float d(4) + sbyte qs[256](256) + short bsums[16](32) = 292 bytes.</summary>
    public const int Q8_K_BlockBytes = 292;

    /// <summary>Elements per Q8_K block.</summary>
    private const int Q8_K_GroupSize = 256;

    // ──────────────────── Q8_K quantization ────────────────────

    /// <summary>
    /// Quantizes f32 activations to Q8_K format for K-quant vec_dot kernels.
    /// Q8_K uses float32 scale (not Half) to avoid precision loss that compounds
    /// across transformer layers when used with K-quant weights.
    /// Block layout: float d(4 bytes) + sbyte qs[256] + short bsums[16](32 bytes) = 292 bytes.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeF32ToQ8_K(float* src, byte* dest, int elementCount)
    {
        if (elementCount % Q8_K_GroupSize != 0)
            throw new ArgumentException(
                $"elementCount must be a multiple of {Q8_K_GroupSize}, got {elementCount}",
                nameof(elementCount));

        if (Avx2.IsSupported)
            QuantizeF32ToQ8_KAvx2(src, dest, elementCount);
        else
            QuantizeF32ToQ8_KScalar(src, dest, elementCount);
    }

    /// <summary>Scalar Q8_K quantization reference implementation.</summary>
    [SkipLocalsInit]
    internal static void QuantizeF32ToQ8_KScalar(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / Q8_K_GroupSize;

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * Q8_K_GroupSize;
            byte* blockDst = dest + block * Q8_K_BlockBytes;

            float maxAbs = 0;
            for (int i = 0; i < Q8_K_GroupSize; i++)
            {
                float abs = MathF.Abs(blockSrc[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            float scale = maxAbs / 127.0f;
            Unsafe.WriteUnaligned(blockDst, scale); // float32, not Half

            sbyte* qs = (sbyte*)(blockDst + 4);
            short* bsums = (short*)(blockDst + 260);

            if (scale == 0)
            {
                for (int i = 0; i < Q8_K_GroupSize; i++)
                    qs[i] = 0;
                for (int i = 0; i < 16; i++)
                    bsums[i] = 0;
            }
            else
            {
                float invScale = 1.0f / scale;
                for (int g = 0; g < 16; g++)
                {
                    int sum = 0;
                    for (int i = 0; i < 16; i++)
                    {
                        int idx = g * 16 + i;
                        int v = (int)MathF.Round(blockSrc[idx] * invScale);
                        sbyte qv = (sbyte)Math.Clamp(v, -127, 127);
                        qs[idx] = qv;
                        sum += qv;
                    }
                    bsums[g] = (short)sum;
                }
            }
        }
    }

    /// <summary>AVX2 Q8_K quantization: 256 floats per block.</summary>
    [SkipLocalsInit]
    internal static void QuantizeF32ToQ8_KAvx2(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / Q8_K_GroupSize;

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * Q8_K_GroupSize;
            byte* blockDst = dest + block * Q8_K_BlockBytes;

            // Max-abs scan over 256 floats (32 × 8-wide vectors)
            Vector256<float> maxVec = Vector256<float>.Zero;
            for (int i = 0; i < Q8_K_GroupSize; i += 8)
                maxVec = Avx.Max(maxVec, Vector256.Abs(Avx.LoadVector256(blockSrc + i)));
            float maxAbs = HorizontalMaxAvx2(maxVec);

            float scale = maxAbs / 127.0f;
            Unsafe.WriteUnaligned(blockDst, scale);

            sbyte* qs = (sbyte*)(blockDst + 4);
            short* bsums = (short*)(blockDst + 260);

            if (scale == 0)
            {
                for (int i = 0; i < Q8_K_GroupSize; i += 32)
                    Vector256<sbyte>.Zero.StoreUnsafe(ref Unsafe.AsRef<sbyte>(qs + i));
                for (int i = 0; i < 16; i++)
                    bsums[i] = 0;
            }
            else
            {
                Vector256<float> vInvScale = Vector256.Create(1.0f / scale);

                // Quantize 32 floats at a time (8 chunks for 256 total)
                for (int chunk = 0; chunk < 8; chunk++)
                {
                    float* chunkSrc = blockSrc + chunk * 32;
                    sbyte* chunkDst = qs + chunk * 32;

                    Vector256<int> i0 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(
                        Avx.Multiply(Avx.LoadVector256(chunkSrc), vInvScale)));
                    Vector256<int> i1 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(
                        Avx.Multiply(Avx.LoadVector256(chunkSrc + 8), vInvScale)));
                    Vector256<int> i2 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(
                        Avx.Multiply(Avx.LoadVector256(chunkSrc + 16), vInvScale)));
                    Vector256<int> i3 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(
                        Avx.Multiply(Avx.LoadVector256(chunkSrc + 24), vInvScale)));

                    Vector256<short> s01 = Avx2.PackSignedSaturate(i0, i1);
                    Vector256<short> s23 = Avx2.PackSignedSaturate(i2, i3);
                    Vector256<sbyte> packed = Avx2.PackSignedSaturate(s01, s23);

                    Vector256<int> permuted = Avx2.PermuteVar8x32(packed.AsInt32(),
                        Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7));

                    permuted.AsByte().StoreUnsafe(ref Unsafe.AsRef<byte>((byte*)chunkDst));
                }

                // Compute bsums: sum each group of 16 consecutive sbyte values
                for (int g = 0; g < 16; g++)
                {
                    int sum = 0;
                    for (int i = 0; i < 16; i++)
                        sum += qs[g * 16 + i];
                    bsums[g] = (short)sum;
                }
            }
        }
    }

    // ──────────────────── Q6_K × Q8_K scalar ────────────────────

    /// <summary>
    /// Scalar Q6_K × Q8_K dot product.
    /// Q6_K layout: ql[128]@0, qh[64]@128, scales[16](int8)@192, d(Half)@208.
    /// Q8_K layout: d(float)@0, qs[256]@4, bsums[16]@260.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ6_K_Q8_KScalar(byte* qk, byte* q8k, int superBlockCount)
    {
        float sumf = 0;

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            byte* ql = qk;
            byte* qh = qk + 128;
            sbyte* scales = (sbyte*)(qk + 192);
            float d6 = (float)Unsafe.ReadUnaligned<Half>(qk + 208);

            float d8 = Unsafe.ReadUnaligned<float>(q8k); // float32 scale — no precision loss
            sbyte* q8qs = (sbyte*)(q8k + 4);             // 256 contiguous values

            float dProd = d6 * d8;

            // 16 sub-blocks of 16 values each, matching llama.cpp element ordering.
            for (int sub = 0; sub < 16; sub++)
            {
                float sc = dProd * scales[sub];

                // Q8_K values are contiguous: sub-block sub → offset sub*16
                sbyte* q8Vals = q8qs + sub * 16;

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

                sumf += sc * sumi;
            }

            qk += Q6_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return sumf;
    }

    // ──────────────────── Q4_K × Q8_K scalar ────────────────────

    /// <summary>
    /// Scalar Q4_K × Q8_K dot product.
    /// Q4_K layout: d(Half@0), dmin(Half@2), scales[12]@4, qs[128]@16.
    /// Q8_K layout: d(float)@0, qs[256]@4, bsums[16]@260.
    /// Uses precomputed bsums for the min correction term.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ4_K_Q8_KScalar(byte* qk, byte* q8k, int superBlockCount)
    {
        float sumf = 0;

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d4 = (float)Unsafe.ReadUnaligned<Half>(qk);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(qk + 2);
            Dequantize.UnpackQ4Q5Scales(qk + 4, scBuf, mnBuf);
            byte* qs = qk + 16;

            float d8 = Unsafe.ReadUnaligned<float>(q8k); // float32 scale
            sbyte* q8qs = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);

            for (int j = 0; j < 8; j++)
            {
                float sc = d4 * scBuf[j];
                float mn = dmin * mnBuf[j];

                sbyte* q8Vals = q8qs + j * 32;

                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                int prodSum = 0;
                for (int i = 0; i < 32; i++)
                {
                    int qsByte = pairIdx * 32 + i;
                    int nibble = nibbleHalf == 0 ? (qs[qsByte] & 0xF) : (qs[qsByte] >> 4);
                    prodSum += nibble * q8Vals[i];
                }

                // Use precomputed bsums: bsums[j*2] + bsums[j*2+1] = sum of 32 Q8_K values
                int q8Sum = bsums[j * 2] + bsums[j * 2 + 1];
                sumf += d8 * (sc * prodSum - mn * q8Sum);
            }

            qk += Q4_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return sumf;
    }

    // ──────────────────── Q5_K × Q8_K scalar ────────────────────

    /// <summary>
    /// Scalar Q5_K × Q8_K dot product.
    /// Q5_K layout: d(Half@0), dmin(Half@2), scales[12]@4, qh[32]@16, qs[128]@48.
    /// Q8_K layout: d(float)@0, qs[256]@4, bsums[16]@260.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_K_Q8_KScalar(byte* qk, byte* q8k, int superBlockCount)
    {
        float sumf = 0;

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d5 = (float)Unsafe.ReadUnaligned<Half>(qk);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(qk + 2);
            Dequantize.UnpackQ4Q5Scales(qk + 4, scBuf, mnBuf);
            byte* qh = qk + 16;
            byte* qs = qk + 48;

            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8qs = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);

            for (int j = 0; j < 8; j++)
            {
                float sc = d5 * scBuf[j];
                float mn = dmin * mnBuf[j];

                sbyte* q8Vals = q8qs + j * 32;

                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                int prodSum = 0;
                for (int i = 0; i < 32; i++)
                {
                    int qsByte = pairIdx * 32 + i;
                    int lo4 = nibbleHalf == 0 ? (qs[qsByte] & 0xF) : (qs[qsByte] >> 4);
                    int bit5 = (qh[i] >> j) & 1;
                    int q = lo4 | (bit5 << 4);
                    prodSum += q * q8Vals[i];
                }

                int q8Sum = bsums[j * 2] + bsums[j * 2 + 1];
                sumf += d8 * (sc * prodSum - mn * q8Sum);
            }

            qk += Q5_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return sumf;
    }

    // ──────────────────── AVX2 vec_dot kernels ────────────────────

    /// <summary>
    /// AVX2 Q4_K × Q8_K dot product. Processes one sub-block (32 values) at a time.
    /// Uses vpmaddubsw for unsigned×signed multiply-accumulate.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ4_K_Q8_KAvx2(byte* qk, byte* q8k, int superBlockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        Vector256<short> ones = Vector256.Create((short)1);

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d4 = (float)Unsafe.ReadUnaligned<Half>(qk);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(qk + 2);
            Dequantize.UnpackQ4Q5Scales(qk + 4, scBuf, mnBuf);
            byte* qs = qk + 16;

            float d8 = Unsafe.ReadUnaligned<float>(q8k); // float32 scale
            sbyte* q8qs = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);

            for (int j = 0; j < 8; j++)
            {
                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs + pairIdx * 32);
                Vector256<byte> nibbles;
                if (nibbleHalf == 0)
                    nibbles = Avx2.And(raw, mask0F);
                else
                    nibbles = Avx2.And(
                        Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);

                // Load 32 Q8_K values (contiguous, no header gaps)
                Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8qs + j * 32);

                // Product sum: unsigned nibbles × signed q8
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(nibbles, q8Vals);
                Vector256<int> prodSum = Avx2.MultiplyAddAdjacent(prod, ones);

                // Q8 raw sum via SIMD
                Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(
                    Vector256.Create((byte)1), q8Vals);
                Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);

                float sc = d4 * scBuf[j];
                float mn = dmin * mnBuf[j];

                Vector256<float> fProd = Avx.ConvertToVector256Single(prodSum);
                Vector256<float> fQ8 = Avx.ConvertToVector256Single(q8Sum);

                Vector256<float> term = Avx.Subtract(
                    Avx.Multiply(Vector256.Create(sc), fProd),
                    Avx.Multiply(Vector256.Create(mn), fQ8));

                acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(d8), term));
            }

            qk += Q4_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>
    /// AVX2 Q6_K × Q8_K dot product.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ6_K_Q8_KAvx2(byte* qk, byte* q8k, int superBlockCount)
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
            float d6 = (float)Unsafe.ReadUnaligned<Half>(qk + 208);

            float d8 = Unsafe.ReadUnaligned<float>(q8k); // float32 scale
            sbyte* q8qs = (sbyte*)(q8k + 4);

            // 16 sub-blocks of 16, processed in pairs (32 values).
            for (int sub = 0; sub < 16; sub += 2)
            {
                float sc0 = d6 * scales[sub];
                float sc1 = d6 * scales[sub + 1];

                int half = sub / 8;
                int sh = sub % 8;
                int qlBase = half * 64 + (sh % 4) * 16;
                bool isUpper = sh >= 4;
                int qhBase = half * 32;
                int qhShift = (sh / 2) * 2;

                Vector256<byte> qlRaw = Unsafe.ReadUnaligned<Vector256<byte>>(ql + qlBase);
                Vector256<byte> nibbles;
                if (isUpper)
                    nibbles = Avx2.And(
                        Avx2.ShiftRightLogical(qlRaw.AsUInt16(), 4).AsByte(), mask0F);
                else
                    nibbles = Avx2.And(qlRaw, mask0F);

                Vector256<byte> qhVec = Unsafe.ReadUnaligned<Vector256<byte>>(qh + qhBase);
                Vector256<byte> qhBits = qhShift switch
                {
                    0 => Avx2.And(qhVec, mask03),
                    2 => Avx2.And(Avx2.ShiftRightLogical(qhVec.AsUInt16(), 2).AsByte(), mask03),
                    4 => Avx2.And(Avx2.ShiftRightLogical(qhVec.AsUInt16(), 4).AsByte(), mask03),
                    _ => Avx2.And(Avx2.ShiftRightLogical(qhVec.AsUInt16(), 6).AsByte(), mask03),
                };

                Vector256<byte> q6u = Avx2.And(
                    Avx2.Or(nibbles, Avx2.ShiftLeftLogical(qhBits.AsUInt16(), 4).AsByte()),
                    Vector256.Create((byte)0x3F));

                // Load 32 Q8_K values (contiguous)
                Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8qs + sub * 16);

                Vector256<short> prod = Avx2.MultiplyAddAdjacent(q6u, q8Vals);
                Vector256<int> prodSum = Avx2.MultiplyAddAdjacent(prod, ones);

                Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), q8Vals);
                Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);

                Vector128<float> prodLo = Avx.ConvertToVector128Single(prodSum.GetLower());
                Vector128<float> prodHi = Avx.ConvertToVector128Single(prodSum.GetUpper());
                Vector128<float> q8Lo = Avx.ConvertToVector128Single(q8Sum.GetLower());
                Vector128<float> q8Hi = Avx.ConvertToVector128Single(q8Sum.GetUpper());

                Vector128<float> bias = Vector128.Create(32f);
                Vector128<float> termLo = Sse.Multiply(Vector128.Create(sc0),
                    Sse.Subtract(prodLo, Sse.Multiply(bias, q8Lo)));
                Vector128<float> termHi = Sse.Multiply(Vector128.Create(sc1),
                    Sse.Subtract(prodHi, Sse.Multiply(bias, q8Hi)));

                Vector256<float> term = Vector256.Create(termLo, termHi);
                acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(d8), term));
            }

            qk += Q6_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>
    /// AVX2 Q5_K × Q8_K dot product.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_K_Q8_KAvx2(byte* qk, byte* q8k, int superBlockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        Vector256<short> ones = Vector256.Create((short)1);

        byte* scBuf = stackalloc byte[8];
        byte* mnBuf = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d5 = (float)Unsafe.ReadUnaligned<Half>(qk);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(qk + 2);
            Dequantize.UnpackQ4Q5Scales(qk + 4, scBuf, mnBuf);
            byte* qh = qk + 16;
            byte* qs = qk + 48;

            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8qs = (sbyte*)(q8k + 4);

            for (int j = 0; j < 8; j++)
            {
                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs + pairIdx * 32);
                Vector256<byte> nibbles;
                if (nibbleHalf == 0)
                    nibbles = Avx2.And(raw, mask0F);
                else
                    nibbles = Avx2.And(
                        Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);

                Vector256<byte> qhVec = Unsafe.ReadUnaligned<Vector256<byte>>(qh);
                byte bitMask = (byte)(1 << j);
                Vector256<byte> hasBit = Avx2.CompareEqual(
                    Avx2.And(qhVec, Vector256.Create(bitMask)), Vector256.Create(bitMask));
                Vector256<byte> bit5Vec = Avx2.And(hasBit, Vector256.Create((byte)16));

                Vector256<byte> q5vals = Avx2.Or(nibbles, bit5Vec);

                // Load 32 Q8_K values (contiguous)
                Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8qs + j * 32);

                Vector256<short> prod = Avx2.MultiplyAddAdjacent(q5vals, q8Vals);
                Vector256<int> prodSum = Avx2.MultiplyAddAdjacent(prod, ones);

                Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), q8Vals);
                Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);

                float sc = d5 * scBuf[j];
                float mn = dmin * mnBuf[j];

                Vector256<float> fProd = Avx.ConvertToVector256Single(prodSum);
                Vector256<float> fQ8 = Avx.ConvertToVector256Single(q8Sum);

                Vector256<float> term = Avx.Subtract(
                    Avx.Multiply(Vector256.Create(sc), fProd),
                    Avx.Multiply(Vector256.Create(mn), fQ8));

                acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(d8), term));
            }

            qk += Q5_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        return HorizontalSumAvx2Float(acc);
    }

    // ──────────────────── True 4-row kernels with shared Q8_K activation loading ──────────────

    /// <summary>
    /// True 4-row Q4_K × Q8_K: loads Q8_K activation once per sub-block, computes 4 dot products.
    /// Saves ~3× L1 cache bandwidth for activations vs calling single-row 4 times.
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ4_K_Q8_K_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* q8k, int superBlockCount, float* results)
    {
        if (!Avx2.IsSupported)
        {
            results[0] = VecDotQ4_K_Q8_KScalar(w0, q8k, superBlockCount);
            results[1] = VecDotQ4_K_Q8_KScalar(w1, q8k, superBlockCount);
            results[2] = VecDotQ4_K_Q8_KScalar(w2, q8k, superBlockCount);
            results[3] = VecDotQ4_K_Q8_KScalar(w3, q8k, superBlockCount);
            return;
        }

        Vector256<float> acc0 = Vector256<float>.Zero, acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero, acc3 = Vector256<float>.Zero;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        Vector256<short> ones = Vector256.Create((short)1);

        byte* sc0 = stackalloc byte[8]; byte* mn0 = stackalloc byte[8];
        byte* sc1 = stackalloc byte[8]; byte* mn1 = stackalloc byte[8];
        byte* sc2 = stackalloc byte[8]; byte* mn2 = stackalloc byte[8];
        byte* sc3 = stackalloc byte[8]; byte* mn3 = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8qs = (sbyte*)(q8k + 4);

            float d4_0 = (float)Unsafe.ReadUnaligned<Half>(w0);
            float dm0 = (float)Unsafe.ReadUnaligned<Half>(w0 + 2);
            Dequantize.UnpackQ4Q5Scales(w0 + 4, sc0, mn0);
            float d4_1 = (float)Unsafe.ReadUnaligned<Half>(w1);
            float dm1 = (float)Unsafe.ReadUnaligned<Half>(w1 + 2);
            Dequantize.UnpackQ4Q5Scales(w1 + 4, sc1, mn1);
            float d4_2 = (float)Unsafe.ReadUnaligned<Half>(w2);
            float dm2 = (float)Unsafe.ReadUnaligned<Half>(w2 + 2);
            Dequantize.UnpackQ4Q5Scales(w2 + 4, sc2, mn2);
            float d4_3 = (float)Unsafe.ReadUnaligned<Half>(w3);
            float dm3 = (float)Unsafe.ReadUnaligned<Half>(w3 + 2);
            Dequantize.UnpackQ4Q5Scales(w3 + 4, sc3, mn3);

            byte* qs0p = w0 + 16; byte* qs1p = w1 + 16;
            byte* qs2p = w2 + 16; byte* qs3p = w3 + 16;

            for (int j = 0; j < 8; j++)
            {
                // Load Q8_K activation ONCE for all 4 rows
                Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8qs + j * 32);
                Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), q8Vals);
                Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);
                Vector256<float> fQ8 = Avx.ConvertToVector256Single(q8Sum);

                int pairIdx = j / 2;
                int nibbleHalf = j % 2;

                // Row 0
                {
                    Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs0p + pairIdx * 32);
                    Vector256<byte> nib = nibbleHalf == 0 ? Avx2.And(raw, mask0F)
                        : Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);
                    Vector256<int> ps = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(nib, q8Vals), ones);
                    Vector256<float> t = Avx.Subtract(
                        Avx.Multiply(Vector256.Create(d4_0 * sc0[j]), Avx.ConvertToVector256Single(ps)),
                        Avx.Multiply(Vector256.Create(dm0 * mn0[j]), fQ8));
                    acc0 = Avx.Add(acc0, Avx.Multiply(Vector256.Create(d8), t));
                }
                // Row 1
                {
                    Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs1p + pairIdx * 32);
                    Vector256<byte> nib = nibbleHalf == 0 ? Avx2.And(raw, mask0F)
                        : Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);
                    Vector256<int> ps = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(nib, q8Vals), ones);
                    Vector256<float> t = Avx.Subtract(
                        Avx.Multiply(Vector256.Create(d4_1 * sc1[j]), Avx.ConvertToVector256Single(ps)),
                        Avx.Multiply(Vector256.Create(dm1 * mn1[j]), fQ8));
                    acc1 = Avx.Add(acc1, Avx.Multiply(Vector256.Create(d8), t));
                }
                // Row 2
                {
                    Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs2p + pairIdx * 32);
                    Vector256<byte> nib = nibbleHalf == 0 ? Avx2.And(raw, mask0F)
                        : Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);
                    Vector256<int> ps = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(nib, q8Vals), ones);
                    Vector256<float> t = Avx.Subtract(
                        Avx.Multiply(Vector256.Create(d4_2 * sc2[j]), Avx.ConvertToVector256Single(ps)),
                        Avx.Multiply(Vector256.Create(dm2 * mn2[j]), fQ8));
                    acc2 = Avx.Add(acc2, Avx.Multiply(Vector256.Create(d8), t));
                }
                // Row 3
                {
                    Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qs3p + pairIdx * 32);
                    Vector256<byte> nib = nibbleHalf == 0 ? Avx2.And(raw, mask0F)
                        : Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), mask0F);
                    Vector256<int> ps = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(nib, q8Vals), ones);
                    Vector256<float> t = Avx.Subtract(
                        Avx.Multiply(Vector256.Create(d4_3 * sc3[j]), Avx.ConvertToVector256Single(ps)),
                        Avx.Multiply(Vector256.Create(dm3 * mn3[j]), fQ8));
                    acc3 = Avx.Add(acc3, Avx.Multiply(Vector256.Create(d8), t));
                }
            }

            w0 += Q4_K_BlockBytes; w1 += Q4_K_BlockBytes;
            w2 += Q4_K_BlockBytes; w3 += Q4_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        results[0] = HorizontalSumAvx2Float(acc0);
        results[1] = HorizontalSumAvx2Float(acc1);
        results[2] = HorizontalSumAvx2Float(acc2);
        results[3] = HorizontalSumAvx2Float(acc3);
    }

    /// <summary>
    /// True 4-row Q5_K × Q8_K with shared activation loading.
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ5_K_Q8_K_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* q8k, int superBlockCount, float* results)
    {
        if (!Avx2.IsSupported)
        {
            results[0] = VecDotQ5_K_Q8_KScalar(w0, q8k, superBlockCount);
            results[1] = VecDotQ5_K_Q8_KScalar(w1, q8k, superBlockCount);
            results[2] = VecDotQ5_K_Q8_KScalar(w2, q8k, superBlockCount);
            results[3] = VecDotQ5_K_Q8_KScalar(w3, q8k, superBlockCount);
            return;
        }

        Vector256<float> acc0 = Vector256<float>.Zero, acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero, acc3 = Vector256<float>.Zero;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        Vector256<short> ones = Vector256.Create((short)1);

        byte* sc0 = stackalloc byte[8]; byte* mn0 = stackalloc byte[8];
        byte* sc1 = stackalloc byte[8]; byte* mn1 = stackalloc byte[8];
        byte* sc2 = stackalloc byte[8]; byte* mn2 = stackalloc byte[8];
        byte* sc3 = stackalloc byte[8]; byte* mn3 = stackalloc byte[8];

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8qs = (sbyte*)(q8k + 4);

            float d5_0 = (float)Unsafe.ReadUnaligned<Half>(w0);
            float dm0 = (float)Unsafe.ReadUnaligned<Half>(w0 + 2);
            Dequantize.UnpackQ4Q5Scales(w0 + 4, sc0, mn0);
            float d5_1 = (float)Unsafe.ReadUnaligned<Half>(w1);
            float dm1 = (float)Unsafe.ReadUnaligned<Half>(w1 + 2);
            Dequantize.UnpackQ4Q5Scales(w1 + 4, sc1, mn1);
            float d5_2 = (float)Unsafe.ReadUnaligned<Half>(w2);
            float dm2 = (float)Unsafe.ReadUnaligned<Half>(w2 + 2);
            Dequantize.UnpackQ4Q5Scales(w2 + 4, sc2, mn2);
            float d5_3 = (float)Unsafe.ReadUnaligned<Half>(w3);
            float dm3 = (float)Unsafe.ReadUnaligned<Half>(w3 + 2);
            Dequantize.UnpackQ4Q5Scales(w3 + 4, sc3, mn3);

            byte* qh0 = w0 + 16; byte* qs0p = w0 + 48;
            byte* qh1 = w1 + 16; byte* qs1p = w1 + 48;
            byte* qh2 = w2 + 16; byte* qs2p = w2 + 48;
            byte* qh3 = w3 + 16; byte* qs3p = w3 + 48;

            Vector256<byte> qhVec0 = Unsafe.ReadUnaligned<Vector256<byte>>(qh0);
            Vector256<byte> qhVec1 = Unsafe.ReadUnaligned<Vector256<byte>>(qh1);
            Vector256<byte> qhVec2 = Unsafe.ReadUnaligned<Vector256<byte>>(qh2);
            Vector256<byte> qhVec3 = Unsafe.ReadUnaligned<Vector256<byte>>(qh3);

            for (int j = 0; j < 8; j++)
            {
                // Shared activation
                Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8qs + j * 32);
                Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), q8Vals);
                Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);
                Vector256<float> fQ8 = Avx.ConvertToVector256Single(q8Sum);

                int pairIdx = j / 2;
                int nibbleHalf = j % 2;
                byte bitMask = (byte)(1 << j);

                // Helper: extract Q5 values for a row
                static Vector256<byte> ExtractQ5(byte* qsRow, int pairIdx, int nibbleHalf,
                    Vector256<byte> qhV, byte bm, Vector256<byte> m0F)
                {
                    Vector256<byte> raw = Unsafe.ReadUnaligned<Vector256<byte>>(qsRow + pairIdx * 32);
                    Vector256<byte> nib = nibbleHalf == 0 ? Avx2.And(raw, m0F)
                        : Avx2.And(Avx2.ShiftRightLogical(raw.AsUInt16(), 4).AsByte(), m0F);
                    Vector256<byte> hasBit = Avx2.CompareEqual(
                        Avx2.And(qhV, Vector256.Create(bm)), Vector256.Create(bm));
                    return Avx2.Or(nib, Avx2.And(hasBit, Vector256.Create((byte)16)));
                }

                // Row 0
                {
                    Vector256<byte> q5 = ExtractQ5(qs0p, pairIdx, nibbleHalf, qhVec0, bitMask, mask0F);
                    Vector256<int> ps = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(q5, q8Vals), ones);
                    Vector256<float> t = Avx.Subtract(
                        Avx.Multiply(Vector256.Create(d5_0 * sc0[j]), Avx.ConvertToVector256Single(ps)),
                        Avx.Multiply(Vector256.Create(dm0 * mn0[j]), fQ8));
                    acc0 = Avx.Add(acc0, Avx.Multiply(Vector256.Create(d8), t));
                }
                // Row 1
                {
                    Vector256<byte> q5 = ExtractQ5(qs1p, pairIdx, nibbleHalf, qhVec1, bitMask, mask0F);
                    Vector256<int> ps = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(q5, q8Vals), ones);
                    Vector256<float> t = Avx.Subtract(
                        Avx.Multiply(Vector256.Create(d5_1 * sc1[j]), Avx.ConvertToVector256Single(ps)),
                        Avx.Multiply(Vector256.Create(dm1 * mn1[j]), fQ8));
                    acc1 = Avx.Add(acc1, Avx.Multiply(Vector256.Create(d8), t));
                }
                // Row 2
                {
                    Vector256<byte> q5 = ExtractQ5(qs2p, pairIdx, nibbleHalf, qhVec2, bitMask, mask0F);
                    Vector256<int> ps = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(q5, q8Vals), ones);
                    Vector256<float> t = Avx.Subtract(
                        Avx.Multiply(Vector256.Create(d5_2 * sc2[j]), Avx.ConvertToVector256Single(ps)),
                        Avx.Multiply(Vector256.Create(dm2 * mn2[j]), fQ8));
                    acc2 = Avx.Add(acc2, Avx.Multiply(Vector256.Create(d8), t));
                }
                // Row 3
                {
                    Vector256<byte> q5 = ExtractQ5(qs3p, pairIdx, nibbleHalf, qhVec3, bitMask, mask0F);
                    Vector256<int> ps = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(q5, q8Vals), ones);
                    Vector256<float> t = Avx.Subtract(
                        Avx.Multiply(Vector256.Create(d5_3 * sc3[j]), Avx.ConvertToVector256Single(ps)),
                        Avx.Multiply(Vector256.Create(dm3 * mn3[j]), fQ8));
                    acc3 = Avx.Add(acc3, Avx.Multiply(Vector256.Create(d8), t));
                }
            }

            w0 += Q5_K_BlockBytes; w1 += Q5_K_BlockBytes;
            w2 += Q5_K_BlockBytes; w3 += Q5_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        results[0] = HorizontalSumAvx2Float(acc0);
        results[1] = HorizontalSumAvx2Float(acc1);
        results[2] = HorizontalSumAvx2Float(acc2);
        results[3] = HorizontalSumAvx2Float(acc3);
    }

    /// <summary>
    /// True 4-row Q6_K × Q8_K with shared activation loading.
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ6_K_Q8_K_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* q8k, int superBlockCount, float* results)
    {
        if (!Avx2.IsSupported)
        {
            results[0] = VecDotQ6_K_Q8_KScalar(w0, q8k, superBlockCount);
            results[1] = VecDotQ6_K_Q8_KScalar(w1, q8k, superBlockCount);
            results[2] = VecDotQ6_K_Q8_KScalar(w2, q8k, superBlockCount);
            results[3] = VecDotQ6_K_Q8_KScalar(w3, q8k, superBlockCount);
            return;
        }

        Vector256<float> acc0 = Vector256<float>.Zero, acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero, acc3 = Vector256<float>.Zero;
        Vector256<byte> mask0F = Vector256.Create((byte)0x0F);
        Vector256<byte> mask03 = Vector256.Create((byte)0x03);
        Vector256<short> ones = Vector256.Create((short)1);

        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            sbyte* q8qs = (sbyte*)(q8k + 4);

            // Weight data for 4 rows
            byte* ql0 = w0; byte* qh0 = w0 + 128; sbyte* s0 = (sbyte*)(w0 + 192);
            float d6_0 = (float)Unsafe.ReadUnaligned<Half>(w0 + 208);
            byte* ql1 = w1; byte* qh1 = w1 + 128; sbyte* s1 = (sbyte*)(w1 + 192);
            float d6_1 = (float)Unsafe.ReadUnaligned<Half>(w1 + 208);
            byte* ql2 = w2; byte* qh2 = w2 + 128; sbyte* s2 = (sbyte*)(w2 + 192);
            float d6_2 = (float)Unsafe.ReadUnaligned<Half>(w2 + 208);
            byte* ql3 = w3; byte* qh3 = w3 + 128; sbyte* s3 = (sbyte*)(w3 + 192);
            float d6_3 = (float)Unsafe.ReadUnaligned<Half>(w3 + 208);

            for (int sub = 0; sub < 16; sub += 2)
            {
                // Shared: load 32 Q8_K values ONCE
                Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8qs + sub * 16);
                Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), q8Vals);
                Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);

                int half = sub / 8;
                int sh = sub % 8;
                int qlBase = half * 64 + (sh % 4) * 16;
                bool isUpper = sh >= 4;
                int qhBase = half * 32;
                int qhShift = (sh / 2) * 2;

                // Helper: extract Q6 values and compute dot for one row
                static void ProcessQ6Row(
                    byte* ql, byte* qh, sbyte* scales, int sub, float d6, float d8,
                    int qlBase, bool isUpper, int qhBase, int qhShift,
                    Vector256<sbyte> q8V, Vector256<int> q8S,
                    Vector256<byte> m0F, Vector256<byte> m03, Vector256<short> o1s,
                    ref Vector256<float> accum)
                {
                    Vector256<byte> qlRaw = Unsafe.ReadUnaligned<Vector256<byte>>(ql + qlBase);
                    Vector256<byte> nib = isUpper
                        ? Avx2.And(Avx2.ShiftRightLogical(qlRaw.AsUInt16(), 4).AsByte(), m0F)
                        : Avx2.And(qlRaw, m0F);

                    Vector256<byte> qhV = Unsafe.ReadUnaligned<Vector256<byte>>(qh + qhBase);
                    Vector256<byte> qhB = qhShift switch
                    {
                        0 => Avx2.And(qhV, m03),
                        2 => Avx2.And(Avx2.ShiftRightLogical(qhV.AsUInt16(), 2).AsByte(), m03),
                        4 => Avx2.And(Avx2.ShiftRightLogical(qhV.AsUInt16(), 4).AsByte(), m03),
                        _ => Avx2.And(Avx2.ShiftRightLogical(qhV.AsUInt16(), 6).AsByte(), m03),
                    };

                    Vector256<byte> q6u = Avx2.And(
                        Avx2.Or(nib, Avx2.ShiftLeftLogical(qhB.AsUInt16(), 4).AsByte()),
                        Vector256.Create((byte)0x3F));

                    Vector256<short> prod = Avx2.MultiplyAddAdjacent(q6u, q8V);
                    Vector256<int> prodSum = Avx2.MultiplyAddAdjacent(prod, o1s);

                    float sc0 = d6 * scales[sub];
                    float sc1 = d6 * scales[sub + 1];
                    Vector128<float> bias = Vector128.Create(32f);

                    Vector128<float> tLo = Sse.Multiply(Vector128.Create(sc0),
                        Sse.Subtract(Avx.ConvertToVector128Single(prodSum.GetLower()),
                            Sse.Multiply(bias, Avx.ConvertToVector128Single(q8S.GetLower()))));
                    Vector128<float> tHi = Sse.Multiply(Vector128.Create(sc1),
                        Sse.Subtract(Avx.ConvertToVector128Single(prodSum.GetUpper()),
                            Sse.Multiply(bias, Avx.ConvertToVector128Single(q8S.GetUpper()))));

                    accum = Avx.Add(accum, Avx.Multiply(Vector256.Create(d8),
                        Vector256.Create(tLo, tHi)));
                }

                ProcessQ6Row(ql0, qh0, s0, sub, d6_0, d8, qlBase, isUpper, qhBase, qhShift,
                    q8Vals, q8Sum, mask0F, mask03, ones, ref acc0);
                ProcessQ6Row(ql1, qh1, s1, sub, d6_1, d8, qlBase, isUpper, qhBase, qhShift,
                    q8Vals, q8Sum, mask0F, mask03, ones, ref acc1);
                ProcessQ6Row(ql2, qh2, s2, sub, d6_2, d8, qlBase, isUpper, qhBase, qhShift,
                    q8Vals, q8Sum, mask0F, mask03, ones, ref acc2);
                ProcessQ6Row(ql3, qh3, s3, sub, d6_3, d8, qlBase, isUpper, qhBase, qhShift,
                    q8Vals, q8Sum, mask0F, mask03, ones, ref acc3);
            }

            w0 += Q6_K_BlockBytes; w1 += Q6_K_BlockBytes;
            w2 += Q6_K_BlockBytes; w3 += Q6_K_BlockBytes;
            q8k += Q8_K_BlockBytes;
        }

        results[0] = HorizontalSumAvx2Float(acc0);
        results[1] = HorizontalSumAvx2Float(acc1);
        results[2] = HorizontalSumAvx2Float(acc2);
        results[3] = HorizontalSumAvx2Float(acc3);
    }

    // ──────────────────── ComputeRows for K-quants ────────────────────

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ4_K(byte* weights, byte* xQ8K, float* result, int m, int superBlockCount)
    {
        int rowBytes = superBlockCount * Q4_K_BlockBytes;

        int row = 0;
        for (; row + 3 < m; row += 4)
        {
            VecDotQ4_K_Q8_K_4Rows(
                weights + (long)row * rowBytes,
                weights + (long)(row + 1) * rowBytes,
                weights + (long)(row + 2) * rowBytes,
                weights + (long)(row + 3) * rowBytes,
                xQ8K, superBlockCount, result + row);
        }
        if (Avx2.IsSupported)
        {
            for (; row < m; row++)
                result[row] = VecDotQ4_K_Q8_KAvx2(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
        else
        {
            for (; row < m; row++)
                result[row] = VecDotQ4_K_Q8_KScalar(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ5_K(byte* weights, byte* xQ8K, float* result, int m, int superBlockCount)
    {
        int rowBytes = superBlockCount * Q5_K_BlockBytes;

        int row = 0;
        for (; row + 3 < m; row += 4)
        {
            VecDotQ5_K_Q8_K_4Rows(
                weights + (long)row * rowBytes,
                weights + (long)(row + 1) * rowBytes,
                weights + (long)(row + 2) * rowBytes,
                weights + (long)(row + 3) * rowBytes,
                xQ8K, superBlockCount, result + row);
        }
        if (Avx2.IsSupported)
        {
            for (; row < m; row++)
                result[row] = VecDotQ5_K_Q8_KAvx2(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
        else
        {
            for (; row < m; row++)
                result[row] = VecDotQ5_K_Q8_KScalar(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ6_K(byte* weights, byte* xQ8K, float* result, int m, int superBlockCount)
    {
        int rowBytes = superBlockCount * Q6_K_BlockBytes;

        int row = 0;
        for (; row + 3 < m; row += 4)
        {
            VecDotQ6_K_Q8_K_4Rows(
                weights + (long)row * rowBytes,
                weights + (long)(row + 1) * rowBytes,
                weights + (long)(row + 2) * rowBytes,
                weights + (long)(row + 3) * rowBytes,
                xQ8K, superBlockCount, result + row);
        }
        if (Avx2.IsSupported)
        {
            for (; row < m; row++)
                result[row] = VecDotQ6_K_Q8_KAvx2(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
        else
        {
            for (; row < m; row++)
                result[row] = VecDotQ6_K_Q8_KScalar(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        }
    }

    // ──────────────────── Gemv for K-quants ────────────────────

    /// <summary>
    /// Q4_K GEMV: weights[M,K] in Q4_K × f32 input[K] → f32 output[M].
    /// Quantizes input to Q8_K, then uses fused Q4_K × Q8_K vec_dot.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ4_K(byte* weights, float* x, float* result, int m, int k)
    {
        if (k % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {KQuantGroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_K_GroupSize;      // Q8_K blocks for quantizing input
        int superBlockCount = k / KQuantGroupSize;
        int xQ8Bytes = blockCount * Q8_K_BlockBytes;

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
                QuantizeF32ToQ8_K(x, xQ8, k);
                ComputeRowsQ4_K(weights, xQ8, result, m, superBlockCount);
            }
            ArrayPool<byte>.Shared.Return(rented);
            return;
        }

        QuantizeF32ToQ8_K(x, xQ8, k);
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

        int blockCount = k / Q8_K_GroupSize;
        int superBlockCount = k / KQuantGroupSize;
        int xQ8Bytes = blockCount * Q8_K_BlockBytes;

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
                QuantizeF32ToQ8_K(x, xQ8, k);
                ComputeRowsQ5_K(weights, xQ8, result, m, superBlockCount);
            }
            ArrayPool<byte>.Shared.Return(rented);
            return;
        }

        QuantizeF32ToQ8_K(x, xQ8, k);
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

        int blockCount = k / Q8_K_GroupSize;
        int superBlockCount = k / KQuantGroupSize;
        int xQ8Bytes = blockCount * Q8_K_BlockBytes;

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
                QuantizeF32ToQ8_K(x, xQ8, k);
                ComputeRowsQ6_K(weights, xQ8, result, m, superBlockCount);
            }
            ArrayPool<byte>.Shared.Return(rented);
            return;
        }

        QuantizeF32ToQ8_K(x, xQ8, k);
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

        int q8BlockCount = k / Q8_K_GroupSize;
        int superBlockCount = k / KQuantGroupSize;
        int q8RowBytes = q8BlockCount * Q8_K_BlockBytes;
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
                int xQ8Bytes = q8BlockCount * Q8_K_BlockBytes;
                byte[] rented = ArrayPool<byte>.Shared.Rent(xQ8Bytes);
                fixed (byte* xQ8 = rented)
                {
                    QuantizeF32ToQ8_K(b, xQ8, k);
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
                QuantizeF32ToQ8_K(b + t * k, inputQ8 + t * q8RowBytes, k);

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

        int q8BlockCount = k / Q8_K_GroupSize;
        int superBlockCount = k / KQuantGroupSize;
        int xQ8Bytes = q8BlockCount * Q8_K_BlockBytes;

        // Quantize x once into pool scratch for thread 0
        byte* xQ8 = (byte*)pool.GetWorkerScratch(0, xQ8Bytes);
        QuantizeF32ToQ8_K(x, xQ8, k);

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

        int q8BlockCount = k / Q8_K_GroupSize;
        int superBlockCount2 = k / KQuantGroupSize;
        int q8RowBytes = q8BlockCount * Q8_K_BlockBytes;
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
                QuantizeF32ToQ8_K(b + t * k, inputQ8 + t * q8RowBytes, k);

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
