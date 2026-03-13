using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Fused operator kernels that combine adjacent operations to keep intermediate values
/// in L1/L2 cache, eliminating redundant DRAM roundtrips on the decode hot path.
/// </summary>
public static unsafe class FusedOps
{
    // ──────────────────── SwiGLU Fusion ────────────────────
    // Fuses SiLu(gate) + Multiply(siluOut, up) into a tiled operation:
    //   result[i] = gate[i] * sigmoid(gate[i]) * up[i]
    // Uses TensorPrimitives.Sigmoid for bit-exact precision with the unfused path.
    // Tiles of 256 floats (1KB) keep the sigmoid intermediate in L1 cache.

    /// <summary>Tile size for SwiGLU. 256 floats = 1024 bytes — fits in L1 data cache.</summary>
    private const int SwiGLUTileSize = 256;

    /// <summary>
    /// Fused SwiGLU activation: <c>result[i] = gate[i] * sigmoid(gate[i]) * up[i]</c>.
    /// Uses tiled <see cref="TensorPrimitives.Sigmoid"/> for exact precision, with the
    /// sigmoid intermediate staying in L1 via a small stack buffer. Eliminates one full
    /// memory pass over intermediateSize compared to separate SiLU + Multiply calls.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SwiGLU(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> result)
    {
        int length = gate.Length;
        Span<float> sigBuf = stackalloc float[SwiGLUTileSize];

        int i = 0;
        for (; i + SwiGLUTileSize <= length; i += SwiGLUTileSize)
        {
            var gTile = gate.Slice(i, SwiGLUTileSize);
            var uTile = up.Slice(i, SwiGLUTileSize);
            var rTile = result.Slice(i, SwiGLUTileSize);

            TensorPrimitives.Sigmoid(gTile, sigBuf);
            TensorPrimitives.Multiply(gTile, sigBuf, rTile);
            TensorPrimitives.Multiply((ReadOnlySpan<float>)rTile, uTile, rTile);
        }

        // Tail
        if (i < length)
        {
            int remaining = length - i;
            var gTile = gate.Slice(i, remaining);
            var uTile = up.Slice(i, remaining);
            var rTile = result.Slice(i, remaining);
            var sigTail = sigBuf.Slice(0, remaining);

            TensorPrimitives.Sigmoid(gTile, sigTail);
            TensorPrimitives.Multiply(gTile, sigTail, rTile);
            TensorPrimitives.Multiply((ReadOnlySpan<float>)rTile, uTile, rTile);
        }
    }

    /// <summary>
    /// Scalar SwiGLU reference implementation for correctness verification.
    /// </summary>
    [SkipLocalsInit]
    internal static void SwiGLUScalar(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> result)
    {
        for (int i = 0; i < gate.Length; i++)
        {
            float g = gate[i];
            float u = up[i];
            float sigmoid = 1.0f / (1.0f + MathF.Exp(-g));
            result[i] = g * sigmoid * u;
        }
    }

    // ──────────────────── RMSNorm + Quantize Fusion ────────────────────
    // Fuses RmsNorm(hidden → normOut) + Quantize(normOut → Q8 scratch) into one kernel
    // that reads hidden once and writes quantized output directly — skipping normOut.

    private const int Q8_0GroupSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int Q8_1GroupSize = 32;
    private const int Q8_1BlockBytes = 36;
    private const int Q8_K_GroupSize = 256;
    private const int Q8_K_BlockBytes = 292;

    /// <summary>
    /// Dispatches fused RmsNorm+Quantize based on quant type.
    /// Returns the quantized scratch pointer if successful, or null for F32/F16 (fall back to unfused).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte* RmsNormQuantize(float* input, ReadOnlySpan<float> weight, float eps,
                                         byte* dest, int dim, QuantizationType qt)
    {
        if (qt == QuantizationType.Q8_0)
        {
            RmsNormQuantizeQ8_0(input, weight, eps, dest, dim);
            return dest;
        }
        if (qt == QuantizationType.Q5_0)
        {
            RmsNormQuantizeQ8_1(input, weight, eps, dest, dim);
            return dest;
        }
        if (qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K)
        {
            RmsNormQuantizeQ8_K(input, weight, eps, dest, dim);
            return dest;
        }
        return null; // F32/F16: can't pre-quantize
    }

    /// <summary>
    /// Fused RMSNorm + Q8_0 quantization. Reads input once, applies normalization,
    /// and quantizes directly to Q8_0 format (Half scale, 32-element blocks).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RmsNormQuantizeQ8_0(float* input, ReadOnlySpan<float> weight, float eps,
                                            byte* dest, int dim)
    {
        float rmsScale = ComputeRmsScale(input, dim, eps);
        int blockCount = dim / Q8_0GroupSize;
        float* normBuf = stackalloc float[Q8_0GroupSize];

        if (Avx2.IsSupported)
        {
            fixed (float* wPtr = weight)
            {
                Vector256<float> vScale = Vector256.Create(rmsScale);
                Vector256<int> permMask = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);

                for (int block = 0; block < blockCount; block++)
                {
                    float* blockInput = input + block * Q8_0GroupSize;
                    byte* blockDst = dest + block * Q8_0BlockBytes;
                    float* wBlock = wPtr + block * Q8_0GroupSize;

                    // Normalize: input * rmsScale * weight → normBuf (4 × 8-wide)
                    for (int j = 0; j < Q8_0GroupSize; j += 8)
                    {
                        var norm = Avx.Multiply(Avx.Multiply(
                            Avx.LoadVector256(blockInput + j), vScale),
                            Avx.LoadVector256(wBlock + j));
                        Avx.Store(normBuf + j, norm);
                    }

                    // MaxAbs scan from normBuf
                    Vector256<float> v0 = Vector256.Abs(Avx.LoadVector256(normBuf));
                    Vector256<float> v1 = Vector256.Abs(Avx.LoadVector256(normBuf + 8));
                    Vector256<float> v2 = Vector256.Abs(Avx.LoadVector256(normBuf + 16));
                    Vector256<float> v3 = Vector256.Abs(Avx.LoadVector256(normBuf + 24));
                    float maxAbs = HorizontalMaxAvx2(Avx.Max(Avx.Max(v0, v1), Avx.Max(v2, v3)));

                    float scale = maxAbs / 127.0f;
                    Unsafe.WriteUnaligned(blockDst, (Half)scale);

                    sbyte* qs = (sbyte*)(blockDst + 2);
                    if (scale == 0)
                    {
                        Vector256<sbyte>.Zero.StoreUnsafe(ref Unsafe.AsRef<sbyte>(qs));
                    }
                    else
                    {
                        Vector256<float> vInvScale = Vector256.Create(1.0f / scale);

                        Vector256<int> i0 = Avx.ConvertToVector256Int32(
                            Avx.Multiply(Avx.LoadVector256(normBuf), vInvScale));
                        Vector256<int> i1 = Avx.ConvertToVector256Int32(
                            Avx.Multiply(Avx.LoadVector256(normBuf + 8), vInvScale));
                        Vector256<int> i2 = Avx.ConvertToVector256Int32(
                            Avx.Multiply(Avx.LoadVector256(normBuf + 16), vInvScale));
                        Vector256<int> i3 = Avx.ConvertToVector256Int32(
                            Avx.Multiply(Avx.LoadVector256(normBuf + 24), vInvScale));

                        Vector256<short> s01 = Avx2.PackSignedSaturate(i0, i1);
                        Vector256<short> s23 = Avx2.PackSignedSaturate(i2, i3);
                        Vector256<sbyte> packed = Avx2.PackSignedSaturate(s01, s23);

                        Vector256<int> permuted = Avx2.PermuteVar8x32(packed.AsInt32(), permMask);
                        permuted.AsByte().StoreUnsafe(ref Unsafe.AsRef<byte>((byte*)qs));
                    }
                }
            }
        }
        else
        {
            // Scalar fallback
            ref float wRef = ref MemoryMarshal.GetReference(weight);

            for (int block = 0; block < blockCount; block++)
            {
                float* blockInput = input + block * Q8_0GroupSize;
                byte* blockDst = dest + block * Q8_0BlockBytes;
                int wOff = block * Q8_0GroupSize;

                float maxAbs = 0;
                for (int i = 0; i < Q8_0GroupSize; i++)
                {
                    float normalized = blockInput[i] * rmsScale * Unsafe.Add(ref wRef, wOff + i);
                    normBuf[i] = normalized;
                    float abs = MathF.Abs(normalized);
                    if (abs > maxAbs) maxAbs = abs;
                }

                float scale = maxAbs / 127.0f;
                Unsafe.WriteUnaligned(blockDst, (Half)scale);

                sbyte* qs = (sbyte*)(blockDst + 2);
                if (scale == 0)
                {
                    for (int i = 0; i < Q8_0GroupSize; i++)
                        qs[i] = 0;
                }
                else
                {
                    float invScale = 1.0f / scale;
                    for (int i = 0; i < Q8_0GroupSize; i++)
                    {
                        int v = (int)MathF.Round(normBuf[i] * invScale);
                        qs[i] = (sbyte)Math.Clamp(v, -127, 127);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Fused RMSNorm + Q8_1 quantization. Reads input once, applies normalization,
    /// and quantizes directly to Q8_1 format (Half d + Half s, 32-element blocks with block sum).
    /// Used when the weight quant type is Q5_0 (which requires Q8_1 input quantization).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RmsNormQuantizeQ8_1(float* input, ReadOnlySpan<float> weight, float eps,
                                            byte* dest, int dim)
    {
        float rmsScale = ComputeRmsScale(input, dim, eps);
        int blockCount = dim / Q8_1GroupSize;
        float* normBuf = stackalloc float[Q8_1GroupSize];

        if (Avx2.IsSupported)
        {
            fixed (float* wPtr = weight)
            {
                Vector256<float> vScale = Vector256.Create(rmsScale);
                Vector256<int> permMask = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);

                for (int block = 0; block < blockCount; block++)
                {
                    float* blockInput = input + block * Q8_1GroupSize;
                    byte* blockDst = dest + block * Q8_1BlockBytes;
                    float* wBlock = wPtr + block * Q8_1GroupSize;

                    // Normalize: input * rmsScale * weight → normBuf
                    for (int j = 0; j < Q8_1GroupSize; j += 8)
                    {
                        var norm = Avx.Multiply(Avx.Multiply(
                            Avx.LoadVector256(blockInput + j), vScale),
                            Avx.LoadVector256(wBlock + j));
                        Avx.Store(normBuf + j, norm);
                    }

                    // MaxAbs scan
                    Vector256<float> v0 = Vector256.Abs(Avx.LoadVector256(normBuf));
                    Vector256<float> v1 = Vector256.Abs(Avx.LoadVector256(normBuf + 8));
                    Vector256<float> v2 = Vector256.Abs(Avx.LoadVector256(normBuf + 16));
                    Vector256<float> v3 = Vector256.Abs(Avx.LoadVector256(normBuf + 24));
                    float maxAbs = HorizontalMaxAvx2(Avx.Max(Avx.Max(v0, v1), Avx.Max(v2, v3)));

                    float scale = maxAbs / 127.0f;
                    Unsafe.WriteUnaligned(blockDst, (Half)scale);

                    sbyte* qs = (sbyte*)(blockDst + 4);
                    if (scale == 0)
                    {
                        Vector256<sbyte>.Zero.StoreUnsafe(ref Unsafe.AsRef<sbyte>(qs));
                        Unsafe.WriteUnaligned(blockDst + 2, (Half)0f);
                    }
                    else
                    {
                        Vector256<float> vInvScale = Vector256.Create(1.0f / scale);

                        Vector256<int> i0 = Avx.ConvertToVector256Int32(
                            Avx.Multiply(Avx.LoadVector256(normBuf), vInvScale));
                        Vector256<int> i1 = Avx.ConvertToVector256Int32(
                            Avx.Multiply(Avx.LoadVector256(normBuf + 8), vInvScale));
                        Vector256<int> i2 = Avx.ConvertToVector256Int32(
                            Avx.Multiply(Avx.LoadVector256(normBuf + 16), vInvScale));
                        Vector256<int> i3 = Avx.ConvertToVector256Int32(
                            Avx.Multiply(Avx.LoadVector256(normBuf + 24), vInvScale));

                        // Clamp to [-127, 127] before summing
                        Vector256<int> clampMin = Vector256.Create(-127);
                        Vector256<int> clampMax = Vector256.Create(127);
                        i0 = Avx2.Min(Avx2.Max(i0, clampMin), clampMax);
                        i1 = Avx2.Min(Avx2.Max(i1, clampMin), clampMax);
                        i2 = Avx2.Min(Avx2.Max(i2, clampMin), clampMax);
                        i3 = Avx2.Min(Avx2.Max(i3, clampMin), clampMax);

                        // Sum all 32 int32 values
                        Vector256<int> isum = Avx2.Add(Avx2.Add(i0, i1), Avx2.Add(i2, i3));
                        int sum = Vector256.Sum(isum);
                        Unsafe.WriteUnaligned(blockDst + 2, (Half)(scale * sum));

                        // Pack int32 → int16 → int8
                        Vector256<short> s01 = Avx2.PackSignedSaturate(i0, i1);
                        Vector256<short> s23 = Avx2.PackSignedSaturate(i2, i3);
                        Vector256<sbyte> packed = Avx2.PackSignedSaturate(s01, s23);

                        Vector256<int> permuted = Avx2.PermuteVar8x32(packed.AsInt32(), permMask);
                        permuted.AsByte().StoreUnsafe(ref Unsafe.AsRef<byte>((byte*)qs));
                    }
                }
            }
        }
        else
        {
            // Scalar fallback
            ref float wRef = ref MemoryMarshal.GetReference(weight);

            for (int block = 0; block < blockCount; block++)
            {
                float* blockInput = input + block * Q8_1GroupSize;
                byte* blockDst = dest + block * Q8_1BlockBytes;
                int wOff = block * Q8_1GroupSize;

                float maxAbs = 0;
                for (int i = 0; i < Q8_1GroupSize; i++)
                {
                    float normalized = blockInput[i] * rmsScale * Unsafe.Add(ref wRef, wOff + i);
                    normBuf[i] = normalized;
                    float abs = MathF.Abs(normalized);
                    if (abs > maxAbs) maxAbs = abs;
                }

                float scale = maxAbs / 127.0f;
                Unsafe.WriteUnaligned(blockDst, (Half)scale);

                sbyte* qs = (sbyte*)(blockDst + 4);
                if (scale == 0)
                {
                    for (int i = 0; i < Q8_1GroupSize; i++)
                        qs[i] = 0;
                    Unsafe.WriteUnaligned(blockDst + 2, (Half)0f);
                }
                else
                {
                    float invScale = 1.0f / scale;
                    int sum = 0;
                    for (int i = 0; i < Q8_1GroupSize; i++)
                    {
                        int v = (int)MathF.Round(normBuf[i] * invScale);
                        v = Math.Clamp(v, -127, 127);
                        qs[i] = (sbyte)v;
                        sum += v;
                    }
                    Unsafe.WriteUnaligned(blockDst + 2, (Half)(scale * sum));
                }
            }
        }
    }

    /// <summary>
    /// Fused RMSNorm + Q8_K quantization. Reads input once, applies normalization,
    /// and quantizes directly to Q8_K format (float32 scale, 256-element blocks with 16 bsums).
    /// Used for K-quant weight types (Q4_K, Q5_K, Q6_K).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RmsNormQuantizeQ8_K(float* input, ReadOnlySpan<float> weight, float eps,
                                            byte* dest, int dim)
    {
        float rmsScale = ComputeRmsScale(input, dim, eps);
        int blockCount = dim / Q8_K_GroupSize;
        float* normBuf = stackalloc float[Q8_K_GroupSize];

        if (Avx2.IsSupported)
        {
            fixed (float* wPtr = weight)
            {
                Vector256<float> vScale = Vector256.Create(rmsScale);
                Vector256<int> permMask = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);

                for (int block = 0; block < blockCount; block++)
                {
                    float* blockInput = input + block * Q8_K_GroupSize;
                    byte* blockDst = dest + block * Q8_K_BlockBytes;
                    float* wBlock = wPtr + block * Q8_K_GroupSize;

                    // Normalize 256 floats: input * rmsScale * weight → normBuf (32 × 8-wide)
                    // Interleave with maxAbs accumulation to avoid a second pass over normBuf
                    Vector256<float> maxVec = Vector256<float>.Zero;
                    for (int j = 0; j < Q8_K_GroupSize; j += 8)
                    {
                        var norm = Avx.Multiply(Avx.Multiply(
                            Avx.LoadVector256(blockInput + j), vScale),
                            Avx.LoadVector256(wBlock + j));
                        Avx.Store(normBuf + j, norm);
                        maxVec = Avx.Max(maxVec, Vector256.Abs(norm));
                    }
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
                            float* chunkSrc = normBuf + chunk * 32;
                            sbyte* chunkDst = qs + chunk * 32;

                            Vector256<int> i0 = Avx.ConvertToVector256Int32(
                                Avx.Multiply(Avx.LoadVector256(chunkSrc), vInvScale));
                            Vector256<int> i1 = Avx.ConvertToVector256Int32(
                                Avx.Multiply(Avx.LoadVector256(chunkSrc + 8), vInvScale));
                            Vector256<int> i2 = Avx.ConvertToVector256Int32(
                                Avx.Multiply(Avx.LoadVector256(chunkSrc + 16), vInvScale));
                            Vector256<int> i3 = Avx.ConvertToVector256Int32(
                                Avx.Multiply(Avx.LoadVector256(chunkSrc + 24), vInvScale));

                            Vector256<short> s01 = Avx2.PackSignedSaturate(i0, i1);
                            Vector256<short> s23 = Avx2.PackSignedSaturate(i2, i3);
                            Vector256<sbyte> packed = Avx2.PackSignedSaturate(s01, s23);

                            Vector256<int> permuted = Avx2.PermuteVar8x32(packed.AsInt32(), permMask);
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
        }
        else
        {
            // Scalar fallback
            ref float wRef = ref MemoryMarshal.GetReference(weight);

            for (int block = 0; block < blockCount; block++)
            {
                float* blockInput = input + block * Q8_K_GroupSize;
                byte* blockDst = dest + block * Q8_K_BlockBytes;
                int wOff = block * Q8_K_GroupSize;

                float maxAbs = 0;
                for (int i = 0; i < Q8_K_GroupSize; i++)
                {
                    float normalized = blockInput[i] * rmsScale * Unsafe.Add(ref wRef, wOff + i);
                    normBuf[i] = normalized;
                    float abs = MathF.Abs(normalized);
                    if (abs > maxAbs) maxAbs = abs;
                }

                float scale = maxAbs / 127.0f;
                Unsafe.WriteUnaligned(blockDst, scale);

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
                        for (int j = 0; j < 16; j++)
                        {
                            int idx = g * 16 + j;
                            int v = (int)MathF.Round(normBuf[idx] * invScale);
                            sbyte qv = (sbyte)Math.Clamp(v, -127, 127);
                            qs[idx] = qv;
                            sum += qv;
                        }
                        bsums[g] = (short)sum;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Reduces a 256-bit float vector to its horizontal maximum (scalar).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMaxAvx2(Vector256<float> v)
    {
        Vector128<float> lo = v.GetLower();
        Vector128<float> hi = v.GetUpper();
        Vector128<float> max128 = Sse.Max(lo, hi);

        Vector128<float> shuf = Sse.MoveHighToLow(max128, max128);
        max128 = Sse.Max(max128, shuf);
        shuf = Sse.Shuffle(max128, max128, 0b_00_01_00_01);
        max128 = Sse.Max(max128, shuf);

        return max128.ToScalar();
    }

    /// <summary>
    /// Computes the RMS normalization scale factor: <c>1 / sqrt(mean(x²) + eps)</c>.
    /// Uses TensorPrimitives.SumOfSquares for SIMD-accelerated sum.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ComputeRmsScale(float* input, int dim, float eps)
    {
        float sumSq = TensorPrimitives.SumOfSquares(new ReadOnlySpan<float>(input, dim));
        return 1.0f / MathF.Sqrt(sumSq / dim + eps);
    }
}
