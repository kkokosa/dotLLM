using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
        // Pass 1: compute RMS scale
        float rmsScale = ComputeRmsScale(input, dim, eps);

        // Pass 2: normalize + quantize per block
        ref float wRef = ref MemoryMarshal.GetReference(weight);
        int blockCount = dim / Q8_0GroupSize;

        // Stack buffer for one block of normalized floats (32 × 4B = 128B — fits in L1)
        float* normBuf = stackalloc float[Q8_0GroupSize];

        for (int block = 0; block < blockCount; block++)
        {
            float* blockInput = input + block * Q8_0GroupSize;
            byte* blockDst = dest + block * Q8_0BlockBytes;
            int wOff = block * Q8_0GroupSize;

            // Normalize into stack buffer
            float maxAbs = 0;
            for (int i = 0; i < Q8_0GroupSize; i++)
            {
                float normalized = blockInput[i] * rmsScale * Unsafe.Add(ref wRef, wOff + i);
                normBuf[i] = normalized;
                float abs = MathF.Abs(normalized);
                if (abs > maxAbs) maxAbs = abs;
            }

            // Quantize from stack buffer
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

        ref float wRef = ref MemoryMarshal.GetReference(weight);
        int blockCount = dim / Q8_1GroupSize;

        float* normBuf = stackalloc float[Q8_1GroupSize];

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

            sbyte* qs = (sbyte*)(blockDst + 4); // Q8_1 has d(2) + s(2) before qs

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

        ref float wRef = ref MemoryMarshal.GetReference(weight);
        int blockCount = dim / Q8_K_GroupSize;

        // Stack buffer for one super-block (256 × 4B = 1024B — fits in L1)
        float* normBuf = stackalloc float[Q8_K_GroupSize];

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
            Unsafe.WriteUnaligned(blockDst, scale); // float32, not Half

            sbyte* qs = (sbyte*)(blockDst + 4);
            short* bsums = (short*)(blockDst + 260); // 4 + 256 = 260

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
