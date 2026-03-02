using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Matrix-vector multiplication kernels for transformer inference.
/// Supports f32 weights and Q8_0 quantized weights with on-the-fly activation quantization.
/// </summary>
public static unsafe class MatMul
{
    /// <summary>Q8_0 block size in bytes: 2 (Half scale) + 32 (sbyte quantized values).</summary>
    private const int Q8_0BlockBytes = 34;

    /// <summary>Number of elements per Q8_0 block.</summary>
    private const int Q8_0GroupSize = 32;

    /// <summary>Stackalloc threshold in bytes. Above this, use ArrayPool.</summary>
    private const int StackAllocThreshold = 8192;

    /// <summary>
    /// f32 GEMV: <c>result[m] = dot(A[m,:], x)</c>.
    /// A is [M,K] row-major, x is [K], result is [M].
    /// </summary>
    /// <param name="a">Pointer to weight matrix A [M×K], row-major.</param>
    /// <param name="x">Pointer to input vector x [K].</param>
    /// <param name="result">Pointer to output vector [M].</param>
    /// <param name="m">Number of rows in A (output dimension).</param>
    /// <param name="k">Number of columns in A (input dimension).</param>
    [SkipLocalsInit]
    public static void GemvF32(float* a, float* x, float* result, int m, int k)
    {
        var xSpan = new ReadOnlySpan<float>(x, k);

        for (int row = 0; row < m; row++)
        {
            var rowSpan = new ReadOnlySpan<float>(a + row * k, k);
            result[row] = TensorPrimitives.Dot(rowSpan, xSpan);
        }
    }

    /// <summary>
    /// Scalar f32 GEMV reference implementation for correctness verification.
    /// </summary>
    [SkipLocalsInit]
    internal static void GemvF32Scalar(float* a, float* x, float* result, int m, int k)
    {
        for (int row = 0; row < m; row++)
        {
            float sum = 0;
            float* rowPtr = a + row * k;
            for (int j = 0; j < k; j++)
                sum += rowPtr[j] * x[j];
            result[row] = sum;
        }
    }

    /// <summary>
    /// Q8_0 GEMV: A is Q8_0 [M,K], x is f32 [K].
    /// Quantizes x to Q8_0 on-the-fly, then uses Q8_0×Q8_0 VecDot per row.
    /// </summary>
    /// <param name="weightsQ8">Pointer to Q8_0 weight data. Each row is K/32 blocks of 34 bytes.</param>
    /// <param name="x">Pointer to f32 input vector [K].</param>
    /// <param name="result">Pointer to f32 output vector [M].</param>
    /// <param name="m">Number of rows (output dimension).</param>
    /// <param name="k">Number of columns (input dimension). Must be a multiple of 32.</param>
    [SkipLocalsInit]
    public static void GemvQ8_0(byte* weightsQ8, float* x, float* result, int m, int k)
    {
        if (k % Q8_0GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_0GroupSize;
        int xQ8Bytes = blockCount * Q8_0BlockBytes;

        // Quantize the activation vector once.
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
                ComputeRows(weightsQ8, xQ8, result, m, blockCount);
            }
            ArrayPool<byte>.Shared.Return(rented);
            return;
        }

        QuantizeF32ToQ8_0(x, xQ8, k);
        ComputeRows(weightsQ8, xQ8, result, m, blockCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeRows(byte* weightsQ8, byte* xQ8, float* result, int m, int blockCount)
    {
        int rowBytes = blockCount * Q8_0BlockBytes;

        if (Avx2.IsSupported)
        {
            for (int row = 0; row < m; row++)
            {
                result[row] = VecDotQ8_0Avx2(weightsQ8 + row * rowBytes, xQ8, blockCount);
            }
        }
        else
        {
            for (int row = 0; row < m; row++)
            {
                result[row] = VecDotQ8_0Scalar(weightsQ8 + row * rowBytes, xQ8, blockCount);
            }
        }
    }

    /// <summary>
    /// Scalar Q8_0 dot product: sum over blocks of (da * db * sum(qa[i] * qb[i])).
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ8_0Scalar(byte* a, byte* b, int blockCount)
    {
        float sumf = 0;

        for (int block = 0; block < blockCount; block++)
        {
            byte* aBlock = a + block * Q8_0BlockBytes;
            byte* bBlock = b + block * Q8_0BlockBytes;

            float da = (float)Unsafe.ReadUnaligned<Half>(aBlock);
            float db = (float)Unsafe.ReadUnaligned<Half>(bBlock);

            sbyte* qa = (sbyte*)(aBlock + 2);
            sbyte* qb = (sbyte*)(bBlock + 2);

            int sumi = 0;
            for (int i = 0; i < Q8_0GroupSize; i++)
                sumi += qa[i] * qb[i];

            sumf += da * db * sumi;
        }

        return sumf;
    }

    /// <summary>
    /// AVX2-accelerated Q8_0 dot product using the sign-flip trick for signed×signed multiply.
    /// Uses <c>Avx2.Sign</c> to handle <c>MultiplyAddAdjacent(ubyte, sbyte)</c> with two signed operands.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ8_0Avx2(byte* a, byte* b, int blockCount)
    {
        Vector256<int> acc = Vector256<int>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);

        float sumf = 0;

        for (int block = 0; block < blockCount; block++)
        {
            byte* aBlock = a + block * Q8_0BlockBytes;
            byte* bBlock = b + block * Q8_0BlockBytes;

            float da = (float)Unsafe.ReadUnaligned<Half>(aBlock);
            float db = (float)Unsafe.ReadUnaligned<Half>(bBlock);

            // Load 32 signed bytes from each operand.
            Vector256<sbyte> va = Unsafe.ReadUnaligned<Vector256<sbyte>>(aBlock + 2);
            Vector256<sbyte> vb = Unsafe.ReadUnaligned<Vector256<sbyte>>(bBlock + 2);

            // Sign trick: abs(va) is unsigned, sign-flip vb where va is negative.
            // This lets us use MultiplyAddAdjacent(ubyte, sbyte) with two signed inputs.
            Vector256<sbyte> absA = Avx2.Sign(va, va);
            Vector256<sbyte> adjB = Avx2.Sign(vb, va);

            // ubyte × sbyte → int16 pairs
            Vector256<short> prod = Avx2.MultiplyAddAdjacent(absA.AsByte(), adjB);

            // int16 pairs → int32
            Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, ones);

            // Horizontal sum of int32 lanes.
            int dotI = HorizontalSumAvx2(isum);

            sumf += da * db * dotI;
        }

        return sumf;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int HorizontalSumAvx2(Vector256<int> v)
    {
        // Sum upper and lower 128-bit halves.
        Vector128<int> lo = v.GetLower();
        Vector128<int> hi = v.GetUpper();
        Vector128<int> sum128 = Sse2.Add(lo, hi);

        // Shuffle and add to get all 4 ints summed.
        Vector128<int> shuf = Sse2.Shuffle(sum128, 0b_01_00_11_10); // swap pairs
        sum128 = Sse2.Add(sum128, shuf);
        shuf = Sse2.Shuffle(sum128, 0b_00_01_00_01); // swap within pairs
        sum128 = Sse2.Add(sum128, shuf);

        return sum128.ToScalar();
    }

    /// <summary>
    /// Quantizes f32 data to Q8_0 format. Per block of 32 floats:
    /// scale = max(|x[i]|) / 127, qs[i] = round(x[i] / scale) clamped to [-127, 127].
    /// </summary>
    /// <param name="src">Source f32 data. Must have <paramref name="elementCount"/> elements.</param>
    /// <param name="dest">Destination Q8_0 buffer. Must have (elementCount/32) × 34 bytes.</param>
    /// <param name="elementCount">Number of float elements. Must be a multiple of 32.</param>
    [SkipLocalsInit]
    internal static void QuantizeF32ToQ8_0(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / Q8_0GroupSize;

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * Q8_0GroupSize;
            byte* blockDst = dest + block * Q8_0BlockBytes;

            // Find max absolute value.
            float maxAbs = 0;
            for (int i = 0; i < Q8_0GroupSize; i++)
            {
                float abs = MathF.Abs(blockSrc[i]);
                if (abs > maxAbs) maxAbs = abs;
            }

            float scale = maxAbs / 127.0f;
            Unsafe.WriteUnaligned(blockDst, (Half)scale);

            sbyte* qs = (sbyte*)(blockDst + 2);
            if (scale == 0)
            {
                // All zeros.
                for (int i = 0; i < Q8_0GroupSize; i++)
                    qs[i] = 0;
            }
            else
            {
                float invScale = 1.0f / scale;
                for (int i = 0; i < Q8_0GroupSize; i++)
                {
                    int v = (int)MathF.Round(blockSrc[i] * invScale);
                    qs[i] = (sbyte)Math.Clamp(v, -127, 127);
                }
            }
        }
    }
}
