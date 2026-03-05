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
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void ComputeRows(byte* weightsQ8, byte* xQ8, float* result, int m, int blockCount)
    {
        int rowBytes = blockCount * Q8_0BlockBytes;

        if (Avx512BW.IsSupported)
        {
            int row = 0;
            // Process 4 rows at a time for cache efficiency.
            for (; row + 3 < m; row += 4)
            {
                VecDotQ8_0Avx512_4Rows(
                    weightsQ8 + row * rowBytes,
                    weightsQ8 + (row + 1) * rowBytes,
                    weightsQ8 + (row + 2) * rowBytes,
                    weightsQ8 + (row + 3) * rowBytes,
                    xQ8, blockCount, result + row);
            }
            for (; row < m; row++)
            {
                result[row] = VecDotQ8_0Avx512(weightsQ8 + row * rowBytes, xQ8, blockCount);
            }
        }
        else if (Avx2.IsSupported)
        {
            int row = 0;
            // Process 4 rows at a time for cache efficiency.
            for (; row + 3 < m; row += 4)
            {
                VecDotQ8_0Avx2_4Rows(
                    weightsQ8 + row * rowBytes,
                    weightsQ8 + (row + 1) * rowBytes,
                    weightsQ8 + (row + 2) * rowBytes,
                    weightsQ8 + (row + 3) * rowBytes,
                    xQ8, blockCount, result + row);
            }
            for (; row < m; row++)
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

    // ──────────────────── Scalar reference ────────────────────

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

    // ──────────────────── AVX2 optimized ────────────────────

    /// <summary>
    /// AVX2-accelerated Q8_0 dot product with FMA float accumulation.
    /// Uses the sign-flip trick for signed×signed multiply and accumulates
    /// in <c>Vector256&lt;float&gt;</c> across all blocks, performing a single
    /// horizontal sum at the end.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ8_0Avx2(byte* a, byte* b, int blockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);

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
            Vector256<sbyte> absA = Avx2.Sign(va, va);
            Vector256<sbyte> adjB = Avx2.Sign(vb, va);

            // ubyte × sbyte → int16 pairs
            Vector256<short> prod = Avx2.MultiplyAddAdjacent(absA.AsByte(), adjB);

            // int16 pairs → int32
            Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, ones);

            // Convert to float, scale, and accumulate.
            Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
            Vector256<float> scale = Vector256.Create(da * db);

            acc += fsum * scale;
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>
    /// AVX2 multi-row (4 rows) Q8_0 dot product. Loads each xQ8 block once and
    /// computes dot products against 4 weight rows simultaneously.
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ8_0Avx2_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* x, int blockCount, float* results)
    {
        Vector256<float> acc0 = Vector256<float>.Zero;
        Vector256<float> acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero;
        Vector256<float> acc3 = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);

        for (int block = 0; block < blockCount; block++)
        {
            byte* xBlock = x + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);

            // Load x data once per block.
            Vector256<sbyte> vx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);
            Vector256<sbyte> absX = Avx2.Sign(vx, vx);

            // Row 0
            {
                byte* wBlock = w0 + block * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<sbyte> adjW = Avx2.Sign(vw, vx);
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(absX.AsByte(), adjW);
                Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, ones);
                Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
                Vector256<float> scale = Vector256.Create(dx * dw);
                acc0 += fsum * scale;
            }

            // Row 1
            {
                byte* wBlock = w1 + block * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<sbyte> adjW = Avx2.Sign(vw, vx);
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(absX.AsByte(), adjW);
                Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, ones);
                Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
                Vector256<float> scale = Vector256.Create(dx * dw);
                acc1 += fsum * scale;
            }

            // Row 2
            {
                byte* wBlock = w2 + block * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<sbyte> adjW = Avx2.Sign(vw, vx);
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(absX.AsByte(), adjW);
                Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, ones);
                Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
                Vector256<float> scale = Vector256.Create(dx * dw);
                acc2 += fsum * scale;
            }

            // Row 3
            {
                byte* wBlock = w3 + block * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<sbyte> adjW = Avx2.Sign(vw, vx);
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(absX.AsByte(), adjW);
                Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, ones);
                Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
                Vector256<float> scale = Vector256.Create(dx * dw);
                acc3 += fsum * scale;
            }
        }

        results[0] = HorizontalSumAvx2Float(acc0);
        results[1] = HorizontalSumAvx2Float(acc1);
        results[2] = HorizontalSumAvx2Float(acc2);
        results[3] = HorizontalSumAvx2Float(acc3);
    }

    // ──────────────────── AVX-512 optimized ────────────────────

    /// <summary>
    /// AVX-512 accelerated Q8_0 dot product. Processes 2 blocks (64 bytes) per iteration
    /// using <c>Vector512</c>. The sign trick is performed on 256-bit halves since
    /// <c>Avx2.Sign</c> has no 512-bit equivalent.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ8_0Avx512(byte* a, byte* b, int blockCount)
    {
        Vector512<float> acc = Vector512<float>.Zero;
        Vector256<short> ones256 = Vector256.Create((short)1);

        int block = 0;

        // Process 2 blocks per iteration.
        for (; block + 1 < blockCount; block += 2)
        {
            byte* aBlock0 = a + block * Q8_0BlockBytes;
            byte* bBlock0 = b + block * Q8_0BlockBytes;
            byte* aBlock1 = a + (block + 1) * Q8_0BlockBytes;
            byte* bBlock1 = b + (block + 1) * Q8_0BlockBytes;

            float da0 = (float)Unsafe.ReadUnaligned<Half>(aBlock0);
            float db0 = (float)Unsafe.ReadUnaligned<Half>(bBlock0);
            float da1 = (float)Unsafe.ReadUnaligned<Half>(aBlock1);
            float db1 = (float)Unsafe.ReadUnaligned<Half>(bBlock1);

            // Load 32 bytes from each block.
            Vector256<sbyte> va0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(aBlock0 + 2);
            Vector256<sbyte> vb0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(bBlock0 + 2);
            Vector256<sbyte> va1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(aBlock1 + 2);
            Vector256<sbyte> vb1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(bBlock1 + 2);

            // Sign trick on 256-bit halves.
            Vector256<sbyte> absA0 = Avx2.Sign(va0, va0);
            Vector256<sbyte> adjB0 = Avx2.Sign(vb0, va0);
            Vector256<sbyte> absA1 = Avx2.Sign(va1, va1);
            Vector256<sbyte> adjB1 = Avx2.Sign(vb1, va1);

            // MAD to int16 then int32 on each half.
            Vector256<short> prod0 = Avx2.MultiplyAddAdjacent(absA0.AsByte(), adjB0);
            Vector256<int> isum0 = Avx2.MultiplyAddAdjacent(prod0, ones256);
            Vector256<short> prod1 = Avx2.MultiplyAddAdjacent(absA1.AsByte(), adjB1);
            Vector256<int> isum1 = Avx2.MultiplyAddAdjacent(prod1, ones256);

            // Combine into 512-bit vectors.
            Vector512<int> isum512 = Vector512.Create(isum0, isum1);
            Vector512<float> fsum512 = Avx512F.ConvertToVector512Single(isum512);

            // Dual scale: lower 8 lanes get da0*db0, upper 8 get da1*db1.
            Vector512<float> scale = Vector512.Create(
                Vector256.Create(da0 * db0),
                Vector256.Create(da1 * db1));

            acc = Avx512F.FusedMultiplyAdd(fsum512, scale, acc);
        }

        float result = HorizontalSumAvx512Float(acc);

        // Handle odd trailing block via AVX2 single-block.
        if (block < blockCount)
        {
            byte* aBlock = a + block * Q8_0BlockBytes;
            byte* bBlock = b + block * Q8_0BlockBytes;

            float da = (float)Unsafe.ReadUnaligned<Half>(aBlock);
            float db = (float)Unsafe.ReadUnaligned<Half>(bBlock);

            Vector256<sbyte> va = Unsafe.ReadUnaligned<Vector256<sbyte>>(aBlock + 2);
            Vector256<sbyte> vb = Unsafe.ReadUnaligned<Vector256<sbyte>>(bBlock + 2);

            Vector256<sbyte> absA = Avx2.Sign(va, va);
            Vector256<sbyte> adjB = Avx2.Sign(vb, va);
            Vector256<short> prod = Avx2.MultiplyAddAdjacent(absA.AsByte(), adjB);
            Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, Vector256.Create((short)1));
            Vector256<float> fsum = Avx.ConvertToVector256Single(isum);

            result += da * db * HorizontalSumAvx2Float(fsum);
        }

        return result;
    }

    /// <summary>
    /// AVX-512 multi-row (4 rows) Q8_0 dot product. Processes 2 blocks per iteration
    /// and computes against 4 weight rows simultaneously.
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ8_0Avx512_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* x, int blockCount, float* results)
    {
        Vector512<float> acc0 = Vector512<float>.Zero;
        Vector512<float> acc1 = Vector512<float>.Zero;
        Vector512<float> acc2 = Vector512<float>.Zero;
        Vector512<float> acc3 = Vector512<float>.Zero;
        Vector256<short> ones256 = Vector256.Create((short)1);

        int block = 0;

        for (; block + 1 < blockCount; block += 2)
        {
            byte* xBlock0 = x + block * Q8_0BlockBytes;
            byte* xBlock1 = x + (block + 1) * Q8_0BlockBytes;
            float dx0 = (float)Unsafe.ReadUnaligned<Half>(xBlock0);
            float dx1 = (float)Unsafe.ReadUnaligned<Half>(xBlock1);

            // Load x data once.
            Vector256<sbyte> vx0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock0 + 2);
            Vector256<sbyte> vx1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock1 + 2);
            Vector256<sbyte> absX0 = Avx2.Sign(vx0, vx0);
            Vector256<sbyte> absX1 = Avx2.Sign(vx1, vx1);

            // Process each weight row.
            ProcessAvx512DualBlock(w0, block, vx0, vx1, absX0, absX1, dx0, dx1, ones256, ref acc0);
            ProcessAvx512DualBlock(w1, block, vx0, vx1, absX0, absX1, dx0, dx1, ones256, ref acc1);
            ProcessAvx512DualBlock(w2, block, vx0, vx1, absX0, absX1, dx0, dx1, ones256, ref acc2);
            ProcessAvx512DualBlock(w3, block, vx0, vx1, absX0, absX1, dx0, dx1, ones256, ref acc3);
        }

        results[0] = HorizontalSumAvx512Float(acc0);
        results[1] = HorizontalSumAvx512Float(acc1);
        results[2] = HorizontalSumAvx512Float(acc2);
        results[3] = HorizontalSumAvx512Float(acc3);

        // Handle odd trailing block via AVX2.
        if (block < blockCount)
        {
            byte* xBlock = x + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);
            Vector256<sbyte> vx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);
            Vector256<sbyte> absX = Avx2.Sign(vx, vx);
            Vector256<short> ones = Vector256.Create((short)1);

            results[0] += ProcessAvx2SingleBlock(w0, block, vx, absX, dx, ones);
            results[1] += ProcessAvx2SingleBlock(w1, block, vx, absX, dx, ones);
            results[2] += ProcessAvx2SingleBlock(w2, block, vx, absX, dx, ones);
            results[3] += ProcessAvx2SingleBlock(w3, block, vx, absX, dx, ones);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessAvx512DualBlock(
        byte* w, int block,
        Vector256<sbyte> vx0, Vector256<sbyte> vx1,
        Vector256<sbyte> absX0, Vector256<sbyte> absX1,
        float dx0, float dx1,
        Vector256<short> ones256,
        ref Vector512<float> acc)
    {
        byte* wBlock0 = w + block * Q8_0BlockBytes;
        byte* wBlock1 = w + (block + 1) * Q8_0BlockBytes;
        float dw0 = (float)Unsafe.ReadUnaligned<Half>(wBlock0);
        float dw1 = (float)Unsafe.ReadUnaligned<Half>(wBlock1);

        Vector256<sbyte> vw0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock0 + 2);
        Vector256<sbyte> vw1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock1 + 2);

        Vector256<sbyte> adjW0 = Avx2.Sign(vw0, vx0);
        Vector256<sbyte> adjW1 = Avx2.Sign(vw1, vx1);

        Vector256<short> prod0 = Avx2.MultiplyAddAdjacent(absX0.AsByte(), adjW0);
        Vector256<int> isum0 = Avx2.MultiplyAddAdjacent(prod0, ones256);
        Vector256<short> prod1 = Avx2.MultiplyAddAdjacent(absX1.AsByte(), adjW1);
        Vector256<int> isum1 = Avx2.MultiplyAddAdjacent(prod1, ones256);

        Vector512<int> isum512 = Vector512.Create(isum0, isum1);
        Vector512<float> fsum512 = Avx512F.ConvertToVector512Single(isum512);

        Vector512<float> scale = Vector512.Create(
            Vector256.Create(dx0 * dw0),
            Vector256.Create(dx1 * dw1));

        acc = Avx512F.FusedMultiplyAdd(fsum512, scale, acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ProcessAvx2SingleBlock(
        byte* w, int block,
        Vector256<sbyte> vx, Vector256<sbyte> absX,
        float dx, Vector256<short> ones)
    {
        byte* wBlock = w + block * Q8_0BlockBytes;
        float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
        Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
        Vector256<sbyte> adjW = Avx2.Sign(vw, vx);
        Vector256<short> prod = Avx2.MultiplyAddAdjacent(absX.AsByte(), adjW);
        Vector256<int> isum = Avx2.MultiplyAddAdjacent(prod, ones);
        Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
        return dx * dw * HorizontalSumAvx2Float(fsum);
    }

    // ──────────────────── Quantization ────────────────────

    /// <summary>
    /// Quantizes f32 data to Q8_0 format. Per block of 32 floats:
    /// scale = max(|x[i]|) / 127, qs[i] = round(x[i] / scale) clamped to [-127, 127].
    /// Dispatches to AVX-512 → AVX2 → scalar at runtime.
    /// </summary>
    /// <param name="src">Source f32 data. Must have <paramref name="elementCount"/> elements.</param>
    /// <param name="dest">Destination Q8_0 buffer. Must have (elementCount/32) × 34 bytes.</param>
    /// <param name="elementCount">Number of float elements. Must be a multiple of 32.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void QuantizeF32ToQ8_0(float* src, byte* dest, int elementCount)
    {
        if (Avx512BW.IsSupported)
            QuantizeF32ToQ8_0Avx512(src, dest, elementCount);
        else if (Avx2.IsSupported)
            QuantizeF32ToQ8_0Avx2(src, dest, elementCount);
        else
            QuantizeF32ToQ8_0Scalar(src, dest, elementCount);
    }

    /// <summary>
    /// Scalar quantization reference implementation.
    /// </summary>
    [SkipLocalsInit]
    internal static void QuantizeF32ToQ8_0Scalar(float* src, byte* dest, int elementCount)
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

    /// <summary>
    /// AVX2 SIMD quantization: processes 32 floats per block using 4 iterations of 8 floats.
    /// </summary>
    [SkipLocalsInit]
    internal static void QuantizeF32ToQ8_0Avx2(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / Q8_0GroupSize;
        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * Q8_0GroupSize;
            byte* blockDst = dest + block * Q8_0BlockBytes;

            // Max-abs scan: 4 loads of 8 floats.
            Vector256<float> v0 = Vector256.Abs(Avx.LoadVector256(blockSrc));
            Vector256<float> v1 = Vector256.Abs(Avx.LoadVector256(blockSrc + 8));
            Vector256<float> v2 = Vector256.Abs(Avx.LoadVector256(blockSrc + 16));
            Vector256<float> v3 = Vector256.Abs(Avx.LoadVector256(blockSrc + 24));

            Vector256<float> max01 = Avx.Max(v0, v1);
            Vector256<float> max23 = Avx.Max(v2, v3);
            Vector256<float> maxAll = Avx.Max(max01, max23);
            float maxAbs = HorizontalMaxAvx2(maxAll);

            float scale = maxAbs / 127.0f;
            Unsafe.WriteUnaligned(blockDst, (Half)scale);

            sbyte* qs = (sbyte*)(blockDst + 2);
            if (scale == 0)
            {
                // Zero out all 32 bytes.
                Vector256<sbyte>.Zero.StoreUnsafe(ref Unsafe.AsRef<sbyte>(qs));
            }
            else
            {
                Vector256<float> vInvScale = Vector256.Create(1.0f / scale);

                // Process 8 floats at a time → round → convert to int32 → pack to sbyte.
                Vector256<int> i0 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(
                    Avx.Multiply(Avx.LoadVector256(blockSrc), vInvScale)));
                Vector256<int> i1 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(
                    Avx.Multiply(Avx.LoadVector256(blockSrc + 8), vInvScale)));
                Vector256<int> i2 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(
                    Avx.Multiply(Avx.LoadVector256(blockSrc + 16), vInvScale)));
                Vector256<int> i3 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(
                    Avx.Multiply(Avx.LoadVector256(blockSrc + 24), vInvScale)));

                // Pack int32 → int16 (saturating).
                Vector256<short> s01 = Avx2.PackSignedSaturate(i0, i1);
                Vector256<short> s23 = Avx2.PackSignedSaturate(i2, i3);

                // Pack int16 → int8 (saturating).
                Vector256<sbyte> packed = Avx2.PackSignedSaturate(s01, s23);

                // Fix AVX2 lane-crossing: PackSignedSaturate interleaves lanes.
                // Permute to get contiguous output: [0,4,1,5,2,6,3,7]
                Vector256<int> permuted = Avx2.PermuteVar8x32(packed.AsInt32(),
                    Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7));

                permuted.AsByte().StoreUnsafe(ref Unsafe.AsRef<byte>((byte*)qs));
            }
        }
    }

    /// <summary>
    /// AVX-512 SIMD quantization: processes 32 floats per block using 2 iterations of 16 floats.
    /// </summary>
    [SkipLocalsInit]
    internal static void QuantizeF32ToQ8_0Avx512(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / Q8_0GroupSize;

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * Q8_0GroupSize;
            byte* blockDst = dest + block * Q8_0BlockBytes;

            // Max-abs scan: 2 loads of 16 floats.
            Vector512<float> v0 = Vector512.Abs(Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc)));
            Vector512<float> v1 = Vector512.Abs(Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc + 16)));

            Vector512<float> maxAll = Avx512F.Max(v0, v1);
            // Reduce 512-bit to scalar max.
            Vector256<float> max256 = Avx.Max(maxAll.GetLower(), maxAll.GetUpper());
            float maxAbs = HorizontalMaxAvx2(max256);

            float scale = maxAbs / 127.0f;
            Unsafe.WriteUnaligned(blockDst, (Half)scale);

            sbyte* qs = (sbyte*)(blockDst + 2);
            if (scale == 0)
            {
                Vector256<sbyte>.Zero.StoreUnsafe(ref Unsafe.AsRef<sbyte>(qs));
            }
            else
            {
                Vector512<float> vInvScale = Vector512.Create(1.0f / scale);

                // Process 16 floats at a time.
                Vector512<int> i0 = Avx512F.ConvertToVector512Int32(Avx512F.RoundScale(
                    Avx512F.Multiply(
                        Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc)),
                        vInvScale),
                    0x08)); // _MM_FROUND_TO_NEAREST_INT
                Vector512<int> i1 = Avx512F.ConvertToVector512Int32(Avx512F.RoundScale(
                    Avx512F.Multiply(
                        Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc + 16)),
                        vInvScale),
                    0x08));

                // Pack int32 → int16 → int8 using AVX2 on each 256-bit half.
                Vector256<short> s0 = Avx2.PackSignedSaturate(i0.GetLower(), i0.GetUpper());
                Vector256<short> s1 = Avx2.PackSignedSaturate(i1.GetLower(), i1.GetUpper());
                Vector256<sbyte> packed = Avx2.PackSignedSaturate(s0, s1);

                // Fix lane-crossing.
                Vector256<int> permuted = Avx2.PermuteVar8x32(packed.AsInt32(),
                    Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7));

                permuted.AsByte().StoreUnsafe(ref Unsafe.AsRef<byte>((byte*)qs));
            }
        }
    }

    // ──────────────────── Q4_K stubs ────────────────────

    /// <summary>
    /// Scalar Q4_K × Q8_0 dot product. Stub for Step 12 implementation.
    /// </summary>
    /// <exception cref="NotImplementedException">Always thrown — this is a placeholder for Step 12.</exception>
    internal static float VecDotQ4_K_Q8_0Scalar(byte* a, byte* b, int blockCount)
    {
        throw new NotImplementedException(
            "Q4_K × Q8_0 dot product is not yet implemented. See Step 12 in docs/ROADMAP.md.");
    }

    // ──────────────────── Horizontal reduction helpers ────────────────────

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalSumAvx2Float(Vector256<float> v)
    {
        // Sum upper and lower 128-bit halves.
        Vector128<float> lo = v.GetLower();
        Vector128<float> hi = v.GetUpper();
        Vector128<float> sum128 = Sse.Add(lo, hi);

        // hadd: [a+b, c+d, a+b, c+d]
        sum128 = Sse3.HorizontalAdd(sum128, sum128);
        // hadd: [a+b+c+d, ...]
        sum128 = Sse3.HorizontalAdd(sum128, sum128);

        return sum128.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalSumAvx512Float(Vector512<float> v)
    {
        Vector256<float> lo = v.GetLower();
        Vector256<float> hi = v.GetUpper();
        return HorizontalSumAvx2Float(Avx.Add(lo, hi));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMaxAvx2(Vector256<float> v)
    {
        Vector128<float> lo = v.GetLower();
        Vector128<float> hi = v.GetUpper();
        Vector128<float> max128 = Sse.Max(lo, hi);

        // Shuffle and max to reduce.
        Vector128<float> shuf = Sse.MoveHighToLow(max128, max128);
        max128 = Sse.Max(max128, shuf);
        shuf = Sse.Shuffle(max128, max128, 0b_00_01_00_01);
        max128 = Sse.Max(max128, shuf);

        return max128.ToScalar();
    }
}
