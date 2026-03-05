using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

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
    /// Computes the number of weight rows per tile that fits within ~50% of a typical 512KB L2 cache.
    /// Result is aligned down to 4 rows for efficient VecDot batching.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ComputeTileM(int rowBytes)
    {
        const int L2Budget = 256 * 1024; // 50% of typical 512KB L2
        int tileM = L2Budget / rowBytes;
        tileM = (tileM / 4) * 4; // align to 4-row VecDot batch
        return Math.Clamp(tileM, 4, 256);
    }

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
    internal static void ComputeRows(byte* weightsQ8, byte* xQ8, float* result, int m, int blockCount)
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
    public static void QuantizeF32ToQ8_0(float* src, byte* dest, int elementCount)
    {
        if (elementCount % Q8_0GroupSize != 0)
            throw new ArgumentException(
                $"elementCount must be a multiple of {Q8_0GroupSize}, got {elementCount}",
                nameof(elementCount));

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

    // ──────────────────── Tiled GEMM helpers ────────────────────

    /// <summary>
    /// Cache-tiled Q8_0 GEMM core. Iterates weight-tile-first so that a tile of weight rows
    /// (~256KB) stays in L2 cache while all N tokens are computed against it.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ComputeGemmTiled(byte* weightsQ8, byte* inputQ8, float* c,
                                         int m, int n, int blockCount)
    {
        int q8RowBytes = blockCount * Q8_0BlockBytes;
        int tileM = ComputeTileM(q8RowBytes);

        for (int mStart = 0; mStart < m; mStart += tileM)
        {
            int tileRows = Math.Min(tileM, m - mStart);
            byte* tileWeights = weightsQ8 + (long)mStart * q8RowBytes;

            for (int t = 0; t < n; t++)
                ComputeRows(tileWeights, inputQ8 + t * q8RowBytes,
                            c + t * m + mStart, tileRows, blockCount);
        }
    }

    // ──────────────────── GEMM ────────────────────

    /// <summary>
    /// Scalar f32 GEMM reference: <c>C[N,M] = B[N,K] × A[M,K]^T</c>.
    /// A is [M,K] row-major (weights), B is [N,K] row-major (inputs), C is [N,M] row-major (outputs).
    /// </summary>
    [SkipLocalsInit]
    internal static void GemmF32Scalar(float* a, float* b, float* c, int m, int k, int n)
    {
        for (int t = 0; t < n; t++)
        {
            float* inputRow = b + t * k;
            float* outputRow = c + t * m;
            GemvF32Scalar(a, inputRow, outputRow, m, k);
        }
    }

    /// <summary>
    /// Optimized f32 GEMM: <c>C[N,M] = B[N,K] × A[M,K]^T</c>.
    /// Uses cache-tiled traversal: weight-tile-first so tiles stay in L2 across tokens.
    /// </summary>
    [SkipLocalsInit]
    public static void GemmF32(float* a, float* b, float* c, int m, int k, int n)
    {
        int rowBytes = k * sizeof(float);
        int tileM = ComputeTileM(rowBytes);

        for (int mStart = 0; mStart < m; mStart += tileM)
        {
            int tileRows = Math.Min(tileM, m - mStart);
            float* tileWeights = a + (long)mStart * k;

            for (int t = 0; t < n; t++)
                GemvF32(tileWeights, b + t * k, c + t * m + mStart, tileRows, k);
        }
    }

    /// <summary>
    /// Q8_0 GEMM: <c>C[N,M] = B[N,K] × A[M,K]^T</c> where A is Q8_0 weights, B is f32 inputs.
    /// Quantizes all N input rows to Q8_0 once, then calls ComputeRows per token.
    /// When N==1, delegates to GemvQ8_0.
    /// </summary>
    /// <param name="weightsQ8">Q8_0 weight matrix [M,K]. Each row is K/32 blocks of 34 bytes.</param>
    /// <param name="b">f32 input matrix [N,K], row-major.</param>
    /// <param name="c">f32 output matrix [N,M], row-major.</param>
    /// <param name="m">Number of weight rows (output dimension).</param>
    /// <param name="k">Number of columns (input dimension). Must be a multiple of 32.</param>
    /// <param name="n">Number of input tokens (batch size).</param>
    /// <param name="preQuantizedInput">Optional pre-quantized Q8_0 input [N * q8RowBytes].
    /// When non-null, skips quantization (caller pre-quantized for reuse across projections).</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ8_0(byte* weightsQ8, float* b, float* c, int m, int k, int n,
                                byte* preQuantizedInput = null)
    {
        if (k % Q8_0GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        if (n == 1)
        {
            if (preQuantizedInput != null)
            {
                // Use pre-quantized input directly.
                int blockCount = k / Q8_0GroupSize;
                ComputeRows(weightsQ8, preQuantizedInput, c, m, blockCount);
            }
            else
            {
                GemvQ8_0(weightsQ8, b, c, m, k);
            }
            return;
        }

        int blockCount2 = k / Q8_0GroupSize;
        int q8RowBytes = blockCount2 * Q8_0BlockBytes;

        if (preQuantizedInput != null)
        {
            // Pre-quantized path: tiled compute directly.
            ComputeGemmTiled(weightsQ8, preQuantizedInput, c, m, n, blockCount2);
            return;
        }

        // Quantize all input rows, then tiled compute.
        int totalQ8Bytes = n * q8RowBytes;
        byte[] rented = ArrayPool<byte>.Shared.Rent(totalQ8Bytes);
        fixed (byte* rentedPtr = rented)
        {
            for (int t = 0; t < n; t++)
                QuantizeF32ToQ8_0(b + t * k, rentedPtr + t * q8RowBytes, k);

            ComputeGemmTiled(weightsQ8, rentedPtr, c, m, n, blockCount2);
        }
        ArrayPool<byte>.Shared.Return(rented);
    }

    // ──────────────────── F16 GEMV / GEMM ────────────────────

    /// <summary>
    /// F16 GEMV: dequantize each row to f32 scratch, then dot product.
    /// A is [M,K] F16 row-major (weights), x is [K] f32, result is [M] f32.
    /// </summary>
    [SkipLocalsInit]
    public static void GemvF16(nint weights, float* x, float* y, int m, int k)
    {
        const int stackThreshold = 2048; // 8KB of floats
        Half* weightsHalf = (Half*)weights;

        if (k <= stackThreshold)
        {
            float* rowBuf = stackalloc float[k];
            for (int row = 0; row < m; row++)
            {
                var srcRow = new ReadOnlySpan<Half>(weightsHalf + row * k, k);
                var destRow = new Span<float>(rowBuf, k);
                TensorPrimitives.ConvertToSingle(srcRow, destRow);
                y[row] = TensorPrimitives.Dot(destRow, new ReadOnlySpan<float>(x, k));
            }
        }
        else
        {
            float[] rented = ArrayPool<float>.Shared.Rent(k);
            try
            {
                for (int row = 0; row < m; row++)
                {
                    var srcRow = new ReadOnlySpan<Half>(weightsHalf + row * k, k);
                    var destRow = rented.AsSpan(0, k);
                    TensorPrimitives.ConvertToSingle(srcRow, destRow);
                    y[row] = TensorPrimitives.Dot(destRow, new ReadOnlySpan<float>(x, k));
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }
    }

    /// <summary>
    /// F16 GEMM: <c>C[N,M] = B[N,K] × A[M,K]^T</c> where A is F16 weights.
    /// Uses cache-tiled traversal: weight-tile-first so tiles stay in L2 across tokens.
    /// Rents one scratch buffer for dequantization, avoiding per-call ArrayPool churn.
    /// </summary>
    [SkipLocalsInit]
    public static void GemmF16(nint weights, float* b, float* c, int m, int k, int n)
    {
        int rowBytes = k * sizeof(Half);
        int tileM = ComputeTileM(rowBytes);
        Half* weightsHalf = (Half*)weights;

        float[] rented = ArrayPool<float>.Shared.Rent(k);
        try
        {
            fixed (float* rowBuf = rented)
            {
                for (int mStart = 0; mStart < m; mStart += tileM)
                {
                    int tileRows = Math.Min(tileM, m - mStart);
                    Half* tileWeightsHalf = weightsHalf + (long)mStart * k;

                    for (int t = 0; t < n; t++)
                    {
                        float* xPtr = b + t * k;
                        float* outPtr = c + t * m + mStart;
                        var xSpan = new ReadOnlySpan<float>(xPtr, k);
                        var destRow = new Span<float>(rowBuf, k);

                        for (int row = 0; row < tileRows; row++)
                        {
                            var srcRow = new ReadOnlySpan<Half>(tileWeightsHalf + row * k, k);
                            TensorPrimitives.ConvertToSingle(srcRow, destRow);
                            outPtr[row] = TensorPrimitives.Dot(destRow, xSpan);
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    // ──────────────────── Parallel overloads ────────────────────

    /// <summary>Minimum M rows before parallelizing GEMV.</summary>
    private const int ParallelMinRows = 32;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PartitionRows(int totalRows, int threadIdx, int threadCount,
                                      out int start, out int count)
    {
        int chunk = ((totalRows + threadCount - 1) / threadCount + 3) & ~3;
        start = threadIdx * chunk;
        if (start >= totalRows) { start = 0; count = 0; return; }
        count = Math.Min(chunk, totalRows - start);
    }

    // ── Context structs ──

    private struct ComputeRowsCtx
    {
        public byte* WeightsQ8;
        public byte* XQ8;
        public float* Result;
        public int M;
        public int BlockCount;
    }

    private struct GemvF32Ctx
    {
        public float* A;
        public float* X;
        public float* Result;
        public int M;
        public int K;
    }

    private struct GemvF16Ctx
    {
        public nint Weights;
        public float* X;
        public float* Y;
        public int M;
        public int K;
        public nint* ScratchPtrs;
    }

    private struct GemmTiledQ8Ctx
    {
        public byte* WeightsQ8;
        public byte* InputQ8;
        public float* C;
        public int M;
        public int N;
        public int BlockCount;
        public int TileM;
        public int Q8RowBytes;
    }

    private struct GemmTiledF32Ctx
    {
        public float* A;
        public float* B;
        public float* C;
        public int M;
        public int K;
        public int N;
        public int TileM;
    }

    private struct GemmTiledF16Ctx
    {
        public nint Weights;
        public float* B;
        public float* C;
        public int M;
        public int K;
        public int N;
        public int TileM;
        public nint* ScratchPtrs;
    }

    // ── Worker methods ──

    private static void ComputeRowsWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<ComputeRowsCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        int rowBytes = ctx.BlockCount * Q8_0BlockBytes;
        ComputeRows(ctx.WeightsQ8 + (long)start * rowBytes, ctx.XQ8,
                    ctx.Result + start, count, ctx.BlockCount);
    }

    private static void GemvF32Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvF32Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvF32(ctx.A + (long)start * ctx.K, ctx.X, ctx.Result + start, count, ctx.K);
    }

    private static void GemvF16Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvF16Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        Half* weightsHalf = (Half*)ctx.Weights;
        float* scratch = (float*)ctx.ScratchPtrs[threadIdx];
        var xSpan = new ReadOnlySpan<float>(ctx.X, ctx.K);
        var destRow = new Span<float>(scratch, ctx.K);
        for (int row = start; row < start + count; row++)
        {
            var srcRow = new ReadOnlySpan<Half>(weightsHalf + (long)row * ctx.K, ctx.K);
            TensorPrimitives.ConvertToSingle(srcRow, destRow);
            ctx.Y[row] = TensorPrimitives.Dot(destRow, xSpan);
        }
    }

    private static void GemmTiledQ8Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmTiledQ8Ctx>((void*)ctxPtr);
        int totalTiles = (ctx.M + ctx.TileM - 1) / ctx.TileM;
        int tilesPerThread = (totalTiles + threadCount - 1) / threadCount;
        int startTile = threadIdx * tilesPerThread;
        int endTile = Math.Min(startTile + tilesPerThread, totalTiles);

        for (int tile = startTile; tile < endTile; tile++)
        {
            int mStart = tile * ctx.TileM;
            int tileRows = Math.Min(ctx.TileM, ctx.M - mStart);
            byte* tileWeights = ctx.WeightsQ8 + (long)mStart * ctx.Q8RowBytes;
            for (int t = 0; t < ctx.N; t++)
                ComputeRows(tileWeights, ctx.InputQ8 + t * ctx.Q8RowBytes,
                            ctx.C + t * ctx.M + mStart, tileRows, ctx.BlockCount);
        }
    }

    private static void GemmTiledF32Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmTiledF32Ctx>((void*)ctxPtr);
        int totalTiles = (ctx.M + ctx.TileM - 1) / ctx.TileM;
        int tilesPerThread = (totalTiles + threadCount - 1) / threadCount;
        int startTile = threadIdx * tilesPerThread;
        int endTile = Math.Min(startTile + tilesPerThread, totalTiles);

        for (int tile = startTile; tile < endTile; tile++)
        {
            int mStart = tile * ctx.TileM;
            int tileRows = Math.Min(ctx.TileM, ctx.M - mStart);
            float* tileWeights = ctx.A + (long)mStart * ctx.K;
            for (int t = 0; t < ctx.N; t++)
                GemvF32(tileWeights, ctx.B + t * ctx.K, ctx.C + t * ctx.M + mStart, tileRows, ctx.K);
        }
    }

    private static void GemmTiledF16Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmTiledF16Ctx>((void*)ctxPtr);
        int totalTiles = (ctx.M + ctx.TileM - 1) / ctx.TileM;
        int tilesPerThread = (totalTiles + threadCount - 1) / threadCount;
        int startTile = threadIdx * tilesPerThread;
        int endTile = Math.Min(startTile + tilesPerThread, totalTiles);

        Half* weightsHalf = (Half*)ctx.Weights;
        float* rowBuf = (float*)ctx.ScratchPtrs[threadIdx];
        var destRow = new Span<float>(rowBuf, ctx.K);

        for (int tile = startTile; tile < endTile; tile++)
        {
            int mStart = tile * ctx.TileM;
            int tileRows = Math.Min(ctx.TileM, ctx.M - mStart);
            Half* tileWeightsHalf = weightsHalf + (long)mStart * ctx.K;
            for (int t = 0; t < ctx.N; t++)
            {
                float* xPtr = ctx.B + t * ctx.K;
                float* outPtr = ctx.C + t * ctx.M + mStart;
                var xSpan = new ReadOnlySpan<float>(xPtr, ctx.K);
                for (int row = 0; row < tileRows; row++)
                {
                    var srcRow = new ReadOnlySpan<Half>(tileWeightsHalf + row * ctx.K, ctx.K);
                    TensorPrimitives.ConvertToSingle(srcRow, destRow);
                    outPtr[row] = TensorPrimitives.Dot(destRow, xSpan);
                }
            }
        }
    }

    // ── Parallel public API ──

    /// <summary>
    /// Q8_0 GEMV with optional parallelism. Falls back to single-threaded when pool is null or M &lt; 32.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ8_0(byte* weightsQ8, float* x, float* result, int m, int k,
                                ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvQ8_0(weightsQ8, x, result, m, k);
            return;
        }

        if (k % Q8_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_0GroupSize;
        int xQ8Bytes = blockCount * Q8_0BlockBytes;

        // Quantize x once (single-threaded) into pool scratch for thread 0
        byte* xQ8 = (byte*)pool.GetWorkerScratch(0, xQ8Bytes);
        QuantizeF32ToQ8_0(x, xQ8, k);

        var ctx = new ComputeRowsCtx
        {
            WeightsQ8 = weightsQ8, XQ8 = xQ8, Result = result,
            M = m, BlockCount = blockCount
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsWorker);
    }

    /// <summary>
    /// f32 GEMV with optional parallelism.
    /// </summary>
    [SkipLocalsInit]
    public static void GemvF32(float* a, float* x, float* result, int m, int k,
                               ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvF32(a, x, result, m, k);
            return;
        }

        var ctx = new GemvF32Ctx { A = a, X = x, Result = result, M = m, K = k };
        pool.Dispatch((nint)(&ctx), &GemvF32Worker);
    }

    /// <summary>
    /// F16 GEMV with optional parallelism. Uses per-worker scratch for dequantization.
    /// </summary>
    [SkipLocalsInit]
    public static void GemvF16(nint weights, float* x, float* y, int m, int k,
                               ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            GemvF16(weights, x, y, m, k);
            return;
        }

        int threadCount = pool.ThreadCount;
        nint* scratchPtrs = stackalloc nint[threadCount];
        int scratchBytes = k * sizeof(float);
        for (int i = 0; i < threadCount; i++)
            scratchPtrs[i] = pool.GetWorkerScratch(i, scratchBytes);

        var ctx = new GemvF16Ctx
        {
            Weights = weights, X = x, Y = y,
            M = m, K = k, ScratchPtrs = scratchPtrs
        };
        pool.Dispatch((nint)(&ctx), &GemvF16Worker);
    }

    /// <summary>
    /// Q8_0 ComputeRows with optional parallelism.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRows(byte* weightsQ8, byte* xQ8, float* result, int m, int blockCount,
                                     ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            ComputeRows(weightsQ8, xQ8, result, m, blockCount);
            return;
        }

        var ctx = new ComputeRowsCtx
        {
            WeightsQ8 = weightsQ8, XQ8 = xQ8, Result = result,
            M = m, BlockCount = blockCount
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsWorker);
    }

    /// <summary>
    /// Q8_0 GEMM with optional parallelism. Quantizes inputs single-threaded, then parallelizes tiled compute.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmQ8_0(byte* weightsQ8, float* b, float* c, int m, int k, int n,
                                ComputeThreadPool? pool, byte* preQuantizedInput = null)
    {
        if (pool is null)
        {
            GemmQ8_0(weightsQ8, b, c, m, k, n, preQuantizedInput);
            return;
        }

        if (k % Q8_0GroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        if (n == 1)
        {
            if (preQuantizedInput != null)
            {
                int blockCount = k / Q8_0GroupSize;
                ComputeRows(weightsQ8, preQuantizedInput, c, m, blockCount, pool);
            }
            else
            {
                GemvQ8_0(weightsQ8, b, c, m, k, pool);
            }
            return;
        }

        int blockCount2 = k / Q8_0GroupSize;
        int q8RowBytes = blockCount2 * Q8_0BlockBytes;
        int tileM = ComputeTileM(q8RowBytes);
        int totalTiles = (m + tileM - 1) / tileM;

        if (preQuantizedInput != null)
        {
            if (totalTiles < 2)
            {
                ComputeGemmTiled(weightsQ8, preQuantizedInput, c, m, n, blockCount2);
                return;
            }
            var ctx = new GemmTiledQ8Ctx
            {
                WeightsQ8 = weightsQ8, InputQ8 = preQuantizedInput, C = c,
                M = m, N = n, BlockCount = blockCount2, TileM = tileM, Q8RowBytes = q8RowBytes
            };
            pool.Dispatch((nint)(&ctx), &GemmTiledQ8Worker);
            return;
        }

        // Quantize all input rows (single-threaded), then parallel tiled compute
        int totalQ8Bytes = n * q8RowBytes;
        byte[] rented = ArrayPool<byte>.Shared.Rent(totalQ8Bytes);
        fixed (byte* rentedPtr = rented)
        {
            for (int t = 0; t < n; t++)
                QuantizeF32ToQ8_0(b + t * k, rentedPtr + t * q8RowBytes, k);

            if (totalTiles < 2)
            {
                ComputeGemmTiled(weightsQ8, rentedPtr, c, m, n, blockCount2);
            }
            else
            {
                var ctx = new GemmTiledQ8Ctx
                {
                    WeightsQ8 = weightsQ8, InputQ8 = rentedPtr, C = c,
                    M = m, N = n, BlockCount = blockCount2, TileM = tileM, Q8RowBytes = q8RowBytes
                };
                pool.Dispatch((nint)(&ctx), &GemmTiledQ8Worker);
            }
        }
        ArrayPool<byte>.Shared.Return(rented);
    }

    /// <summary>
    /// f32 GEMM with optional parallelism.
    /// </summary>
    [SkipLocalsInit]
    public static void GemmF32(float* a, float* b, float* c, int m, int k, int n,
                               ComputeThreadPool? pool)
    {
        if (pool is null)
        {
            GemmF32(a, b, c, m, k, n);
            return;
        }

        if (n == 1)
        {
            GemvF32(a, b, c, m, k, pool);
            return;
        }

        int rowBytes = k * sizeof(float);
        int tileM = ComputeTileM(rowBytes);
        int totalTiles = (m + tileM - 1) / tileM;

        if (totalTiles < 2)
        {
            GemmF32(a, b, c, m, k, n);
            return;
        }

        var ctx = new GemmTiledF32Ctx { A = a, B = b, C = c, M = m, K = k, N = n, TileM = tileM };
        pool.Dispatch((nint)(&ctx), &GemmTiledF32Worker);
    }

    /// <summary>
    /// F16 GEMM with optional parallelism. Uses per-worker scratch for dequantization.
    /// </summary>
    [SkipLocalsInit]
    public static void GemmF16(nint weights, float* b, float* c, int m, int k, int n,
                               ComputeThreadPool? pool)
    {
        if (pool is null)
        {
            GemmF16(weights, b, c, m, k, n);
            return;
        }

        if (n == 1)
        {
            GemvF16(weights, b, c, m, k, pool);
            return;
        }

        int rowBytes = k * sizeof(Half);
        int tileM = ComputeTileM(rowBytes);
        int totalTiles = (m + tileM - 1) / tileM;

        if (totalTiles < 2)
        {
            GemmF16(weights, b, c, m, k, n);
            return;
        }

        int threadCount = pool.ThreadCount;
        nint* scratchPtrs = stackalloc nint[threadCount];
        int scratchBytes = k * sizeof(float);
        for (int i = 0; i < threadCount; i++)
            scratchPtrs[i] = pool.GetWorkerScratch(i, scratchBytes);

        var ctx = new GemmTiledF16Ctx
        {
            Weights = weights, B = b, C = c,
            M = m, K = k, N = n, TileM = tileM, ScratchPtrs = scratchPtrs
        };
        pool.Dispatch((nint)(&ctx), &GemmTiledF16Worker);
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
