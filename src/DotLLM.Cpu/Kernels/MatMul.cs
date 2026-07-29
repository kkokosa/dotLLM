using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Matrix-vector multiplication kernels for transformer inference.
/// Supports f32 weights and Q8_0 quantized weights with on-the-fly activation quantization.
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>Q8_0 block size in bytes: 2 (Half scale) + 32 (sbyte quantized values).</summary>
    private const int Q8_0BlockBytes = 34;

    /// <summary>Number of elements per Q8_0 block.</summary>
    private const int Q8_0GroupSize = 32;

    /// <summary>Q8_1 block size in bytes: 2 (Half d) + 2 (Half s) + 32 (sbyte quantized values).</summary>
    public const int Q8_1BlockBytes = 36;

    /// <summary>Number of elements per Q8_1 block.</summary>
    private const int Q8_1GroupSize = 32;

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

        if (Vector512.IsHardwareAccelerated)
        {
            int row = 0;
            // Process 4 rows at a time for cache efficiency.
            for (; row + 3 < m; row += 4)
            {
                VecDotQ8_0Vector512_4Rows(
                    weightsQ8 + row * rowBytes,
                    weightsQ8 + (row + 1) * rowBytes,
                    weightsQ8 + (row + 2) * rowBytes,
                    weightsQ8 + (row + 3) * rowBytes,
                    xQ8, blockCount, result + row);
            }
            for (; row < m; row++)
            {
                result[row] = VecDotQ8_0Vector512(weightsQ8 + row * rowBytes, xQ8, blockCount);
            }
        }
        else if (Vector256.IsHardwareAccelerated)
        {
            int row = 0;
            // Process 4 rows at a time for cache efficiency.
            for (; row + 3 < m; row += 4)
            {
                VecDotQ8_0Vector256_4Rows(
                    weightsQ8 + row * rowBytes,
                    weightsQ8 + (row + 1) * rowBytes,
                    weightsQ8 + (row + 2) * rowBytes,
                    weightsQ8 + (row + 3) * rowBytes,
                    xQ8, blockCount, result + row);
            }
            for (; row < m; row++)
            {
                result[row] = VecDotQ8_0Vector256(weightsQ8 + row * rowBytes, xQ8, blockCount);
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

    // ──────────────────── R4 Interleaved VecDot + ComputeRows for Q8_0 ────────────────────

    /// <summary>
    /// Scalar Q8_0 dot product for a single row within an R4-interleaved group.
    /// Block stride is 4 * Q8_0BlockBytes (blocks from 4 rows are interleaved).
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ8_0ScalarR4(byte* groupBase, int rowInGroup, byte* xQ8, int blockCount)
    {
        float sumf = 0;
        const int wStride = 4 * Q8_0BlockBytes;

        for (int block = 0; block < blockCount; block++)
        {
            byte* wBlock = groupBase + block * wStride + rowInGroup * Q8_0BlockBytes;
            byte* xBlock = xQ8 + block * Q8_0BlockBytes;

            float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);

            sbyte* qw = (sbyte*)(wBlock + 2);
            sbyte* qx = (sbyte*)(xBlock + 2);

            int sumi = 0;
            for (int i = 0; i < Q8_0GroupSize; i++)
                sumi += qw[i] * qx[i];

            sumf += dw * dx * sumi;
        }
        return sumf;
    }

    /// <summary>
    /// Vector256 4-row Q8_0 dot product for R4-interleaved layout.
    /// Blocks from 4 rows are interleaved: [r0_b0][r1_b0][r2_b0][r3_b0][r0_b1]...
    /// Block stride is 4 * Q8_0BlockBytes, so all 4 blocks for a column fit in 136 bytes (2-3 cache lines).
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ8_0Vector256_4RowsR4(
        byte* groupBase, byte* x, int blockCount, float* results)
    {
        Vector256<float> acc0 = Vector256<float>.Zero;
        Vector256<float> acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero;
        Vector256<float> acc3 = Vector256<float>.Zero;
        const int wStride = 4 * Q8_0BlockBytes;

        for (int block = 0; block < blockCount; block++)
        {
            byte* xBlock = x + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);
            Vector256<sbyte> vx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);

            byte* blockBase = groupBase + block * wStride;

            // Row 0
            {
                byte* wBlock = blockBase;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<int> isum = DotProductSByte32(vw, vx);
                acc0 = Vector256.FusedMultiplyAdd(Vector256.Create(dx * dw), Vector256.ConvertToSingle(isum), acc0);
            }

            // Row 1
            {
                byte* wBlock = blockBase + Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<int> isum = DotProductSByte32(vw, vx);
                acc1 = Vector256.FusedMultiplyAdd(Vector256.Create(dx * dw), Vector256.ConvertToSingle(isum), acc1);
            }

            // Row 2
            {
                byte* wBlock = blockBase + 2 * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<int> isum = DotProductSByte32(vw, vx);
                acc2 = Vector256.FusedMultiplyAdd(Vector256.Create(dx * dw), Vector256.ConvertToSingle(isum), acc2);
            }

            // Row 3
            {
                byte* wBlock = blockBase + 3 * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<int> isum = DotProductSByte32(vw, vx);
                acc3 = Vector256.FusedMultiplyAdd(Vector256.Create(dx * dw), Vector256.ConvertToSingle(isum), acc3);
            }
        }

        results[0] = Vector256.Sum(acc0);
        results[1] = Vector256.Sum(acc1);
        results[2] = Vector256.Sum(acc2);
        results[3] = Vector256.Sum(acc3);
    }

    /// <summary>
    /// Processes R4-interleaved Q8_0 weights where groups of 4 rows have their blocks
    /// stored contiguously: [r0_b0][r1_b0][r2_b0][r3_b0][r0_b1][r1_b1]...
    /// Reads sequentially instead of striding, improving cache and prefetch behavior.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ8_0Interleaved(byte* repackedWeights, byte* xQ8, float* result,
        int fullGroups, int tailRows, int blockCount)
    {
        int groupBytes = 4 * blockCount * Q8_0BlockBytes;

        if (Vector256.IsHardwareAccelerated)
        {
            for (int g = 0; g < fullGroups; g++)
            {
                byte* groupBase = repackedWeights + (long)g * groupBytes;
                VecDotQ8_0Vector256_4RowsR4(groupBase, xQ8, blockCount, result + g * 4);
            }
        }
        else
        {
            for (int g = 0; g < fullGroups; g++)
            {
                byte* groupBase = repackedWeights + (long)g * groupBytes;
                for (int r = 0; r < 4; r++)
                    result[g * 4 + r] = VecDotQ8_0ScalarR4(groupBase, r, xQ8, blockCount);
            }
        }

        // Tail rows (row-major, after interleaved data)
        if (tailRows > 0)
        {
            int rowBytes = blockCount * Q8_0BlockBytes;
            byte* tailBase = repackedWeights + (long)fullGroups * groupBytes;
            for (int r = 0; r < tailRows; r++)
                result[fullGroups * 4 + r] = Vector256.IsHardwareAccelerated
                    ? VecDotQ8_0Vector256(tailBase + (long)r * rowBytes, xQ8, blockCount)
                    : VecDotQ8_0Scalar(tailBase + (long)r * rowBytes, xQ8, blockCount);
        }
    }

    /// <summary>
    /// Parallel R4-interleaved Q8_0 ComputeRows. Partitions groups across threads.
    /// Falls back to single-threaded when pool is null or M is small.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ8_0Interleaved(byte* repackedWeights, byte* xQ8, float* result,
        int fullGroups, int tailRows, int blockCount, ComputeThreadPool? pool)
    {
        int m = fullGroups * 4 + tailRows;
        if (pool is null || m < ParallelMinRows)
        {
            ComputeRowsQ8_0Interleaved(repackedWeights, xQ8, result, fullGroups, tailRows, blockCount);
            return;
        }

        var ctx = new ComputeRowsR4Ctx
        {
            RepackedWeights = repackedWeights, XQ = xQ8, Result = result,
            M = m, FullGroups = fullGroups, TailRows = tailRows,
            BlockCount = blockCount, BlockBytes = Q8_0BlockBytes
        };
        pool.Dispatch((nint)(&ctx), &ComputeRowsQ8_0R4Worker);
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

    // ──────────────────── Vector256 optimized ────────────────────

    /// <summary>
    /// Vector256-accelerated Q8_0 dot product with FMA float accumulation.
    /// Uses the sign-flip trick for signed×signed multiply and accumulates
    /// in <c>Vector256&lt;float&gt;</c> across all blocks, performing a single
    /// horizontal sum at the end.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ8_0Vector256(byte* a, byte* b, int blockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;

        for (int block = 0; block < blockCount; block++)
        {
            byte* aBlock = a + block * Q8_0BlockBytes;
            byte* bBlock = b + block * Q8_0BlockBytes;

            float da = (float)Unsafe.ReadUnaligned<Half>(aBlock);
            float db = (float)Unsafe.ReadUnaligned<Half>(bBlock);

            Vector256<sbyte> va = Unsafe.ReadUnaligned<Vector256<sbyte>>(aBlock + 2);
            Vector256<sbyte> vb = Unsafe.ReadUnaligned<Vector256<sbyte>>(bBlock + 2);
            Vector256<float> fsum = Vector256.ConvertToSingle(DotProductSByte32(va, vb));
            Vector256<float> scale = Vector256.Create(da * db);

            acc = Vector256.FusedMultiplyAdd(scale, fsum, acc);
        }

        return Vector256.Sum(acc);
    }

    /// <summary>
    /// Vector256 multi-row (4 rows) Q8_0 dot product. Loads each xQ8 block once and
    /// computes dot products against 4 weight rows simultaneously.
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ8_0Vector256_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* x, int blockCount, float* results)
    {
        Vector256<float> acc0 = Vector256<float>.Zero;
        Vector256<float> acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero;
        Vector256<float> acc3 = Vector256<float>.Zero;

        for (int block = 0; block < blockCount; block++)
        {
            byte* xBlock = x + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);

            // Load x data once per block.
            Vector256<sbyte> vx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);

            // Row 0
            {
                byte* wBlock = w0 + block * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<float> fsum = Vector256.ConvertToSingle(DotProductSByte32(vw, vx));
                Vector256<float> scale = Vector256.Create(dx * dw);
                acc0 = Vector256.FusedMultiplyAdd(scale, fsum, acc0);
            }

            // Row 1
            {
                byte* wBlock = w1 + block * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<float> fsum = Vector256.ConvertToSingle(DotProductSByte32(vw, vx));
                Vector256<float> scale = Vector256.Create(dx * dw);
                acc1 = Vector256.FusedMultiplyAdd(scale, fsum, acc1);
            }

            // Row 2
            {
                byte* wBlock = w2 + block * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<float> fsum = Vector256.ConvertToSingle(DotProductSByte32(vw, vx));
                Vector256<float> scale = Vector256.Create(dx * dw);
                acc2 = Vector256.FusedMultiplyAdd(scale, fsum, acc2);
            }

            // Row 3
            {
                byte* wBlock = w3 + block * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
                Vector256<float> fsum = Vector256.ConvertToSingle(DotProductSByte32(vw, vx));
                Vector256<float> scale = Vector256.Create(dx * dw);
                acc3 = Vector256.FusedMultiplyAdd(scale, fsum, acc3);
            }
        }

        results[0] = Vector256.Sum(acc0);
        results[1] = Vector256.Sum(acc1);
        results[2] = Vector256.Sum(acc2);
        results[3] = Vector256.Sum(acc3);
    }

    // ──────────────────── Vector512 optimized ────────────────────

    /// <summary>
    /// Vector512-accelerated Q8_0 dot product. Processes 2 blocks (64 bytes) per iteration
    /// using <c>Vector512</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ8_0Vector512(byte* a, byte* b, int blockCount)
    {
        Vector512<float> acc = Vector512<float>.Zero;

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

            Vector256<sbyte> va0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(aBlock0 + 2);
            Vector256<sbyte> vb0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(bBlock0 + 2);
            Vector256<sbyte> va1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(aBlock1 + 2);
            Vector256<sbyte> vb1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(bBlock1 + 2);
            Vector256<int> isum0 = DotProductSByte32(va0, vb0);
            Vector256<int> isum1 = DotProductSByte32(va1, vb1);

            // Combine into 512-bit vectors.
            Vector512<int> isum512 = Vector512.Create(isum0, isum1);
            Vector512<float> fsum512 = Vector512.ConvertToSingle(isum512);

            // Dual scale: lower 8 lanes get da0*db0, upper 8 get da1*db1.
            Vector512<float> scale = Vector512.Create(
                Vector256.Create(da0 * db0),
                Vector256.Create(da1 * db1));

            acc = Vector512.FusedMultiplyAdd(fsum512, scale, acc);
        }

        float result = Vector512.Sum(acc);

        // Handle an odd trailing block via the Vector256 single-block path.
        if (block < blockCount)
        {
            byte* aBlock = a + block * Q8_0BlockBytes;
            byte* bBlock = b + block * Q8_0BlockBytes;

            float da = (float)Unsafe.ReadUnaligned<Half>(aBlock);
            float db = (float)Unsafe.ReadUnaligned<Half>(bBlock);

            Vector256<sbyte> va = Unsafe.ReadUnaligned<Vector256<sbyte>>(aBlock + 2);
            Vector256<sbyte> vb = Unsafe.ReadUnaligned<Vector256<sbyte>>(bBlock + 2);
            Vector256<float> fsum = Vector256.ConvertToSingle(DotProductSByte32(va, vb));

            result += da * db * Vector256.Sum(fsum);
        }

        return result;
    }

    /// <summary>
    /// Vector512 multi-row (4 rows) Q8_0 dot product. Processes 2 blocks per iteration
    /// and computes against 4 weight rows simultaneously.
    /// </summary>
    [SkipLocalsInit]
    internal static void VecDotQ8_0Vector512_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* x, int blockCount, float* results)
    {
        Vector512<float> acc0 = Vector512<float>.Zero;
        Vector512<float> acc1 = Vector512<float>.Zero;
        Vector512<float> acc2 = Vector512<float>.Zero;
        Vector512<float> acc3 = Vector512<float>.Zero;

        int block = 0;

        for (; block + 1 < blockCount; block += 2)
        {
            byte* xBlock0 = x + block * Q8_0BlockBytes;
            byte* xBlock1 = x + (block + 1) * Q8_0BlockBytes;
            float dx0 = (float)Unsafe.ReadUnaligned<Half>(xBlock0);
            float dx1 = (float)Unsafe.ReadUnaligned<Half>(xBlock1);

            Vector256<sbyte> vx0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock0 + 2);
            Vector256<sbyte> vx1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock1 + 2);

            // Process each weight row.
            ProcessVector512DualBlock(w0, block, vx0, vx1, dx0, dx1, ref acc0);
            ProcessVector512DualBlock(w1, block, vx0, vx1, dx0, dx1, ref acc1);
            ProcessVector512DualBlock(w2, block, vx0, vx1, dx0, dx1, ref acc2);
            ProcessVector512DualBlock(w3, block, vx0, vx1, dx0, dx1, ref acc3);
        }

        results[0] = Vector512.Sum(acc0);
        results[1] = Vector512.Sum(acc1);
        results[2] = Vector512.Sum(acc2);
        results[3] = Vector512.Sum(acc3);

        // Handle an odd trailing block via Vector256.
        if (block < blockCount)
        {
            byte* xBlock = x + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);
            Vector256<sbyte> vx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);

            results[0] += ProcessVector256SingleBlock(w0, block, vx, dx);
            results[1] += ProcessVector256SingleBlock(w1, block, vx, dx);
            results[2] += ProcessVector256SingleBlock(w2, block, vx, dx);
            results[3] += ProcessVector256SingleBlock(w3, block, vx, dx);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessVector512DualBlock(
        byte* w, int block,
        Vector256<sbyte> vx0, Vector256<sbyte> vx1,
        float dx0, float dx1,
        ref Vector512<float> acc)
    {
        byte* wBlock0 = w + block * Q8_0BlockBytes;
        byte* wBlock1 = w + (block + 1) * Q8_0BlockBytes;
        float dw0 = (float)Unsafe.ReadUnaligned<Half>(wBlock0);
        float dw1 = (float)Unsafe.ReadUnaligned<Half>(wBlock1);

        Vector256<sbyte> vw0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock0 + 2);
        Vector256<sbyte> vw1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock1 + 2);

        Vector256<int> isum0 = DotProductSByte32(vw0, vx0);
        Vector256<int> isum1 = DotProductSByte32(vw1, vx1);

        Vector512<int> isum512 = Vector512.Create(isum0, isum1);
        Vector512<float> fsum512 = Vector512.ConvertToSingle(isum512);

        Vector512<float> scale = Vector512.Create(
            Vector256.Create(dx0 * dw0),
            Vector256.Create(dx1 * dw1));

        acc = Vector512.FusedMultiplyAdd(fsum512, scale, acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ProcessVector256SingleBlock(
        byte* w, int block,
        Vector256<sbyte> vx,
        float dx)
    {
        byte* wBlock = w + block * Q8_0BlockBytes;
        float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
        Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);
        Vector256<float> fsum = Vector256.ConvertToSingle(DotProductSByte32(vw, vx));
        return dx * dw * Vector256.Sum(fsum);
    }

    // ──────────────────── Quantization ────────────────────

    /// <summary>
    /// Quantizes f32 data to Q8_0 format. Per block of 32 floats:
    /// scale = max(|x[i]|) / 127, qs[i] = round(x[i] / scale) clamped to [-127, 127].
    /// Dispatches to the widest available vector width, then falls back to scalar.
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

        if (Vector512.IsHardwareAccelerated)
            QuantizeF32ToQ8_0Vector512(src, dest, elementCount);
        else if (Vector256.IsHardwareAccelerated)
            QuantizeF32ToQ8_0Vector256(src, dest, elementCount);
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
    /// Vector256 SIMD quantization: processes 32 floats per block using 4 iterations of 8 floats.
    /// </summary>
    [SkipLocalsInit]
    internal static void QuantizeF32ToQ8_0Vector256(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / Q8_0GroupSize;
        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * Q8_0GroupSize;
            byte* blockDst = dest + block * Q8_0BlockBytes;
            ref float blockSrcRef = ref Unsafe.AsRef<float>(blockSrc);

            // Max-abs scan: 4 loads of 8 floats.
            Vector256<float> v0 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef));
            Vector256<float> v1 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef, 8));
            Vector256<float> v2 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef, 16));
            Vector256<float> v3 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef, 24));

            Vector256<float> max01 = Vector256.MaxNative(v0, v1);
            Vector256<float> max23 = Vector256.MaxNative(v2, v3);
            Vector256<float> maxAll = Vector256.MaxNative(max01, max23);
            float maxAbs = HorizontalMaxVector256(maxAll);

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

                Vector256<int> i0 = Vector256.ConvertToInt32Native(Vector256.Round(Vector256.LoadUnsafe(ref blockSrcRef) * vInvScale));
                Vector256<int> i1 = Vector256.ConvertToInt32Native(Vector256.Round(Vector256.LoadUnsafe(ref blockSrcRef, 8) * vInvScale));
                Vector256<int> i2 = Vector256.ConvertToInt32Native(Vector256.Round(Vector256.LoadUnsafe(ref blockSrcRef, 16) * vInvScale));
                Vector256<int> i3 = Vector256.ConvertToInt32Native(Vector256.Round(Vector256.LoadUnsafe(ref blockSrcRef, 24) * vInvScale));

                Vector256<short> s01 = Vector256.NarrowWithSaturation(i0, i1);
                Vector256<short> s23 = Vector256.NarrowWithSaturation(i2, i3);
                Vector256<sbyte> packed = Vector256.NarrowWithSaturation(s01, s23);

                packed.StoreUnsafe(ref Unsafe.AsRef<sbyte>(qs));
            }
        }
    }

    /// <summary>
    /// Vector512 SIMD quantization: processes 32 floats per block using 2 iterations of 16 floats.
    /// </summary>
    [SkipLocalsInit]
    internal static void QuantizeF32ToQ8_0Vector512(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / Q8_0GroupSize;

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * Q8_0GroupSize;
            byte* blockDst = dest + block * Q8_0BlockBytes;

            // Max-abs scan: 2 loads of 16 floats.
            Vector512<float> v0 = Vector512.Abs(Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc)));
            Vector512<float> v1 = Vector512.Abs(Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc + 16)));

            Vector512<float> maxAll = Vector512.MaxNative(v0, v1);
            // Reduce 512-bit to scalar max.
            Vector256<float> max256 = Vector256.MaxNative(maxAll.GetLower(), maxAll.GetUpper());
            float maxAbs = HorizontalMaxVector256(max256);

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

                Vector512<int> i0 = Vector512.ConvertToInt32Native(
                    Vector512.Round(Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc)) * vInvScale));
                Vector512<int> i1 = Vector512.ConvertToInt32Native(
                    Vector512.Round(Vector512.LoadUnsafe(ref Unsafe.AsRef<float>(blockSrc + 16)) * vInvScale));

                Vector512<short> s = Vector512.NarrowWithSaturation(i0, i1);
                Vector256<sbyte> packed = Vector256.NarrowWithSaturation(s.GetLower(), s.GetUpper());

                packed.StoreUnsafe(ref Unsafe.AsRef<sbyte>(qs));
            }
        }
    }

    // ──────────────────── Q8_1 Quantization ────────────────────

    /// <summary>
    /// Quantizes f32 data to Q8_1 format. Per block of 32 floats:
    /// d = max(|x[i]|) / 127, qs[i] = round(x[i] / d) clamped to [-127, 127],
    /// s = d * sum(qs[0..31]). Layout: Half d (2) + Half s (2) + sbyte[32] (32) = 36 bytes.
    /// Dispatches to Vector256 or falls back to scalar at runtime.
    /// </summary>
    /// <param name="src">Source f32 data. Must have <paramref name="elementCount"/> elements.</param>
    /// <param name="dest">Destination Q8_1 buffer. Must have (elementCount/32) × 36 bytes.</param>
    /// <param name="elementCount">Number of float elements. Must be a multiple of 32.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeF32ToQ8_1(float* src, byte* dest, int elementCount)
    {
        if (elementCount % Q8_1GroupSize != 0)
            throw new ArgumentException(
                $"elementCount must be a multiple of {Q8_1GroupSize}, got {elementCount}",
                nameof(elementCount));

        if (Vector256.IsHardwareAccelerated)
            QuantizeF32ToQ8_1Vector256(src, dest, elementCount);
        else
            QuantizeF32ToQ8_1Scalar(src, dest, elementCount);
    }

    /// <summary>
    /// Scalar Q8_1 quantization reference implementation.
    /// </summary>
    [SkipLocalsInit]
    internal static void QuantizeF32ToQ8_1Scalar(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / Q8_1GroupSize;

        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * Q8_1GroupSize;
            byte* blockDst = dest + block * Q8_1BlockBytes;

            // Find max absolute value.
            float maxAbs = 0;
            for (int i = 0; i < Q8_1GroupSize; i++)
            {
                float abs = MathF.Abs(blockSrc[i]);
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
                    int v = (int)MathF.Round(blockSrc[i] * invScale);
                    v = Math.Clamp(v, -127, 127);
                    qs[i] = (sbyte)v;
                    sum += v;
                }
                Unsafe.WriteUnaligned(blockDst + 2, (Half)(scale * sum));
            }
        }
    }

    /// <summary>
    /// Vector256 Q8_1 quantization. Same pack pipeline as Q8_0 but additionally computes
    /// <c>s = d * sum(qs)</c> from the int32 vectors before packing.
    /// </summary>
    [SkipLocalsInit]
    internal static void QuantizeF32ToQ8_1Vector256(float* src, byte* dest, int elementCount)
    {
        int blockCount = elementCount / Q8_1GroupSize;
        for (int block = 0; block < blockCount; block++)
        {
            float* blockSrc = src + block * Q8_1GroupSize;
            byte* blockDst = dest + block * Q8_1BlockBytes;
            ref float blockSrcRef = ref Unsafe.AsRef<float>(blockSrc);

            // Max-abs scan: 4 loads of 8 floats.
            Vector256<float> v0 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef));
            Vector256<float> v1 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef, 8));
            Vector256<float> v2 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef, 16));
            Vector256<float> v3 = Vector256.Abs(Vector256.LoadUnsafe(ref blockSrcRef, 24));

            Vector256<float> max01 = Vector256.MaxNative(v0, v1);
            Vector256<float> max23 = Vector256.MaxNative(v2, v3);
            Vector256<float> maxAll = Vector256.MaxNative(max01, max23);
            float maxAbs = HorizontalMaxVector256(maxAll);

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

                Vector256<int> i0 = Vector256.ConvertToInt32Native(Vector256.Round(Vector256.LoadUnsafe(ref blockSrcRef) * vInvScale));
                Vector256<int> i1 = Vector256.ConvertToInt32Native(Vector256.Round(Vector256.LoadUnsafe(ref blockSrcRef, 8) * vInvScale));
                Vector256<int> i2 = Vector256.ConvertToInt32Native(Vector256.Round(Vector256.LoadUnsafe(ref blockSrcRef, 16) * vInvScale));
                Vector256<int> i3 = Vector256.ConvertToInt32Native(Vector256.Round(Vector256.LoadUnsafe(ref blockSrcRef, 24) * vInvScale));

                // Clamp int32 to [-127, 127] before summing so the stored sum
                // matches the saturated qs values (cvtps2dq can produce out-of-range
                // values for non-finite inputs).
                Vector256<int> clampMin = Vector256.Create(-127);
                Vector256<int> clampMax = Vector256.Create(127);
                i0 = Vector256.Min(Vector256.Max(i0, clampMin), clampMax);
                i1 = Vector256.Min(Vector256.Max(i1, clampMin), clampMax);
                i2 = Vector256.Min(Vector256.Max(i2, clampMin), clampMax);
                i3 = Vector256.Min(Vector256.Max(i3, clampMin), clampMax);

                // Sum all 32 int32 values (2 vpaddd + Vector256.Sum)
                Vector256<int> isum = (i0 + i1) + (i2 + i3);
                int sum = Vector256.Sum(isum);

                Unsafe.WriteUnaligned(blockDst + 2, (Half)(scale * sum));

                Vector256<short> s01 = Vector256.NarrowWithSaturation(i0, i1);
                Vector256<short> s23 = Vector256.NarrowWithSaturation(i2, i3);
                Vector256<sbyte> packed = Vector256.NarrowWithSaturation(s01, s23);

                packed.StoreUnsafe(ref Unsafe.AsRef<sbyte>(qs));
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

    // ──────────────────── Outer-product tiled GEMM (R4 layout) ────────────────────

    /// <summary>
    /// Converts a Half (IEEE 754 binary16) at the given pointer to float.
    /// Extracted into a tiny helper so RyuJIT reliably emits <c>vcvtph2ps</c> (F16C) for the
    /// <c>Half→float</c> cast. In large methods with high register pressure, the JIT may fail
    /// to inline <c>Half.op_Explicit</c> and fall back to a software function call.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HalfBitsToFloat(byte* ptr)
    {
        return (float)Unsafe.ReadUnaligned<Half>(ptr);
    }

    /// <summary>
    /// Scalar reference outer-product microkernel for Q8_0 R4 layout.
    /// Computes 4 weight rows × 3 tokens simultaneously using R4-interleaved weights.
    /// Output layout: C[token * cStride + row] for token=0..2, row=0..3.
    /// </summary>
    [SkipLocalsInit]
    internal static void OuterProductQ8_0Scalar_4x3(
        byte* groupBase, byte* x0, byte* x1, byte* x2,
        float* c, int blockCount, int cStride)
    {
        const int wStride = 4 * Q8_0BlockBytes;

        // 12 accumulators: [row][token]
        float acc00 = 0, acc01 = 0, acc02 = 0;
        float acc10 = 0, acc11 = 0, acc12 = 0;
        float acc20 = 0, acc21 = 0, acc22 = 0;
        float acc30 = 0, acc31 = 0, acc32 = 0;

        for (int b = 0; b < blockCount; b++)
        {
            byte* blockBase = groupBase + b * wStride;
            byte* x0Block = x0 + b * Q8_0BlockBytes;
            byte* x1Block = x1 + b * Q8_0BlockBytes;
            byte* x2Block = x2 + b * Q8_0BlockBytes;

            float dx0 = (float)Unsafe.ReadUnaligned<Half>(x0Block);
            float dx1 = (float)Unsafe.ReadUnaligned<Half>(x1Block);
            float dx2 = (float)Unsafe.ReadUnaligned<Half>(x2Block);

            for (int r = 0; r < 4; r++)
            {
                byte* wBlock = blockBase + r * Q8_0BlockBytes;
                float dw = (float)Unsafe.ReadUnaligned<Half>(wBlock);
                sbyte* qw = (sbyte*)(wBlock + 2);

                // Token 0
                {
                    sbyte* qx = (sbyte*)(x0Block + 2);
                    int sumi = 0;
                    for (int i = 0; i < Q8_0GroupSize; i++)
                        sumi += qw[i] * qx[i];
                    float val = dw * dx0 * sumi;
                    switch (r) { case 0: acc00 += val; break; case 1: acc10 += val; break; case 2: acc20 += val; break; default: acc30 += val; break; }
                }
                // Token 1
                {
                    sbyte* qx = (sbyte*)(x1Block + 2);
                    int sumi = 0;
                    for (int i = 0; i < Q8_0GroupSize; i++)
                        sumi += qw[i] * qx[i];
                    float val = dw * dx1 * sumi;
                    switch (r) { case 0: acc01 += val; break; case 1: acc11 += val; break; case 2: acc21 += val; break; default: acc31 += val; break; }
                }
                // Token 2
                {
                    sbyte* qx = (sbyte*)(x2Block + 2);
                    int sumi = 0;
                    for (int i = 0; i < Q8_0GroupSize; i++)
                        sumi += qw[i] * qx[i];
                    float val = dw * dx2 * sumi;
                    switch (r) { case 0: acc02 += val; break; case 1: acc12 += val; break; case 2: acc22 += val; break; default: acc32 += val; break; }
                }
            }
        }

        c[0 * cStride + 0] = acc00; c[0 * cStride + 1] = acc10; c[0 * cStride + 2] = acc20; c[0 * cStride + 3] = acc30;
        c[1 * cStride + 0] = acc01; c[1 * cStride + 1] = acc11; c[1 * cStride + 2] = acc21; c[1 * cStride + 3] = acc31;
        c[2 * cStride + 0] = acc02; c[2 * cStride + 1] = acc12; c[2 * cStride + 2] = acc22; c[2 * cStride + 3] = acc32;
    }

    // TODO: Experiment with 2×3 tile (2 rows × 3 tokens): 6 acc + 6 token + 1 ones + 3 temps = 16 YMM.
    // This would process rows in pairs instead of individually, reducing token reloads by 2×
    // while staying within the 16 YMM register budget. Needs benchmarking.

    /// <summary>
    /// Vector256 outer-product microkernel for Q8_0 R4 layout.
    /// Processes 4 weight rows × 3 tokens with 12 YMM accumulators, 1 <c>ones</c>, 3 temporaries = 16 YMM.
    /// Weight block is loaded once and reused across 3 tokens (3× cache reuse vs inner-product).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void OuterProductQ8_0Vector256_4x3(
        byte* groupBase, byte* x0, byte* x1, byte* x2,
        float* c, int blockCount, int cStride)
    {
        const int wStride = 4 * Q8_0BlockBytes;

        // Process one weight row at a time with only 3 accumulators (one per token).
        // This keeps register pressure at ~10 YMM (3 acc + 1 ones + 3 vx + 1 absX + 1 vw + 1 temp),
        // well within the 16 YMM budget. Token blocks are reloaded per row but hit L1 cache.
        for (int r = 0; r < 4; r++)
        {
            Vector256<float> a0 = Vector256<float>.Zero;
            Vector256<float> a1 = Vector256<float>.Zero;
            Vector256<float> a2 = Vector256<float>.Zero;

            for (int b = 0; b < blockCount; b++)
            {
                byte* wBlock = groupBase + b * wStride + r * Q8_0BlockBytes;
                float dw = HalfBitsToFloat(wBlock);
                Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wBlock + 2);

                // Token 0
                byte* xb0 = x0 + b * Q8_0BlockBytes;
                float dx0 = HalfBitsToFloat(xb0);
                Vector256<sbyte> vx0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb0 + 2);
                Vector256<int> isum0 = DotProductSByte32(vw, vx0);
                a0 = Vector256.FusedMultiplyAdd(Vector256.Create(dx0 * dw), Vector256.ConvertToSingle(isum0), a0);

                // Token 1
                byte* xb1 = x1 + b * Q8_0BlockBytes;
                float dx1 = HalfBitsToFloat(xb1);
                Vector256<sbyte> vx1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb1 + 2);
                Vector256<int> isum1 = DotProductSByte32(vw, vx1);
                a1 = Vector256.FusedMultiplyAdd(Vector256.Create(dx1 * dw), Vector256.ConvertToSingle(isum1), a1);

                // Token 2
                byte* xb2 = x2 + b * Q8_0BlockBytes;
                float dx2 = HalfBitsToFloat(xb2);
                Vector256<sbyte> vx2 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb2 + 2);
                Vector256<int> isum2 = DotProductSByte32(vw, vx2);
                a2 = Vector256.FusedMultiplyAdd(Vector256.Create(dx2 * dw), Vector256.ConvertToSingle(isum2), a2);
            }

            c[0 * cStride + r] = Vector256.Sum(a0);
            c[1 * cStride + r] = Vector256.Sum(a1);
            c[2 * cStride + r] = Vector256.Sum(a2);
        }
    }

    /// <summary>
    /// Vector512 outer-product microkernel for Q8_0 R4 layout.
    /// Processes 4 weight rows × 6 tokens with 24 ZMM accumulators via dual-block (2 blocks/iteration).
    /// Uses 256-bit sign trick on each block half, then combines into 512-bit for FMA.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void OuterProductQ8_0Vector512_4x6(
        byte* groupBase, byte* x0, byte* x1, byte* x2, byte* x3, byte* x4, byte* x5,
        float* c, int blockCount, int cStride)
    {
        const int wStride = 4 * Q8_0BlockBytes;

        // 24 accumulators: acc_r{row}t{token}
        Vector512<float> a0t0 = Vector512<float>.Zero, a0t1 = Vector512<float>.Zero, a0t2 = Vector512<float>.Zero;
        Vector512<float> a0t3 = Vector512<float>.Zero, a0t4 = Vector512<float>.Zero, a0t5 = Vector512<float>.Zero;
        Vector512<float> a1t0 = Vector512<float>.Zero, a1t1 = Vector512<float>.Zero, a1t2 = Vector512<float>.Zero;
        Vector512<float> a1t3 = Vector512<float>.Zero, a1t4 = Vector512<float>.Zero, a1t5 = Vector512<float>.Zero;
        Vector512<float> a2t0 = Vector512<float>.Zero, a2t1 = Vector512<float>.Zero, a2t2 = Vector512<float>.Zero;
        Vector512<float> a2t3 = Vector512<float>.Zero, a2t4 = Vector512<float>.Zero, a2t5 = Vector512<float>.Zero;
        Vector512<float> a3t0 = Vector512<float>.Zero, a3t1 = Vector512<float>.Zero, a3t2 = Vector512<float>.Zero;
        Vector512<float> a3t3 = Vector512<float>.Zero, a3t4 = Vector512<float>.Zero, a3t5 = Vector512<float>.Zero;
        int block = 0;

        // Process 2 blocks per iteration (512-bit)
        for (; block + 1 < blockCount; block += 2)
        {
            byte* blockBase0 = groupBase + block * wStride;
            byte* blockBase1 = groupBase + (block + 1) * wStride;

            // Load 6 token pairs
            LoadDualBlockToken(x0, block, out float dx0_0, out float dx0_1, out var vx0Lo, out var vx0Hi);
            LoadDualBlockToken(x1, block, out float dx1_0, out float dx1_1, out var vx1Lo, out var vx1Hi);
            LoadDualBlockToken(x2, block, out float dx2_0, out float dx2_1, out var vx2Lo, out var vx2Hi);
            LoadDualBlockToken(x3, block, out float dx3_0, out float dx3_1, out var vx3Lo, out var vx3Hi);
            LoadDualBlockToken(x4, block, out float dx4_0, out float dx4_1, out var vx4Lo, out var vx4Hi);
            LoadDualBlockToken(x5, block, out float dx5_0, out float dx5_1, out var vx5Lo, out var vx5Hi);

            // Row 0
            {
                byte* wb0 = blockBase0;
                byte* wb1 = blockBase1;
                float dw0 = HalfBitsToFloat(wb0);
                float dw1 = HalfBitsToFloat(wb1);
                Vector256<sbyte> vwLo = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb0 + 2);
                Vector256<sbyte> vwHi = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb1 + 2);

                Vector512DualBlockFma(vwLo, vwHi, vx0Lo, vx0Hi, dx0_0 * dw0, dx0_1 * dw1, ref a0t0);
                Vector512DualBlockFma(vwLo, vwHi, vx1Lo, vx1Hi, dx1_0 * dw0, dx1_1 * dw1, ref a0t1);
                Vector512DualBlockFma(vwLo, vwHi, vx2Lo, vx2Hi, dx2_0 * dw0, dx2_1 * dw1, ref a0t2);
                Vector512DualBlockFma(vwLo, vwHi, vx3Lo, vx3Hi, dx3_0 * dw0, dx3_1 * dw1, ref a0t3);
                Vector512DualBlockFma(vwLo, vwHi, vx4Lo, vx4Hi, dx4_0 * dw0, dx4_1 * dw1, ref a0t4);
                Vector512DualBlockFma(vwLo, vwHi, vx5Lo, vx5Hi, dx5_0 * dw0, dx5_1 * dw1, ref a0t5);
            }

            // Row 1
            {
                byte* wb0 = blockBase0 + Q8_0BlockBytes;
                byte* wb1 = blockBase1 + Q8_0BlockBytes;
                float dw0 = HalfBitsToFloat(wb0);
                float dw1 = HalfBitsToFloat(wb1);
                Vector256<sbyte> vwLo = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb0 + 2);
                Vector256<sbyte> vwHi = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb1 + 2);

                Vector512DualBlockFma(vwLo, vwHi, vx0Lo, vx0Hi, dx0_0 * dw0, dx0_1 * dw1, ref a1t0);
                Vector512DualBlockFma(vwLo, vwHi, vx1Lo, vx1Hi, dx1_0 * dw0, dx1_1 * dw1, ref a1t1);
                Vector512DualBlockFma(vwLo, vwHi, vx2Lo, vx2Hi, dx2_0 * dw0, dx2_1 * dw1, ref a1t2);
                Vector512DualBlockFma(vwLo, vwHi, vx3Lo, vx3Hi, dx3_0 * dw0, dx3_1 * dw1, ref a1t3);
                Vector512DualBlockFma(vwLo, vwHi, vx4Lo, vx4Hi, dx4_0 * dw0, dx4_1 * dw1, ref a1t4);
                Vector512DualBlockFma(vwLo, vwHi, vx5Lo, vx5Hi, dx5_0 * dw0, dx5_1 * dw1, ref a1t5);
            }

            // Row 2
            {
                byte* wb0 = blockBase0 + 2 * Q8_0BlockBytes;
                byte* wb1 = blockBase1 + 2 * Q8_0BlockBytes;
                float dw0 = HalfBitsToFloat(wb0);
                float dw1 = HalfBitsToFloat(wb1);
                Vector256<sbyte> vwLo = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb0 + 2);
                Vector256<sbyte> vwHi = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb1 + 2);

                Vector512DualBlockFma(vwLo, vwHi, vx0Lo, vx0Hi, dx0_0 * dw0, dx0_1 * dw1, ref a2t0);
                Vector512DualBlockFma(vwLo, vwHi, vx1Lo, vx1Hi, dx1_0 * dw0, dx1_1 * dw1, ref a2t1);
                Vector512DualBlockFma(vwLo, vwHi, vx2Lo, vx2Hi, dx2_0 * dw0, dx2_1 * dw1, ref a2t2);
                Vector512DualBlockFma(vwLo, vwHi, vx3Lo, vx3Hi, dx3_0 * dw0, dx3_1 * dw1, ref a2t3);
                Vector512DualBlockFma(vwLo, vwHi, vx4Lo, vx4Hi, dx4_0 * dw0, dx4_1 * dw1, ref a2t4);
                Vector512DualBlockFma(vwLo, vwHi, vx5Lo, vx5Hi, dx5_0 * dw0, dx5_1 * dw1, ref a2t5);
            }

            // Row 3
            {
                byte* wb0 = blockBase0 + 3 * Q8_0BlockBytes;
                byte* wb1 = blockBase1 + 3 * Q8_0BlockBytes;
                float dw0 = HalfBitsToFloat(wb0);
                float dw1 = HalfBitsToFloat(wb1);
                Vector256<sbyte> vwLo = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb0 + 2);
                Vector256<sbyte> vwHi = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb1 + 2);

                Vector512DualBlockFma(vwLo, vwHi, vx0Lo, vx0Hi, dx0_0 * dw0, dx0_1 * dw1, ref a3t0);
                Vector512DualBlockFma(vwLo, vwHi, vx1Lo, vx1Hi, dx1_0 * dw0, dx1_1 * dw1, ref a3t1);
                Vector512DualBlockFma(vwLo, vwHi, vx2Lo, vx2Hi, dx2_0 * dw0, dx2_1 * dw1, ref a3t2);
                Vector512DualBlockFma(vwLo, vwHi, vx3Lo, vx3Hi, dx3_0 * dw0, dx3_1 * dw1, ref a3t3);
                Vector512DualBlockFma(vwLo, vwHi, vx4Lo, vx4Hi, dx4_0 * dw0, dx4_1 * dw1, ref a3t4);
                Vector512DualBlockFma(vwLo, vwHi, vx5Lo, vx5Hi, dx5_0 * dw0, dx5_1 * dw1, ref a3t5);
            }
        }

        // Store results: c[token * cStride + row]
        c[0 * cStride + 0] = Vector512.Sum(a0t0);
        c[0 * cStride + 1] = Vector512.Sum(a1t0);
        c[0 * cStride + 2] = Vector512.Sum(a2t0);
        c[0 * cStride + 3] = Vector512.Sum(a3t0);
        c[1 * cStride + 0] = Vector512.Sum(a0t1);
        c[1 * cStride + 1] = Vector512.Sum(a1t1);
        c[1 * cStride + 2] = Vector512.Sum(a2t1);
        c[1 * cStride + 3] = Vector512.Sum(a3t1);
        c[2 * cStride + 0] = Vector512.Sum(a0t2);
        c[2 * cStride + 1] = Vector512.Sum(a1t2);
        c[2 * cStride + 2] = Vector512.Sum(a2t2);
        c[2 * cStride + 3] = Vector512.Sum(a3t2);
        c[3 * cStride + 0] = Vector512.Sum(a0t3);
        c[3 * cStride + 1] = Vector512.Sum(a1t3);
        c[3 * cStride + 2] = Vector512.Sum(a2t3);
        c[3 * cStride + 3] = Vector512.Sum(a3t3);
        c[4 * cStride + 0] = Vector512.Sum(a0t4);
        c[4 * cStride + 1] = Vector512.Sum(a1t4);
        c[4 * cStride + 2] = Vector512.Sum(a2t4);
        c[4 * cStride + 3] = Vector512.Sum(a3t4);
        c[5 * cStride + 0] = Vector512.Sum(a0t5);
        c[5 * cStride + 1] = Vector512.Sum(a1t5);
        c[5 * cStride + 2] = Vector512.Sum(a2t5);
        c[5 * cStride + 3] = Vector512.Sum(a3t5);

        // Handle an odd trailing block via Vector256.
        if (block < blockCount)
        {
            byte* blockBase = groupBase + block * wStride;
            // Process each token's trailing block against all 4 weight rows
            ProcessVector512TailBlock(blockBase, x0, block, c, cStride, 0);
            ProcessVector512TailBlock(blockBase, x1, block, c, cStride, 1);
            ProcessVector512TailBlock(blockBase, x2, block, c, cStride, 2);
            ProcessVector512TailBlock(blockBase, x3, block, c, cStride, 3);
            ProcessVector512TailBlock(blockBase, x4, block, c, cStride, 4);
            ProcessVector512TailBlock(blockBase, x5, block, c, cStride, 5);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LoadDualBlockToken(byte* x, int block,
        out float dx0, out float dx1,
        out Vector256<sbyte> vxLo, out Vector256<sbyte> vxHi)
    {
        byte* xb0 = x + block * Q8_0BlockBytes;
        byte* xb1 = x + (block + 1) * Q8_0BlockBytes;
        dx0 = HalfBitsToFloat(xb0);
        dx1 = HalfBitsToFloat(xb1);
        vxLo = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb0 + 2);
        vxHi = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb1 + 2);
    }

    // TODO: Check disasm on 512-bit hardware — Vector512.Create(vec256, vec256) may emit
    // extra insert work instead of staying fully in ZMM. Revisit if it shows up in benchmarks.

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Vector512DualBlockFma(
        Vector256<sbyte> vwLo, Vector256<sbyte> vwHi,
        Vector256<sbyte> vxLo, Vector256<sbyte> vxHi,
        float scale0, float scale1,
        ref Vector512<float> acc)
    {
        Vector256<int> isum0 = DotProductSByte32(vwLo, vxLo);
        Vector256<int> isum1 = DotProductSByte32(vwHi, vxHi);

        Vector512<int> isum512 = Vector512.Create(isum0, isum1);
        Vector512<float> fsum = Vector512.ConvertToSingle(isum512);
        Vector512<float> scale = Vector512.Create(
            Vector256.Create(scale0),
            Vector256.Create(scale1));

        acc = Vector512.FusedMultiplyAdd(fsum, scale, acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessVector512TailBlock(byte* blockBase, byte* x, int block,
        float* c, int cStride, int tokenIdx)
    {
        byte* xBlock = x + block * Q8_0BlockBytes;
        float dxt = HalfBitsToFloat(xBlock);
        Vector256<sbyte> vxt = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);

        for (int r = 0; r < 4; r++)
        {
            byte* wb = blockBase + r * Q8_0BlockBytes;
            float dw = HalfBitsToFloat(wb);
            Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wb + 2);
            Vector256<float> fsum = Vector256.ConvertToSingle(DotProductSByte32(vw, vxt));
            c[tokenIdx * cStride + r] += dxt * dw * Vector256.Sum(fsum);
        }
    }

    /// <summary>
    /// Outer-product GEMM for Q8_0 R4-interleaved weights.
    /// Processes weight groups in steps of 4 rows and token batches of 3 (Vector256) or 6 (Vector512).
    /// Falls back to <see cref="VecDotQ8_0Vector256_4RowsR4"/> for tail tokens
    /// and <see cref="VecDotQ8_0Vector256"/>/<see cref="VecDotQ8_0Scalar"/> for tail rows.
    /// </summary>
    /// <param name="repackedWeights">R4-interleaved Q8_0 weight data.</param>
    /// <param name="inputQ8">Pre-quantized Q8_0 input [N × q8RowBytes].</param>
    /// <param name="c">Output matrix [N × M], row-major (C[token * m + row]).</param>
    /// <param name="fullGroups">Number of complete R4 groups (M / 4).</param>
    /// <param name="tailRows">Remaining rows after full groups (M % 4).</param>
    /// <param name="blockCount">Blocks per row (K / 32).</param>
    /// <param name="m">Total output rows.</param>
    /// <param name="n">Number of tokens (batch size).</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void OuterProductGemmQ8_0(byte* repackedWeights, byte* inputQ8, float* c,
        int fullGroups, int tailRows, int blockCount, int m, int n)
    {
        int q8RowBytes = blockCount * Q8_0BlockBytes;
        int groupBytes = 4 * q8RowBytes;

        for (int g = 0; g < fullGroups; g++)
        {
            byte* groupBase = repackedWeights + (long)g * groupBytes;
            int baseRow = g * 4;
            int t = 0;

            if (Vector512.IsHardwareAccelerated)
            {
                // Vector512: 4×6 tiles
                int nFull6 = (n / 6) * 6;
                for (; t < nFull6; t += 6)
                {
                    OuterProductQ8_0Vector512_4x6(
                        groupBase,
                        inputQ8 + (long)t * q8RowBytes,
                        inputQ8 + (long)(t + 1) * q8RowBytes,
                        inputQ8 + (long)(t + 2) * q8RowBytes,
                        inputQ8 + (long)(t + 3) * q8RowBytes,
                        inputQ8 + (long)(t + 4) * q8RowBytes,
                        inputQ8 + (long)(t + 5) * q8RowBytes,
                        c + (long)t * m + baseRow, blockCount, m);
                }
                // Remaining tokens: use Vector256 4×3 tiles
                int nFull3 = t + ((n - t) / 3) * 3;
                for (; t < nFull3; t += 3)
                {
                    OuterProductQ8_0Vector256_4x3(
                        groupBase,
                        inputQ8 + (long)t * q8RowBytes,
                        inputQ8 + (long)(t + 1) * q8RowBytes,
                        inputQ8 + (long)(t + 2) * q8RowBytes,
                        c + (long)t * m + baseRow, blockCount, m);
                }
                // Single tail tokens
                for (; t < n; t++)
                {
                    VecDotQ8_0Vector256_4RowsR4(groupBase, inputQ8 + (long)t * q8RowBytes,
                        blockCount, c + (long)t * m + baseRow);
                }
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                // Vector256: 4×3 tiles
                int nFull3 = (n / 3) * 3;
                for (; t < nFull3; t += 3)
                {
                    OuterProductQ8_0Vector256_4x3(
                        groupBase,
                        inputQ8 + (long)t * q8RowBytes,
                        inputQ8 + (long)(t + 1) * q8RowBytes,
                        inputQ8 + (long)(t + 2) * q8RowBytes,
                        c + (long)t * m + baseRow, blockCount, m);
                }
                // Tail tokens
                for (; t < n; t++)
                {
                    VecDotQ8_0Vector256_4RowsR4(groupBase, inputQ8 + (long)t * q8RowBytes,
                        blockCount, c + (long)t * m + baseRow);
                }
            }
            else
            {
                // Scalar fallback
                int nFull3 = (n / 3) * 3;
                for (; t < nFull3; t += 3)
                {
                    OuterProductQ8_0Scalar_4x3(
                        groupBase,
                        inputQ8 + (long)t * q8RowBytes,
                        inputQ8 + (long)(t + 1) * q8RowBytes,
                        inputQ8 + (long)(t + 2) * q8RowBytes,
                        c + (long)t * m + baseRow, blockCount, m);
                }
                for (; t < n; t++)
                {
                    for (int r = 0; r < 4; r++)
                        c[(long)t * m + baseRow + r] = VecDotQ8_0ScalarR4(
                            groupBase, r, inputQ8 + (long)t * q8RowBytes, blockCount);
                }
            }
        }

        // Tail rows (row-major layout, not interleaved)
        if (tailRows > 0)
        {
            int rowBytes = blockCount * Q8_0BlockBytes;
            byte* tailBase = repackedWeights + (long)fullGroups * groupBytes;
            int baseRow = fullGroups * 4;

            for (int t = 0; t < n; t++)
            {
                byte* xQ8 = inputQ8 + (long)t * q8RowBytes;
                for (int r = 0; r < tailRows; r++)
                {
                    c[(long)t * m + baseRow + r] = Vector256.IsHardwareAccelerated
                        ? VecDotQ8_0Vector256(tailBase + (long)r * rowBytes, xQ8, blockCount)
                        : VecDotQ8_0Scalar(tailBase + (long)r * rowBytes, xQ8, blockCount);
                }
            }
        }
    }

    /// <summary>Context for parallel outer-product Q8_0 GEMM dispatch.</summary>
    private struct OuterProductGemmQ8Ctx
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
    /// Parallel outer-product GEMM for Q8_0 R4 weights.
    /// Partitions R4 groups across threads. Last thread handles tail rows.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void OuterProductGemmQ8_0(byte* repackedWeights, byte* inputQ8, float* c,
        int fullGroups, int tailRows, int blockCount, int m, int n, ComputeThreadPool? pool)
    {
        if (pool is null || m < ParallelMinRows)
        {
            OuterProductGemmQ8_0(repackedWeights, inputQ8, c, fullGroups, tailRows, blockCount, m, n);
            return;
        }

        var ctx = new OuterProductGemmQ8Ctx
        {
            RepackedWeights = repackedWeights, InputQ8 = inputQ8, C = c,
            FullGroups = fullGroups, TailRows = tailRows,
            BlockCount = blockCount, M = m, N = n
        };
        pool.Dispatch((nint)(&ctx), &OuterProductGemmQ8_0Worker);
    }

    private static void OuterProductGemmQ8_0Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<OuterProductGemmQ8Ctx>((void*)ctxPtr);

        // Partition groups across threads
        int totalGroups = ctx.FullGroups + (ctx.TailRows > 0 ? 1 : 0);
        int groupsPerThread = (totalGroups + threadCount - 1) / threadCount;
        int startGroup = threadIdx * groupsPerThread;
        int endGroup = Math.Min(startGroup + groupsPerThread, totalGroups);

        if (startGroup >= totalGroups) return;

        int q8RowBytes = ctx.BlockCount * Q8_0BlockBytes;
        int groupBytes = 4 * q8RowBytes;

        for (int g = startGroup; g < endGroup; g++)
        {
            if (g < ctx.FullGroups)
            {
                byte* groupBase = ctx.RepackedWeights + (long)g * groupBytes;
                int baseRow = g * 4;
                int t = 0;

                if (Vector512.IsHardwareAccelerated)
                {
                    int nFull6 = (ctx.N / 6) * 6;
                    for (; t < nFull6; t += 6)
                    {
                        OuterProductQ8_0Vector512_4x6(
                            groupBase,
                            ctx.InputQ8 + (long)t * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 1) * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 2) * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 3) * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 4) * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 5) * q8RowBytes,
                            ctx.C + (long)t * ctx.M + baseRow, ctx.BlockCount, ctx.M);
                    }
                    int nFull3 = t + ((ctx.N - t) / 3) * 3;
                    for (; t < nFull3; t += 3)
                    {
                        OuterProductQ8_0Vector256_4x3(
                            groupBase,
                            ctx.InputQ8 + (long)t * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 1) * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 2) * q8RowBytes,
                            ctx.C + (long)t * ctx.M + baseRow, ctx.BlockCount, ctx.M);
                    }
                    for (; t < ctx.N; t++)
                    {
                        VecDotQ8_0Vector256_4RowsR4(groupBase, ctx.InputQ8 + (long)t * q8RowBytes,
                            ctx.BlockCount, ctx.C + (long)t * ctx.M + baseRow);
                    }
                }
                else if (Vector256.IsHardwareAccelerated)
                {
                    int nFull3 = (ctx.N / 3) * 3;
                    for (; t < nFull3; t += 3)
                    {
                        OuterProductQ8_0Vector256_4x3(
                            groupBase,
                            ctx.InputQ8 + (long)t * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 1) * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 2) * q8RowBytes,
                            ctx.C + (long)t * ctx.M + baseRow, ctx.BlockCount, ctx.M);
                    }
                    for (; t < ctx.N; t++)
                    {
                        VecDotQ8_0Vector256_4RowsR4(groupBase, ctx.InputQ8 + (long)t * q8RowBytes,
                            ctx.BlockCount, ctx.C + (long)t * ctx.M + baseRow);
                    }
                }
                else
                {
                    int nFull3 = (ctx.N / 3) * 3;
                    for (; t < nFull3; t += 3)
                    {
                        OuterProductQ8_0Scalar_4x3(
                            groupBase,
                            ctx.InputQ8 + (long)t * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 1) * q8RowBytes,
                            ctx.InputQ8 + (long)(t + 2) * q8RowBytes,
                            ctx.C + (long)t * ctx.M + baseRow, ctx.BlockCount, ctx.M);
                    }
                    for (; t < ctx.N; t++)
                    {
                        for (int r = 0; r < 4; r++)
                            ctx.C[(long)t * ctx.M + baseRow + r] = VecDotQ8_0ScalarR4(
                                groupBase, r, ctx.InputQ8 + (long)t * q8RowBytes, ctx.BlockCount);
                    }
                }
            }
            else
            {
                // Tail rows
                byte* tailBase = ctx.RepackedWeights + (long)ctx.FullGroups * groupBytes;
                int baseRow = ctx.FullGroups * 4;
                int rowBytes = ctx.BlockCount * Q8_0BlockBytes;
                for (int t = 0; t < ctx.N; t++)
                {
                    byte* xQ8 = ctx.InputQ8 + (long)t * q8RowBytes;
                    for (int r = 0; r < ctx.TailRows; r++)
                    {
                        ctx.C[(long)t * ctx.M + baseRow + r] = Vector256.IsHardwareAccelerated
                            ? VecDotQ8_0Vector256(tailBase + (long)r * rowBytes, xQ8, ctx.BlockCount)
                            : VecDotQ8_0Scalar(tailBase + (long)r * rowBytes, xQ8, ctx.BlockCount);
                    }
                }
            }
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

    private struct ComputeRowsR4Ctx
    {
        public byte* RepackedWeights;
        public byte* XQ;
        public float* Result;
        public int M;
        public int FullGroups;
        public int TailRows;
        public int BlockCount;
        public int BlockBytes;
    }

    // ── Worker methods ──

    private static void ComputeRowsQ8_0R4Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<ComputeRowsR4Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;

        int groupBytes = 4 * ctx.BlockCount * Q8_0BlockBytes;
        int rowBytes = ctx.BlockCount * Q8_0BlockBytes;
        int end = start + count;

        // Process full groups in [start, end)
        int startGroup = start / 4;
        int endGroup = Math.Min(end / 4, ctx.FullGroups);
        for (int g = startGroup; g < endGroup; g++)
        {
            byte* groupBase = ctx.RepackedWeights + (long)g * groupBytes;
            if (Vector256.IsHardwareAccelerated)
                VecDotQ8_0Vector256_4RowsR4(groupBase, ctx.XQ, ctx.BlockCount, ctx.Result + g * 4);
            else
                for (int r = 0; r < 4; r++)
                    ctx.Result[g * 4 + r] = VecDotQ8_0ScalarR4(groupBase, r, ctx.XQ, ctx.BlockCount);
        }

        // Process tail rows if they fall within this thread's range
        if (ctx.TailRows > 0 && end > ctx.FullGroups * 4)
        {
            int tailStart = Math.Max(start, ctx.FullGroups * 4) - ctx.FullGroups * 4;
            int tailEnd = Math.Min(end, ctx.M) - ctx.FullGroups * 4;
            byte* tailBase = ctx.RepackedWeights + (long)ctx.FullGroups * groupBytes;
            for (int r = tailStart; r < tailEnd; r++)
                ctx.Result[ctx.FullGroups * 4 + r] = Vector256.IsHardwareAccelerated
                    ? VecDotQ8_0Vector256(tailBase + (long)r * rowBytes, ctx.XQ, ctx.BlockCount)
                    : VecDotQ8_0Scalar(tailBase + (long)r * rowBytes, ctx.XQ, ctx.BlockCount);
        }
    }

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

    // ──────────────────── Horizontal reduction helpers ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> DotProductSByte32(Vector256<sbyte> left, Vector256<sbyte> right)
    {
        Vector256<short> left16Lo = Vector256.WidenLower(left);
        Vector256<short> left16Hi = Vector256.WidenUpper(left);
        Vector256<short> right16Lo = Vector256.WidenLower(right);
        Vector256<short> right16Hi = Vector256.WidenUpper(right);

        return PairwiseSumInt16Products(left16Lo * right16Lo)
            + PairwiseSumInt16Products(left16Hi * right16Hi);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> DotProductByte32SByte32(Vector256<byte> left, Vector256<sbyte> right)
    {
        Vector256<short> left16Lo = Vector256.WidenLower(left).AsInt16();
        Vector256<short> left16Hi = Vector256.WidenUpper(left).AsInt16();
        Vector256<short> right16Lo = Vector256.WidenLower(right);
        Vector256<short> right16Hi = Vector256.WidenUpper(right);

        return PairwiseSumInt16Products(left16Lo * right16Lo)
            + PairwiseSumInt16Products(left16Hi * right16Hi);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> PairwiseSumInt16Products(Vector256<short> products)
    {
        // Each int32 lane is two adjacent int16 products packed together.
        Vector256<int> lanes = products.AsInt32();
        Vector256<int> low = Vector256.ShiftRightArithmetic(Vector256.ShiftLeft(lanes, 16), 16);
        Vector256<int> high = Vector256.ShiftRightArithmetic(lanes, 16);
        return low + high;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMaxVector256(Vector256<float> v)
    {
        Vector128<float> lo = v.GetLower();
        Vector128<float> hi = v.GetUpper();
        Vector128<float> max128 = Vector128.MaxNative(lo, hi);

        // Shuffle and max to reduce.
        Vector128<float> shuf = Vector128.ShuffleNative(max128, Vector128.Create(2, 3, 0, 1));
        max128 = Vector128.MaxNative(max128, shuf);
        shuf = Vector128.ShuffleNative(max128, Vector128.Create(1, 0, 3, 2));
        max128 = Vector128.MaxNative(max128, shuf);

        return max128.ToScalar();
    }
}
