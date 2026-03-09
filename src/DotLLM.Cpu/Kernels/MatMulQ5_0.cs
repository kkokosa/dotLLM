using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Q5_0 × Q8_0 matrix multiplication kernels.
/// Q5_0 block: 22 bytes = Half(d) + uint32(qh) + byte[16](qs) → 32 elements.
/// Input is quantized to Q8_0, then fused integer dot product avoids dequant-to-float.
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>Q5_0 block size in bytes: 2 (Half d) + 4 (uint32 qh) + 16 (byte[16] qs).</summary>
    private const int Q5_0BlockBytes = 22;

    /// <summary>Number of elements per Q5_0 block.</summary>
    private const int Q5_0GroupSize = 32;

    // ──────────────────── Q5_0 × Q8_0 Scalar ────────────────────

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

    // ──────────────────── Q5_0 × Q8_0 AVX2 ────────────────────

    /// <summary>
    /// AVX2-accelerated Q5_0 × Q8_0 dot product.
    /// Unpacks 5-bit values from Q5_0 blocks, computes <c>sum(q5_unsigned * q8) - 16 * sum(q8)</c>
    /// to avoid per-element subtraction.
    /// </summary>
    [SkipLocalsInit]
    internal static float VecDotQ5_0Q8_0Avx2(byte* q5, byte* q8, int blockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<short> ones = Vector256.Create((short)1);
        byte* expanded = stackalloc byte[32];

        for (int block = 0; block < blockCount; block++)
        {
            byte* q5Block = q5 + block * Q5_0BlockBytes;
            byte* q8Block = q8 + block * Q8_0BlockBytes;

            float d5 = (float)Unsafe.ReadUnaligned<Half>(q5Block);
            float d8 = (float)Unsafe.ReadUnaligned<Half>(q8Block);

            uint qh = Unsafe.ReadUnaligned<uint>(q5Block + 2);

            // Unpack 16 nibble bytes → 32 4-bit values: lo nibbles [0..15], hi nibbles [16..31]
            Vector128<byte> qsRaw = Unsafe.ReadUnaligned<Vector128<byte>>(q5Block + 6);
            Vector128<byte> lo128 = Sse2.And(qsRaw, Vector128.Create((byte)0x0F));
            Vector128<byte> hi128 = Sse2.And(
                Sse2.ShiftRightLogical(qsRaw.AsUInt16(), 4).AsByte(),
                Vector128.Create((byte)0x0F));
            Vector256<byte> nibbles = Vector256.Create(lo128, hi128);

            // Expand 32 qh bits to 32 bytes, each 0 or 16 (bit << 4)
            for (int i = 0; i < 32; i++)
                expanded[i] = (byte)(((qh >> i) & 1) << 4);

            Vector256<byte> bit5Vec = Unsafe.ReadUnaligned<Vector256<byte>>(expanded);
            Vector256<byte> q5vals = Avx2.Or(nibbles, bit5Vec); // 5-bit unsigned [0..31]

            // Load Q8_0 signed values
            Vector256<sbyte> q8Vals = Unsafe.ReadUnaligned<Vector256<sbyte>>(q8Block + 2);

            // ubyte × sbyte → int16 pairs (vpmaddubsw)
            Vector256<short> prod = Avx2.MultiplyAddAdjacent(q5vals, q8Vals);
            // int16 → int32 (vpmaddwd)
            Vector256<int> prodSum = Avx2.MultiplyAddAdjacent(prod, ones);

            // Sum of Q8 values for the -16 offset: sum((q5-16)*q8) = sum(q5*q8) - 16*sum(q8)
            Vector256<short> q8Sums = Avx2.MultiplyAddAdjacent(Vector256.Create((byte)1), q8Vals);
            Vector256<int> q8Sum = Avx2.MultiplyAddAdjacent(q8Sums, ones);

            // Accumulate: d5*d8 * (prodSum - 16*q8Sum)
            float scale = d5 * d8;
            acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(scale),
                Avx.Subtract(
                    Avx.ConvertToVector256Single(prodSum),
                    Avx.Multiply(Vector256.Create(16f), Avx.ConvertToVector256Single(q8Sum)))));
        }

        return HorizontalSumAvx2Float(acc);
    }

    // ──────────────────── Q5_0 × Q8_0 4-row variant ────────────────────

    [SkipLocalsInit]
    internal static void VecDotQ5_0Q8_0Avx2_4Rows(
        byte* w0, byte* w1, byte* w2, byte* w3,
        byte* q8, int blockCount, float* results)
    {
        results[0] = Avx2.IsSupported
            ? VecDotQ5_0Q8_0Avx2(w0, q8, blockCount)
            : VecDotQ5_0Q8_0Scalar(w0, q8, blockCount);
        results[1] = Avx2.IsSupported
            ? VecDotQ5_0Q8_0Avx2(w1, q8, blockCount)
            : VecDotQ5_0Q8_0Scalar(w1, q8, blockCount);
        results[2] = Avx2.IsSupported
            ? VecDotQ5_0Q8_0Avx2(w2, q8, blockCount)
            : VecDotQ5_0Q8_0Scalar(w2, q8, blockCount);
        results[3] = Avx2.IsSupported
            ? VecDotQ5_0Q8_0Avx2(w3, q8, blockCount)
            : VecDotQ5_0Q8_0Scalar(w3, q8, blockCount);
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
                VecDotQ5_0Q8_0Avx2_4Rows(
                    weights + (long)row * rowBytes,
                    weights + (long)(row + 1) * rowBytes,
                    weights + (long)(row + 2) * rowBytes,
                    weights + (long)(row + 3) * rowBytes,
                    xQ8, blockCount, result + row);
            }
            for (; row < m; row++)
                result[row] = VecDotQ5_0Q8_0Avx2(weights + (long)row * rowBytes, xQ8, blockCount);
        }
        else
        {
            for (int row = 0; row < m; row++)
                result[row] = VecDotQ5_0Q8_0Scalar(weights + (long)row * rowBytes, xQ8, blockCount);
        }
    }

    // ──────────────────── GemvQ5_0 ────────────────────

    /// <summary>
    /// Q5_0 GEMV: weights[M,K] in Q5_0 × f32 input[K] → f32 output[M].
    /// Quantizes input to Q8_0, then uses fused Q5_0 × Q8_0 vec_dot.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemvQ5_0(byte* weights, float* x, float* result, int m, int k)
    {
        if (k % Q5_0GroupSize != 0)
            throw new ArgumentException(
                $"k must be a multiple of {Q5_0GroupSize}, got {k}", nameof(k));

        int blockCount = k / Q8_0GroupSize; // same group size as Q8_0
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
                ComputeRowsQ5_0(weights, xQ8, result, m, blockCount);
            }
            ArrayPool<byte>.Shared.Return(rented);
            return;
        }

        QuantizeF32ToQ8_0(x, xQ8, k);
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

        int blockCount = k / Q8_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;
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
                QuantizeF32ToQ8_0(b + t * k, inputQ8 + t * q8RowBytes, k);

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

        int blockCount = k / Q8_0GroupSize;
        int xQ8Bytes = blockCount * Q8_0BlockBytes;

        byte* xQ8 = (byte*)pool.GetWorkerScratch(0, xQ8Bytes);
        QuantizeF32ToQ8_0(x, xQ8, k);

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
                int blockCount = k / Q8_0GroupSize;
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

        int q8BlockCount = k / Q8_0GroupSize;
        int q8RowBytes = q8BlockCount * Q8_0BlockBytes;
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
                QuantizeF32ToQ8_0(b + t * k, inputQ8 + t * q8RowBytes, k);

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
