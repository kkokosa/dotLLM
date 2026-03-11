using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Fused decode dispatch: combines multiple GEMV projections (Q/K/V or Gate/Up) into a single
/// <see cref="ComputeThreadPool.Dispatch"/> call, reducing per-token dispatch overhead.
/// Projections are fused only within the same quant family (Q8_0/Q5_0 share Q8_0 input,
/// K-quants share Q8_K input). Cross-family projections dispatch separately.
/// </summary>
public static unsafe partial class MatMul
{
    // ──────────────────── Quant family classification ────────────────────

    /// <summary>
    /// Quant family grouping for pre-quantization reuse.
    /// Q8_0 uses Q8_0 input quantization, Q5_0 uses Q8_1 (precomputed block sums),
    /// K-quants use Q8_K. Cross-family projections cannot share pre-quantized input.
    /// </summary>
    private enum QuantFamily { None, Q8_0Family, Q8_1Family, KQuantFamily }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static QuantFamily GetQuantFamily(QuantizationType qt) => qt switch
    {
        QuantizationType.Q8_0 => QuantFamily.Q8_0Family,
        QuantizationType.Q5_0 => QuantFamily.Q8_1Family,
        QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K => QuantFamily.KQuantFamily,
        _ => QuantFamily.None,
    };

    // ──────────────────── Helpers ────────────────────

    /// <summary>
    /// Returns the <c>ComputeRows</c> function pointer for a given quant type.
    /// These functions have signature: <c>(byte* weights, byte* preQuantInput, float* result, int m, int blockCount) → void</c>.
    /// Returns null for F32/F16 (not supported in fused path).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static delegate*<byte*, byte*, float*, int, int, void> GetComputeRowsFn(QuantizationType qt) => qt switch
    {
        QuantizationType.Q8_0 => &ComputeRows,
        QuantizationType.Q5_0 => &ComputeRowsQ5_0,
        QuantizationType.Q4_K => &ComputeRowsQ4_K,
        QuantizationType.Q5_K => &ComputeRowsQ5_K,
        QuantizationType.Q6_K => &ComputeRowsQ6_K,
        _ => null,
    };

    /// <summary>Returns per-row weight block bytes for the given quant type.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetWeightBlockBytes(QuantizationType qt) => qt switch
    {
        QuantizationType.Q8_0 => Q8_0BlockBytes,
        QuantizationType.Q5_0 => Q5_0BlockBytes,
        QuantizationType.Q4_K => Q4_K_BlockBytes,
        QuantizationType.Q5_K => Q5_K_BlockBytes,
        QuantizationType.Q6_K => Q6_K_BlockBytes,
        _ => 0,
    };

    /// <summary>
    /// Returns the block count for a given k dimension and quant type.
    /// Q8-family uses k/32, K-quant uses k/256.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetBlockCount(int k, QuantizationType qt) => GetQuantFamily(qt) switch
    {
        QuantFamily.Q8_0Family or QuantFamily.Q8_1Family => k / Q8_0GroupSize,
        QuantFamily.KQuantFamily => k / KQuantGroupSize,
        _ => 0,
    };

    /// <summary>Returns per-row weight bytes: blockCount * blockBytes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GetWeightRowBytes(int k, QuantizationType qt)
    {
        int blockBytes = GetWeightBlockBytes(qt);
        return GetQuantFamily(qt) switch
        {
            QuantFamily.Q8_0Family or QuantFamily.Q8_1Family => (k / Q8_0GroupSize) * blockBytes,
            QuantFamily.KQuantFamily => (k / KQuantGroupSize) * blockBytes,
            _ => 0,
        };
    }

    // ──────────────────── Context structs ────────────────────

    /// <summary>Single projection descriptor within a fused group.</summary>
    private struct FusedProjection
    {
        public byte* Weights;
        public float* Result;
        public int M;
        public int WeightRowBytes;
        public delegate*<byte*, byte*, float*, int, int, void> ComputeRows;
    }

    /// <summary>Fused context for up to 3 projections (Q/K/V) in one dispatch.</summary>
    private struct FusedDecode3Ctx
    {
        public FusedProjection Proj0;
        public FusedProjection Proj1;
        public FusedProjection Proj2;
        public byte* PreQuantInput;
        public int BlockCount;
        public int TotalRows;
    }

    /// <summary>Fused context for 2 projections (Gate/Up) in one dispatch.</summary>
    private struct FusedDecode2Ctx
    {
        public FusedProjection Proj0;
        public FusedProjection Proj1;
        public byte* PreQuantInput;
        public int BlockCount;
        public int TotalRows;
    }

    // ──────────────────── Workers ────────────────────

    private static void FusedDecode3Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<FusedDecode3Ctx>((void*)ctxPtr);
        PartitionRows(ctx.TotalRows, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;

        int end = start + count;
        int offset = 0;

        // Proj0
        if (ctx.Proj0.M > 0)
        {
            int projStart = Math.Max(0, start - offset);
            int projEnd = Math.Min(ctx.Proj0.M, end - offset);
            if (projEnd > projStart)
                ctx.Proj0.ComputeRows(
                    ctx.Proj0.Weights + (long)projStart * ctx.Proj0.WeightRowBytes,
                    ctx.PreQuantInput,
                    ctx.Proj0.Result + projStart,
                    projEnd - projStart, ctx.BlockCount);
            offset += ctx.Proj0.M;
        }

        // Proj1
        if (ctx.Proj1.M > 0)
        {
            int projStart = Math.Max(0, start - offset);
            int projEnd = Math.Min(ctx.Proj1.M, end - offset);
            if (projEnd > projStart)
                ctx.Proj1.ComputeRows(
                    ctx.Proj1.Weights + (long)projStart * ctx.Proj1.WeightRowBytes,
                    ctx.PreQuantInput,
                    ctx.Proj1.Result + projStart,
                    projEnd - projStart, ctx.BlockCount);
            offset += ctx.Proj1.M;
        }

        // Proj2
        if (ctx.Proj2.M > 0)
        {
            int projStart = Math.Max(0, start - offset);
            int projEnd = Math.Min(ctx.Proj2.M, end - offset);
            if (projEnd > projStart)
                ctx.Proj2.ComputeRows(
                    ctx.Proj2.Weights + (long)projStart * ctx.Proj2.WeightRowBytes,
                    ctx.PreQuantInput,
                    ctx.Proj2.Result + projStart,
                    projEnd - projStart, ctx.BlockCount);
        }
    }

    private static void FusedDecode2Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<FusedDecode2Ctx>((void*)ctxPtr);
        PartitionRows(ctx.TotalRows, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;

        int end = start + count;
        int offset = 0;

        // Proj0
        if (ctx.Proj0.M > 0)
        {
            int projStart = Math.Max(0, start - offset);
            int projEnd = Math.Min(ctx.Proj0.M, end - offset);
            if (projEnd > projStart)
                ctx.Proj0.ComputeRows(
                    ctx.Proj0.Weights + (long)projStart * ctx.Proj0.WeightRowBytes,
                    ctx.PreQuantInput,
                    ctx.Proj0.Result + projStart,
                    projEnd - projStart, ctx.BlockCount);
            offset += ctx.Proj0.M;
        }

        // Proj1
        if (ctx.Proj1.M > 0)
        {
            int projStart = Math.Max(0, start - offset);
            int projEnd = Math.Min(ctx.Proj1.M, end - offset);
            if (projEnd > projStart)
                ctx.Proj1.ComputeRows(
                    ctx.Proj1.Weights + (long)projStart * ctx.Proj1.WeightRowBytes,
                    ctx.PreQuantInput,
                    ctx.Proj1.Result + projStart,
                    projEnd - projStart, ctx.BlockCount);
        }
    }

    // ──────────────────── Build FusedProjection helper ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static FusedProjection MakeProjection(byte* weights, QuantizationType qt, float* result, int m, int k)
    {
        return new FusedProjection
        {
            Weights = weights,
            Result = result,
            M = m,
            WeightRowBytes = GetWeightRowBytes(k, qt),
            ComputeRows = GetComputeRowsFn(qt),
        };
    }

    // ──────────────────── Public API ────────────────────

    /// <summary>
    /// Fused decode GEMV for up to 3 projections (Q/K/V). All projections share the same
    /// pre-quantized input. When all projections share the same quant family, dispatches in
    /// a single <see cref="ComputeThreadPool.Dispatch"/> call. When families differ (rare),
    /// same-family projections are fused together and cross-family projections dispatch
    /// individually with self-quantizing GEMV (correct but no dispatch savings).
    /// </summary>
    /// <param name="w0">Weight pointer for projection 0.</param>
    /// <param name="qt0">Quant type for projection 0.</param>
    /// <param name="r0">Result pointer for projection 0.</param>
    /// <param name="m0">Output rows for projection 0.</param>
    /// <param name="w1">Weight pointer for projection 1.</param>
    /// <param name="qt1">Quant type for projection 1.</param>
    /// <param name="r1">Result pointer for projection 1.</param>
    /// <param name="m1">Output rows for projection 1.</param>
    /// <param name="w2">Weight pointer for projection 2.</param>
    /// <param name="qt2">Quant type for projection 2.</param>
    /// <param name="r2">Result pointer for projection 2.</param>
    /// <param name="m2">Output rows for projection 2.</param>
    /// <param name="input">f32 input vector [k].</param>
    /// <param name="preQuantInput">Pre-quantized input for qt0's quant family (Q8_0 or Q8_K). Null if not available.</param>
    /// <param name="k">Input dimension (shared by all projections).</param>
    /// <param name="pool">Thread pool for parallel dispatch.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void FusedDecodeGemv3(
        byte* w0, QuantizationType qt0, float* r0, int m0,
        byte* w1, QuantizationType qt1, float* r1, int m1,
        byte* w2, QuantizationType qt2, float* r2, int m2,
        float* input, byte* preQuantInput, int k,
        ComputeThreadPool pool)
    {
        // No pre-quantized input → fall back to individual dispatches (self-quantizing GEMV)
        if (preQuantInput == null)
        {
            DispatchSingle(w0, qt0, r0, m0, input, null, k, pool);
            DispatchSingle(w1, qt1, r1, m1, input, null, k, pool);
            DispatchSingle(w2, qt2, r2, m2, input, null, k, pool);
            return;
        }

        var f0 = GetQuantFamily(qt0);
        var f1 = GetQuantFamily(qt1);
        var f2 = GetQuantFamily(qt2);

        // All same family → single fused dispatch (common case)
        if (f0 != QuantFamily.None && f0 == f1 && f1 == f2)
        {
            int blockCount = GetBlockCount(k, qt0);
            var ctx = new FusedDecode3Ctx
            {
                Proj0 = MakeProjection(w0, qt0, r0, m0, k),
                Proj1 = MakeProjection(w1, qt1, r1, m1, k),
                Proj2 = MakeProjection(w2, qt2, r2, m2, k),
                PreQuantInput = preQuantInput,
                BlockCount = blockCount,
                TotalRows = m0 + m1 + m2,
            };
            pool.Dispatch((nint)(&ctx), &FusedDecode3Worker);
            return;
        }

        // Cross-family: preQuantInput is valid only for f0's family.
        // Same-family projections can fuse with preQuantInput if they match f0.
        // Different-family projections dispatch individually with null preQuant
        // (self-quantizing GEMV handles its own input quantization).
        bool f1MatchesF0 = f1 == f0;
        bool f2MatchesF0 = f2 == f0;

        if (f0 != QuantFamily.None && f1MatchesF0)
        {
            // Fuse proj0 + proj1 (same family as preQuantInput), dispatch proj2 individually
            int blockCount = GetBlockCount(k, qt0);
            var ctx2 = new FusedDecode2Ctx
            {
                Proj0 = MakeProjection(w0, qt0, r0, m0, k),
                Proj1 = MakeProjection(w1, qt1, r1, m1, k),
                PreQuantInput = preQuantInput,
                BlockCount = blockCount,
                TotalRows = m0 + m1,
            };
            pool.Dispatch((nint)(&ctx2), &FusedDecode2Worker);
            DispatchSingle(w2, qt2, r2, m2, input, null, k, pool);
        }
        else if (f0 != QuantFamily.None && f2MatchesF0)
        {
            // Fuse proj0 + proj2 (same family as preQuantInput), dispatch proj1 individually
            int blockCount = GetBlockCount(k, qt0);
            var ctx2 = new FusedDecode2Ctx
            {
                Proj0 = MakeProjection(w0, qt0, r0, m0, k),
                Proj1 = MakeProjection(w2, qt2, r2, m2, k),
                PreQuantInput = preQuantInput,
                BlockCount = blockCount,
                TotalRows = m0 + m2,
            };
            pool.Dispatch((nint)(&ctx2), &FusedDecode2Worker);
            DispatchSingle(w1, qt1, r1, m1, input, null, k, pool);
        }
        else
        {
            // proj0 dispatches with its preQuantInput, proj1/proj2 individually (no preQuant)
            DispatchSingle(w0, qt0, r0, m0, input, preQuantInput, k, pool);
            DispatchSingle(w1, qt1, r1, m1, input, null, k, pool);
            DispatchSingle(w2, qt2, r2, m2, input, null, k, pool);
        }
    }

    /// <summary>
    /// Fused decode GEMV for 2 projections (Gate/Up). Same-family projections are dispatched
    /// in a single call; cross-family dispatches separately.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void FusedDecodeGemv2(
        byte* w0, QuantizationType qt0, float* r0, int m0,
        byte* w1, QuantizationType qt1, float* r1, int m1,
        float* input, byte* preQuantInput, int k,
        ComputeThreadPool pool)
    {
        // No pre-quantized input → fall back to individual dispatches (self-quantizing GEMV)
        if (preQuantInput == null)
        {
            DispatchSingle(w0, qt0, r0, m0, input, null, k, pool);
            DispatchSingle(w1, qt1, r1, m1, input, null, k, pool);
            return;
        }

        var f0 = GetQuantFamily(qt0);
        var f1 = GetQuantFamily(qt1);

        if (f0 != QuantFamily.None && f0 == f1)
        {
            // Same family → single fused dispatch
            int blockCount = GetBlockCount(k, qt0);
            var ctx = new FusedDecode2Ctx
            {
                Proj0 = MakeProjection(w0, qt0, r0, m0, k),
                Proj1 = MakeProjection(w1, qt1, r1, m1, k),
                PreQuantInput = preQuantInput,
                BlockCount = blockCount,
                TotalRows = m0 + m1,
            };
            pool.Dispatch((nint)(&ctx), &FusedDecode2Worker);
        }
        else
        {
            // Different families → dispatch individually
            // proj0 can use preQuantInput (it matches), proj1 must self-quantize
            DispatchSingle(w0, qt0, r0, m0, input, preQuantInput, k, pool);
            DispatchSingle(w1, qt1, r1, m1, input, null, k, pool);
        }
    }

    /// <summary>
    /// Dispatches a single projection using the appropriate existing parallel GEMV path.
    /// For F32/F16, calls GemvF32/GemvF16. For quantized types with preQuantInput, dispatches
    /// via ComputeRows directly. When preQuantInput is null, falls back to the self-quantizing
    /// GEMV which handles its own input quantization.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DispatchSingle(byte* weights, QuantizationType qt, float* result,
                                       int m, float* input, byte* preQuantInput, int k,
                                       ComputeThreadPool pool)
    {
        if (qt == QuantizationType.F32)
        {
            GemvF32((float*)weights, input, result, m, k, pool);
            return;
        }
        if (qt == QuantizationType.F16)
        {
            GemvF16((nint)weights, input, result, m, k, pool);
            return;
        }

        if (preQuantInput != null)
        {
            // Pre-quantized input available and format-compatible → dispatch via ComputeRows
            var fn = GetComputeRowsFn(qt);
            if (fn == null) return;

            int blockCount = GetBlockCount(k, qt);
            var ctx = new FusedDecode2Ctx
            {
                Proj0 = MakeProjection(weights, qt, result, m, k),
                Proj1 = default, // M=0, unused
                PreQuantInput = preQuantInput,
                BlockCount = blockCount,
                TotalRows = m,
            };
            pool.Dispatch((nint)(&ctx), &FusedDecode2Worker);
        }
        else
        {
            // No compatible preQuant → fall back to individual GEMV (handles its own quantization)
            switch (qt)
            {
                case QuantizationType.Q8_0: GemvQ8_0(weights, input, result, m, k, pool); break;
                case QuantizationType.Q5_0: GemvQ5_0(weights, input, result, m, k, pool); break;
                case QuantizationType.Q4_K: GemvQ4_K(weights, input, result, m, k, pool); break;
                case QuantizationType.Q5_K: GemvQ5_K(weights, input, result, m, k, pool); break;
                case QuantizationType.Q6_K: GemvQ6_K(weights, input, result, m, k, pool); break;
                default:
                    throw new NotSupportedException(
                        $"Fused decode does not support {qt}. Use standard Gemm path.");
            }
        }
    }
}
