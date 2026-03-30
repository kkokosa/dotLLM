using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Scaled dot-product attention kernel with GQA head broadcast and causal masking.
/// Handles MHA, MQA, and GQA via <c>groupSize = numHeads / numKvHeads</c>.
/// <para>
/// For each query head <c>h</c>, the corresponding KV head is <c>h / groupSize</c>.
/// Attention is computed as: <c>softmax((Q @ K^T) / sqrt(headDim) + causalMask) @ V</c>.
/// </para>
/// </summary>
public static class Attention
{
    /// <summary>Stackalloc threshold in bytes. Above this, use ArrayPool.</summary>
    private const int StackAllocThreshold = 8192;

    /// <summary>Maximum tile size for tiled attention. Used for constant-size stackalloc.</summary>
    private const int MaxTileSize = 256;

    /// <summary>
    /// Computes scaled dot-product attention with causal masking and GQA head broadcast.
    /// Convenience overload that computes <c>scale = 1/sqrt(headDim)</c>.
    /// </summary>
    /// <param name="q">Query tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">Key tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">Value tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="output">Output tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Number of query positions (tokens being generated).</param>
    /// <param name="seqKv">Number of key/value positions (total context length).</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="positionOffset">Position offset for causal mask. For prefill: 0. For decode: number of cached tokens.</param>
    /// <param name="slidingWindowSize">Optional sliding window size. When non-null, limits attention to the most recent positions.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                Span<float> output,
                                int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                int positionOffset, int? slidingWindowSize = null)
        => Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                   positionOffset, 1.0f / MathF.Sqrt(headDim), slidingWindowSize);

    /// <summary>
    /// Computes scaled dot-product attention with causal masking, GQA head broadcast, and caller-provided scale.
    /// </summary>
    /// <param name="q">Query tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">Key tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">Value tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="output">Output tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Number of query positions (tokens being generated).</param>
    /// <param name="seqKv">Number of key/value positions (total context length).</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="positionOffset">Position offset for causal mask. For prefill: 0. For decode: number of cached tokens.</param>
    /// <param name="scale">Attention scale factor applied to dot-product scores.</param>
    /// <param name="slidingWindowSize">Optional sliding window size. When non-null, limits attention to the most recent positions.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                Span<float> output,
                                int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                int positionOffset, float scale, int? slidingWindowSize = null)
    {
        if (headDim <= 0)
            throw new ArgumentException($"headDim must be positive, got {headDim}", nameof(headDim));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));

        int groupSize = numHeads / numKvHeads;
        int qStride = numHeads * headDim;
        int kvStride = numKvHeads * headDim;
        int scoreSize = seqQ * seqKv;

        // Small score matrix: use existing naive path (SIMD softmax is faster for tiny sequences)
        if (scoreSize * sizeof(float) <= StackAllocThreshold)
        {
            Span<float> scores = stackalloc float[scoreSize];
            ExecuteCore(q, k, v, output, scores, seqQ, seqKv, numHeads, headDim,
                        groupSize, scale, qStride, kvStride, positionOffset, slidingWindowSize);
        }
        else
        {
            // Tiled path: only tileSize floats on stack instead of seqQ*seqKv
            int tileSize = ComputeTileSize(headDim);
            Span<float> tileScores = stackalloc float[MaxTileSize];
            ExecuteTiledCore(q, k, v, output, tileScores, seqQ, seqKv, numHeads, headDim,
                             groupSize, scale, qStride, kvStride, positionOffset, tileSize, slidingWindowSize ?? 0);
        }
    }

    private static void ExecuteCore(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                     Span<float> output, Span<float> scores,
                                     int seqQ, int seqKv, int numHeads, int headDim,
                                     int groupSize, float scale, int qStride, int kvStride,
                                     int positionOffset, int? slidingWindowSize = null)
    {
        for (int h = 0; h < numHeads; h++)
        {
            int kvH = h / groupSize;

            // 1. Scaled dot-product scores: Q_h @ K_kvH^T, scaled
            ScaledDotProductScores(q, k, scores, seqQ, seqKv, headDim, scale,
                                   h, kvH, qStride, kvStride);

            // 2. Apply causal mask
            ApplyCausalMask(scores, seqQ, seqKv, positionOffset, slidingWindowSize);

            // 3. Fast softmax per row (approximate exp — sufficient for attention)
            for (int i = 0; i < seqQ; i++)
            {
                var row = scores.Slice(i * seqKv, seqKv);
                Softmax.ExecuteFast(row, row);
            }

            // 4. Weighted sum: weights @ V_kvH → output_h
            WeightedValues(scores, v, output, seqQ, seqKv, headDim,
                           h, kvH, qStride, kvStride);
        }
    }

    /// <summary>
    /// Computes KV tile size targeting L2 cache residency.
    /// Each tile loads Tc K vectors + Tc V vectors of headDim floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ComputeTileSize(int headDim)
    {
        const int L2Budget = 256 * 1024; // 256 KB conservative L2 estimate
        int bytesPerKvToken = headDim * sizeof(float) * 2; // K + V
        int tc = L2Budget / bytesPerKvToken;
        return Math.Clamp(tc, 64, MaxTileSize);
    }

    /// <summary>
    /// Tiled attention with online softmax. Processes KV in tiles of <paramref name="tileSize"/> tokens,
    /// maintaining running max and sum_exp for numerically stable softmax without materializing the
    /// full seqQ×seqKv score matrix. Memory: O(tileSize) per head instead of O(seqQ×seqKv).
    /// </summary>
    private static void ExecuteTiledCore(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                          Span<float> output, Span<float> tileScores,
                                          int seqQ, int seqKv, int numHeads, int headDim,
                                          int groupSize, float scale, int qStride, int kvStride,
                                          int positionOffset, int tileSize, int slidingWindowSize = 0)
    {
        for (int h = 0; h < numHeads; h++)
        {
            ExecuteTiledCore(q, k, v, output, tileScores, seqQ, seqKv, 1, headDim,
                             1, scale, qStride, kvStride, positionOffset, tileSize, slidingWindowSize,
                             h, h / groupSize);
        }
    }

    // ──────────────────── Parallel overloads ────────────────────

    /// <summary>
    /// Pointer-based attention with optional head-parallel execution via <paramref name="pool"/>.
    /// When pool is null or numHeads &lt; 2, falls back to the single-threaded span-based path.
    /// </summary>
    [SkipLocalsInit]
    public static unsafe void Execute(float* q, float* k, float* v, float* output,
                                      int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                      int positionOffset, ComputeThreadPool? pool,
                                      int? slidingWindowSize = null)
        => Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                   positionOffset, 1.0f / MathF.Sqrt(headDim), pool, slidingWindowSize);

    /// <summary>
    /// Pointer-based attention with caller-provided scale and optional head-parallel execution.
    /// </summary>
    [SkipLocalsInit]
    public static unsafe void Execute(float* q, float* k, float* v, float* output,
                                      int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                      int positionOffset, float scale, ComputeThreadPool? pool,
                                      int? slidingWindowSize = null)
    {
        if (headDim <= 0)
            throw new ArgumentException($"headDim must be positive, got {headDim}", nameof(headDim));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));

        if (pool is null || numHeads < 2)
        {
            // Fall back to span-based single-threaded path
            int qLen = seqQ * numHeads * headDim;
            int kvLen = seqKv * numKvHeads * headDim;
            Execute(
                new ReadOnlySpan<float>(q, qLen),
                new ReadOnlySpan<float>(k, kvLen),
                new ReadOnlySpan<float>(v, kvLen),
                new Span<float>(output, qLen),
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, scale, slidingWindowSize);
            return;
        }

        int scoreSize = seqQ * seqKv;

        // Small score matrix: naive parallel path with per-worker scratch
        if (scoreSize * sizeof(float) <= StackAllocThreshold)
        {
            int scratchBytes = scoreSize * sizeof(float);
            int threadCount = pool.ThreadCount;
            nint* scratchPtrs = stackalloc nint[threadCount];
            for (int i = 0; i < threadCount; i++)
                scratchPtrs[i] = pool.GetWorkerScratch(i, scratchBytes);

            var ctx = new AttentionCtx
            {
                Q = q, K = k, V = v, Output = output,
                SeqQ = seqQ, SeqKv = seqKv, NumHeads = numHeads, NumKvHeads = numKvHeads,
                HeadDim = headDim, Scale = scale, PositionOffset = positionOffset,
                GroupSize = numHeads / numKvHeads,
                QStride = numHeads * headDim,
                KvStride = numKvHeads * headDim,
                ScoreSize = scoreSize,
                ScratchPtrs = scratchPtrs,
                SlidingWindowSize = slidingWindowSize ?? 0
            };
            pool.Dispatch((nint)(&ctx), &AttentionWorker);
        }
        else
        {
            // Large score matrix: tiled parallel path — no scratch pre-allocation needed
            int tileSize = ComputeTileSize(headDim);

            var ctx = new TiledAttentionCtx
            {
                Q = q, K = k, V = v, Output = output,
                SeqQ = seqQ, SeqKv = seqKv, NumHeads = numHeads, NumKvHeads = numKvHeads,
                HeadDim = headDim, Scale = scale, PositionOffset = positionOffset,
                GroupSize = numHeads / numKvHeads,
                QStride = numHeads * headDim,
                KvStride = numKvHeads * headDim,
                TileSize = tileSize,
                SlidingWindowSize = slidingWindowSize ?? 0
            };
            pool.Dispatch((nint)(&ctx), &TiledAttentionWorker);
        }
    }

    private unsafe struct AttentionCtx
    {
        public float* Q;
        public float* K;
        public float* V;
        public float* Output;
        public int SeqQ;
        public int SeqKv;
        public int NumHeads;
        public int NumKvHeads;
        public int HeadDim;
        public float Scale;
        public int PositionOffset;
        public int GroupSize;
        public int QStride;
        public int KvStride;
        public int ScoreSize;
        public nint* ScratchPtrs;
        /// <summary>Sliding window size. 0 means no sliding window (full context).</summary>
        public int SlidingWindowSize;
    }

    private static unsafe void AttentionWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<AttentionCtx>((void*)ctxPtr);

        // Partition heads across threads
        int headsPerThread = (ctx.NumHeads + threadCount - 1) / threadCount;
        int startHead = threadIdx * headsPerThread;
        int endHead = Math.Min(startHead + headsPerThread, ctx.NumHeads);
        if (startHead >= ctx.NumHeads) return;

        float* scores = (float*)ctx.ScratchPtrs[threadIdx];
        var scoresSpan = new Span<float>(scores, ctx.ScoreSize);

        var qSpan = new ReadOnlySpan<float>(ctx.Q, ctx.SeqQ * ctx.QStride);
        var kSpan = new ReadOnlySpan<float>(ctx.K, ctx.SeqKv * ctx.KvStride);
        var vSpan = new ReadOnlySpan<float>(ctx.V, ctx.SeqKv * ctx.KvStride);
        var outSpan = new Span<float>(ctx.Output, ctx.SeqQ * ctx.QStride);

        int? slidingWindow = ctx.SlidingWindowSize > 0 ? ctx.SlidingWindowSize : null;

        for (int h = startHead; h < endHead; h++)
        {
            int kvH = h / ctx.GroupSize;

            ScaledDotProductScores(qSpan, kSpan, scoresSpan, ctx.SeqQ, ctx.SeqKv, ctx.HeadDim, ctx.Scale,
                                   h, kvH, ctx.QStride, ctx.KvStride);

            ApplyCausalMask(scoresSpan, ctx.SeqQ, ctx.SeqKv, ctx.PositionOffset, slidingWindow);

            for (int i = 0; i < ctx.SeqQ; i++)
            {
                var row = scoresSpan.Slice(i * ctx.SeqKv, ctx.SeqKv);
                Softmax.ExecuteFast(row, row);
            }

            WeightedValues(scoresSpan, vSpan, outSpan, ctx.SeqQ, ctx.SeqKv, ctx.HeadDim,
                           h, kvH, ctx.QStride, ctx.KvStride);
        }
    }

    private unsafe struct TiledAttentionCtx
    {
        public float* Q;
        public float* K;
        public float* V;
        public float* Output;
        public int SeqQ;
        public int SeqKv;
        public int NumHeads;
        public int NumKvHeads;
        public int HeadDim;
        public float Scale;
        public int PositionOffset;
        public int GroupSize;
        public int QStride;
        public int KvStride;
        public int TileSize;
        /// <summary>Sliding window size. 0 means no sliding window (full context).</summary>
        public int SlidingWindowSize;
    }

    [SkipLocalsInit]
    private static unsafe void TiledAttentionWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<TiledAttentionCtx>((void*)ctxPtr);

        // Partition heads across threads
        int headsPerThread = (ctx.NumHeads + threadCount - 1) / threadCount;
        int startHead = threadIdx * headsPerThread;
        int endHead = Math.Min(startHead + headsPerThread, ctx.NumHeads);
        if (startHead >= ctx.NumHeads) return;

        var qSpan = new ReadOnlySpan<float>(ctx.Q, ctx.SeqQ * ctx.QStride);
        var kSpan = new ReadOnlySpan<float>(ctx.K, ctx.SeqKv * ctx.KvStride);
        var vSpan = new ReadOnlySpan<float>(ctx.V, ctx.SeqKv * ctx.KvStride);
        var outSpan = new Span<float>(ctx.Output, ctx.SeqQ * ctx.QStride);

        // Each worker stackallocs its own tile scores — max 256 * 4 = 1024 bytes
        Span<float> tileScores = stackalloc float[MaxTileSize];

        for (int h = startHead; h < endHead; h++)
        {
            ExecuteTiledCore(qSpan, kSpan, vSpan, outSpan, tileScores,
                             ctx.SeqQ, ctx.SeqKv, 1, ctx.HeadDim,
                             1, ctx.Scale, ctx.QStride, ctx.KvStride,
                             ctx.PositionOffset, ctx.TileSize, ctx.SlidingWindowSize,
                             h, h / ctx.GroupSize);
        }
    }

    /// <summary>
    /// Tiled attention overload for the parallel worker: processes a single head identified by
    /// <paramref name="headIdx"/> and <paramref name="kvHeadIdx"/>.
    /// </summary>
    private static void ExecuteTiledCore(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                          Span<float> output, Span<float> tileScores,
                                          int seqQ, int seqKv, int numHeads, int headDim,
                                          int groupSize, float scale, int qStride, int kvStride,
                                          int positionOffset, int tileSize, int slidingWindowSize,
                                          int headIdx, int kvHeadIdx)
    {
        int window = slidingWindowSize;

        for (int i = 0; i < seqQ; i++)
        {
            var qRow = q.Slice(i * qStride + headIdx * headDim, headDim);
            var outRow = output.Slice(i * qStride + headIdx * headDim, headDim);
            outRow.Clear();

            // Causal + sliding window bounds
            int visibleEnd = Math.Min(seqKv, positionOffset + i + 1);
            int visibleStart = (window > 0)
                ? Math.Max(0, positionOffset + i - window + 1)
                : 0;

            if (visibleStart >= visibleEnd)
                continue;

            float maxSoFar = float.NegativeInfinity;
            float sumExp = 0f;

            for (int tileBase = visibleStart; tileBase < visibleEnd; tileBase += tileSize)
            {
                int tileLen = Math.Min(tileSize, visibleEnd - tileBase);
                var scores = tileScores.Slice(0, tileLen);

                for (int j = 0; j < tileLen; j++)
                {
                    var kRow = k.Slice((tileBase + j) * kvStride + kvHeadIdx * headDim, headDim);
                    scores[j] = TensorPrimitives.Dot(qRow, kRow) * scale;
                }

                float tileMax = TensorPrimitives.Max(scores);
                float newMax = MathF.Max(maxSoFar, tileMax);
                float correction = FastMath.FastExp(maxSoFar - newMax);

                if (correction < 1f)
                {
                    sumExp *= correction;
                    TensorPrimitives.Multiply(outRow, correction, outRow);
                }

                sumExp += FastMath.ExpSumAndStore(scores, scores, -newMax);

                for (int j = 0; j < tileLen; j++)
                {
                    float w = scores[j];
                    if (w == 0f) continue;
                    var vRow = v.Slice((tileBase + j) * kvStride + kvHeadIdx * headDim, headDim);
                    TensorPrimitives.MultiplyAdd(vRow, w, outRow, outRow);
                }

                maxSoFar = newMax;
            }

            if (sumExp > 0f)
                TensorPrimitives.Multiply(outRow, 1f / sumExp, outRow);
        }
    }

    /// <summary>
    /// Scalar reference implementation for correctness verification.
    /// Convenience overload that computes <c>scale = 1/sqrt(headDim)</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                        Span<float> output,
                                        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                        int positionOffset, int? slidingWindowSize = null)
        => ExecuteScalar(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                         positionOffset, 1.0f / MathF.Sqrt(headDim), slidingWindowSize);

    /// <summary>
    /// Scalar reference implementation with caller-provided scale.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                        Span<float> output,
                                        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                        int positionOffset, float scale, int? slidingWindowSize = null)
    {
        if (headDim <= 0)
            throw new ArgumentException($"headDim must be positive, got {headDim}", nameof(headDim));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));

        int groupSize = numHeads / numKvHeads;
        int qStride = numHeads * headDim;
        int kvStride = numKvHeads * headDim;

        float[] scores = new float[seqQ * seqKv];

        for (int h = 0; h < numHeads; h++)
        {
            int kvH = h / groupSize;

            // Scores: Q_h @ K_kvH^T
            for (int i = 0; i < seqQ; i++)
            {
                for (int j = 0; j < seqKv; j++)
                {
                    float dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += q[i * qStride + h * headDim + d] * k[j * kvStride + kvH * headDim + d];
                    scores[i * seqKv + j] = dot * scale;
                }
            }

            // Causal mask
            for (int i = 0; i < seqQ; i++)
            {
                for (int j = 0; j < seqKv; j++)
                {
                    if (j > positionOffset + i)
                        scores[i * seqKv + j] = float.NegativeInfinity;
                }
            }

            // Sliding window mask
            if (slidingWindowSize.HasValue)
            {
                int window = slidingWindowSize.Value;
                for (int i = 0; i < seqQ; i++)
                {
                    int earliestVisible = positionOffset + i - window + 1;
                    for (int j = 0; j < seqKv && j < earliestVisible; j++)
                    {
                        scores[i * seqKv + j] = float.NegativeInfinity;
                    }
                }
            }

            // Softmax per row
            for (int i = 0; i < seqQ; i++)
                Softmax.ExecuteScalar(scores.AsSpan(i * seqKv, seqKv), scores.AsSpan(i * seqKv, seqKv));

            // Weighted values
            for (int i = 0; i < seqQ; i++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0;
                    for (int j = 0; j < seqKv; j++)
                        sum += scores[i * seqKv + j] * v[j * kvStride + kvH * headDim + d];
                    output[i * qStride + h * headDim + d] = sum;
                }
            }
        }
    }

    /// <summary>
    /// Computes scaled dot-product scores: <c>scores[i,j] = (Q_h[i,:] · K_kvH[j,:]) * scale</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ScaledDotProductScores(ReadOnlySpan<float> q, ReadOnlySpan<float> k,
                                                 Span<float> scores,
                                                 int seqQ, int seqKv, int headDim, float scale,
                                                 int headIdx, int kvHeadIdx,
                                                 int qStride, int kvStride)
    {
        for (int i = 0; i < seqQ; i++)
        {
            var qRow = q.Slice(i * qStride + headIdx * headDim, headDim);
            for (int j = 0; j < seqKv; j++)
            {
                var kRow = k.Slice(j * kvStride + kvHeadIdx * headDim, headDim);
                scores[i * seqKv + j] = TensorPrimitives.Dot(qRow, kRow) * scale;
            }
        }
    }

    /// <summary>
    /// Applies causal (autoregressive) mask. Sets <c>scores[i,j] = -inf</c> where
    /// <c>j &gt; positionOffset + i</c> (query at position <c>positionOffset + i</c>
    /// cannot attend to keys at later positions).
    /// Optionally applies a sliding window mask that limits attention to the most recent
    /// <paramref name="slidingWindowSize"/> positions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ApplyCausalMask(Span<float> scores, int seqQ, int seqKv, int positionOffset,
                                         int? slidingWindowSize = null)
    {
        for (int i = 0; i < seqQ; i++)
        {
            for (int j = positionOffset + i + 1; j < seqKv; j++)
            {
                scores[i * seqKv + j] = float.NegativeInfinity;
            }
        }

        if (slidingWindowSize.HasValue)
        {
            int window = slidingWindowSize.Value;
            for (int i = 0; i < seqQ; i++)
            {
                int earliestVisible = positionOffset + i - window + 1;
                for (int j = 0; j < seqKv && j < earliestVisible; j++)
                {
                    scores[i * seqKv + j] = float.NegativeInfinity;
                }
            }
        }
    }

    /// <summary>
    /// Computes weighted sum: <c>output_h[i,:] = sum_j(weights[i,j] * V_kvH[j,:])</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void WeightedValues(ReadOnlySpan<float> weights, ReadOnlySpan<float> v,
                                         Span<float> output,
                                         int seqQ, int seqKv, int headDim,
                                         int headIdx, int kvHeadIdx,
                                         int qStride, int kvStride)
    {
        for (int i = 0; i < seqQ; i++)
        {
            var outSlice = output.Slice(i * qStride + headIdx * headDim, headDim);
            outSlice.Clear();

            for (int j = 0; j < seqKv; j++)
            {
                float w = weights[i * seqKv + j];
                // Exact 0f comparison is safe: softmax(exp(-∞ - max)) == 0f exactly in IEEE 754.
                // ApplyCausalMask writes float.NegativeInfinity, which exp() maps to exactly zero.
                if (w == 0f) continue;

                var vRow = v.Slice(j * kvStride + kvHeadIdx * headDim, headDim);
                TensorPrimitives.MultiplyAdd(vRow, w, outSlice, outSlice);
            }
        }
    }

    // ──────────────────── Quantized KV-cache attention ────────────────────

    /// <summary>
    /// Attention with quantized KV-cache. Dequantizes tiles on-the-fly during attention
    /// computation, then processes the full-precision window region directly.
    /// </summary>
    [SkipLocalsInit]
    public static unsafe void Execute(float* q, IQuantizedKvCache kvCache, int layerIndex,
                                       float* output,
                                       int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                       int positionOffset, ComputeThreadPool? pool,
                                       int? slidingWindowSize = null)
    {
        if (headDim <= 0)
            throw new ArgumentException($"headDim must be positive, got {headDim}", nameof(headDim));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));

        float scale = 1.0f / MathF.Sqrt(headDim);
        int kvStride = numKvHeads * headDim;
        int qStride = numHeads * headDim;
        int tileSize = ComputeTileSize(headDim);

        int quantLen = kvCache.QuantizedLength;
        int windowLen = kvCache.WindowLength;
        byte* kQuant = (byte*)kvCache.GetQuantizedKeysPtr(layerIndex);
        byte* vQuant = (byte*)kvCache.GetQuantizedValuesPtr(layerIndex);
        float* kWindow = (float*)kvCache.GetWindowKeysPtr(layerIndex);
        float* vWindow = (float*)kvCache.GetWindowValuesPtr(layerIndex);
        int kQuantRowBytes = kvCache.KeyQuantizedRowBytes;
        int vQuantRowBytes = kvCache.ValueQuantizedRowBytes;

        if (pool is null || numHeads < 2)
        {
            Span<float> tileScores = stackalloc float[MaxTileSize];
            for (int h = 0; h < numHeads; h++)
            {
                ExecuteTiledQuantizedHead(
                    q, kQuant, vQuant, kWindow, vWindow, output, tileScores,
                    seqQ, quantLen, windowLen, headDim, scale,
                    qStride, kvStride, kQuantRowBytes, vQuantRowBytes,
                    positionOffset, tileSize, slidingWindowSize ?? 0,
                    kvCache.KeyDType, kvCache.ValueDType,
                    h, h / (numHeads / numKvHeads),
                    kvCache.WindowCapacity);
            }
        }
        else
        {
            var ctx = new QuantizedTiledCtx
            {
                Q = q, KQuant = kQuant, VQuant = vQuant,
                KWindow = kWindow, VWindow = vWindow, Output = output,
                SeqQ = seqQ, QuantLen = quantLen, WindowLen = windowLen,
                NumHeads = numHeads, NumKvHeads = numKvHeads,
                HeadDim = headDim, Scale = scale,
                PositionOffset = positionOffset,
                GroupSize = numHeads / numKvHeads,
                QStride = qStride, KvStride = kvStride,
                KQuantRowBytes = kQuantRowBytes, VQuantRowBytes = vQuantRowBytes,
                TileSize = tileSize,
                SlidingWindowSize = slidingWindowSize ?? 0,
                KeyDType = kvCache.KeyDType, ValueDType = kvCache.ValueDType,
                WindowCapacity = kvCache.WindowCapacity
            };
            pool.Dispatch((nint)(&ctx), &QuantizedTiledAttentionWorker);
        }
    }

    private unsafe struct QuantizedTiledCtx
    {
        public float* Q;
        public byte* KQuant;
        public byte* VQuant;
        public float* KWindow;
        public float* VWindow;
        public float* Output;
        public int SeqQ;
        public int QuantLen;
        public int WindowLen;
        public int NumHeads;
        public int NumKvHeads;
        public int HeadDim;
        public float Scale;
        public int PositionOffset;
        public int GroupSize;
        public int QStride;
        public int KvStride;
        public int KQuantRowBytes;
        public int VQuantRowBytes;
        public int TileSize;
        public int SlidingWindowSize;
        public KvCacheDType KeyDType;
        public KvCacheDType ValueDType;
        public int WindowCapacity;
    }

    [SkipLocalsInit]
    private static unsafe void QuantizedTiledAttentionWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<QuantizedTiledCtx>((void*)ctxPtr);

        int headsPerThread = (ctx.NumHeads + threadCount - 1) / threadCount;
        int startHead = threadIdx * headsPerThread;
        int endHead = Math.Min(startHead + headsPerThread, ctx.NumHeads);
        if (startHead >= ctx.NumHeads) return;

        Span<float> tileScores = stackalloc float[MaxTileSize];

        for (int h = startHead; h < endHead; h++)
        {
            ExecuteTiledQuantizedHead(
                ctx.Q, ctx.KQuant, ctx.VQuant, ctx.KWindow, ctx.VWindow, ctx.Output, tileScores,
                ctx.SeqQ, ctx.QuantLen, ctx.WindowLen, ctx.HeadDim, ctx.Scale,
                ctx.QStride, ctx.KvStride, ctx.KQuantRowBytes, ctx.VQuantRowBytes,
                ctx.PositionOffset, ctx.TileSize, ctx.SlidingWindowSize,
                ctx.KeyDType, ctx.ValueDType,
                h, h / ctx.GroupSize,
                ctx.WindowCapacity);
        }
    }

    /// <summary>
    /// Processes a single head for quantized KV-cache attention.
    /// Phase 1: iterate tiles over quantized region with per-tile dequant.
    /// Phase 2: iterate tiles over full-precision window region.
    /// Uses online softmax throughout both phases.
    /// </summary>
    [SkipLocalsInit]
    private static unsafe void ExecuteTiledQuantizedHead(
        float* q, byte* kQuant, byte* vQuant, float* kWindow, float* vWindow, float* output,
        Span<float> tileScores,
        int seqQ, int quantLen, int windowLen, int headDim, float scale,
        int qStride, int kvStride, int kQuantRowBytes, int vQuantRowBytes,
        int positionOffset, int tileSize, int slidingWindowSize,
        KvCacheDType keyDType, KvCacheDType valueDType,
        int headIdx, int kvHeadIdx,
        int windowCapacity)
    {
        int seqKv = quantLen + windowLen;
        int window = slidingWindowSize;

        // Per-tile scratch for dequantized K and V rows
        // Budget: tileSize * headDim * sizeof(float) per buffer
        int scratchElems = tileSize * headDim;
        float* kScratch;
        float* vScratch;
        bool scratchAllocated = false;

        int scratchBytes = scratchElems * sizeof(float);
        if (scratchBytes <= StackAllocThreshold)
        {
            float* kBuf = stackalloc float[scratchElems];
            float* vBuf = stackalloc float[scratchElems];
            kScratch = kBuf;
            vScratch = vBuf;
        }
        else
        {
            kScratch = (float*)NativeMemory.AlignedAlloc((nuint)scratchBytes, 64);
            vScratch = (float*)NativeMemory.AlignedAlloc((nuint)scratchBytes, 64);
            scratchAllocated = true;
        }

        try
        {
            for (int i = 0; i < seqQ; i++)
            {
                var qRow = new ReadOnlySpan<float>(q + i * qStride + headIdx * headDim, headDim);
                var outRow = new Span<float>(output + i * qStride + headIdx * headDim, headDim);
                outRow.Clear();

                int visibleEnd = Math.Min(seqKv, positionOffset + i + 1);
                int visibleStart = (window > 0)
                    ? Math.Max(0, positionOffset + i - window + 1)
                    : 0;
                if (visibleStart >= visibleEnd) continue;

                float maxSoFar = float.NegativeInfinity;
                float sumExp = 0f;

                // ── Phase 1: Quantized region [visibleStart..min(quantLen, visibleEnd)) ──
                int quantEnd = Math.Min(quantLen, visibleEnd);
                int quantStart = Math.Max(visibleStart, 0);

                for (int tileBase = quantStart; tileBase < quantEnd; tileBase += tileSize)
                {
                    int tileLen = Math.Min(tileSize, quantEnd - tileBase);
                    var scores = tileScores.Slice(0, tileLen);

                    // Dequantize tile of K for this head
                    DequantTile(kQuant, tileBase, tileLen, kvStride, kQuantRowBytes,
                                kvHeadIdx, headDim, keyDType, kScratch);

                    // Compute scores: dot(qRow, kScratch[j*headDim .. (j+1)*headDim]) * scale
                    for (int j = 0; j < tileLen; j++)
                    {
                        var kRow = new ReadOnlySpan<float>(kScratch + j * headDim, headDim);
                        scores[j] = TensorPrimitives.Dot(qRow, kRow) * scale;
                    }

                    // Online softmax update
                    float tileMax = TensorPrimitives.Max(scores);
                    float newMax = MathF.Max(maxSoFar, tileMax);
                    float correction = FastMath.FastExp(maxSoFar - newMax);

                    if (correction < 1f)
                    {
                        sumExp *= correction;
                        TensorPrimitives.Multiply(outRow, correction, outRow);
                    }

                    sumExp += FastMath.ExpSumAndStore(scores, scores, -newMax);

                    // Dequantize tile of V for this head and accumulate
                    DequantTile(vQuant, tileBase, tileLen, kvStride, vQuantRowBytes,
                                kvHeadIdx, headDim, valueDType, vScratch);

                    for (int j = 0; j < tileLen; j++)
                    {
                        float w = scores[j];
                        if (w == 0f) continue;
                        var vRow = new ReadOnlySpan<float>(vScratch + j * headDim, headDim);
                        TensorPrimitives.MultiplyAdd(vRow, w, outRow, outRow);
                    }

                    maxSoFar = newMax;
                }

                // ── Phase 2: Window region [max(quantLen, visibleStart)..visibleEnd) ──
                int windowStart = Math.Max(quantLen, visibleStart);
                for (int tileBase = windowStart; tileBase < visibleEnd; tileBase += tileSize)
                {
                    int tileLen = Math.Min(tileSize, visibleEnd - tileBase);
                    var scores = tileScores.Slice(0, tileLen);

                    for (int j = 0; j < tileLen; j++)
                    {
                        // Map logical position to ring buffer index
                        int ringIdx = (tileBase + j) % windowCapacity;
                        var kRow = new ReadOnlySpan<float>(
                            kWindow + ringIdx * kvStride + kvHeadIdx * headDim, headDim);
                        scores[j] = TensorPrimitives.Dot(qRow, kRow) * scale;
                    }

                    float tileMax = TensorPrimitives.Max(scores);
                    float newMax = MathF.Max(maxSoFar, tileMax);
                    float correction = FastMath.FastExp(maxSoFar - newMax);

                    if (correction < 1f)
                    {
                        sumExp *= correction;
                        TensorPrimitives.Multiply(outRow, correction, outRow);
                    }

                    sumExp += FastMath.ExpSumAndStore(scores, scores, -newMax);

                    for (int j = 0; j < tileLen; j++)
                    {
                        float w = scores[j];
                        if (w == 0f) continue;
                        int ringIdx = (tileBase + j) % windowCapacity;
                        var vRow = new ReadOnlySpan<float>(
                            vWindow + ringIdx * kvStride + kvHeadIdx * headDim, headDim);
                        TensorPrimitives.MultiplyAdd(vRow, w, outRow, outRow);
                    }

                    maxSoFar = newMax;
                }

                if (sumExp > 0f)
                    TensorPrimitives.Multiply(outRow, 1f / sumExp, outRow);
            }
        }
        finally
        {
            if (scratchAllocated)
            {
                NativeMemory.AlignedFree(kScratch);
                NativeMemory.AlignedFree(vScratch);
            }
        }
    }

    /// <summary>
    /// Dequantizes a tile of KV data (Q8_0 or Q4_0) for a specific head into the scratch buffer.
    /// Output layout: <c>[tileLen, headDim]</c> contiguous floats.
    /// </summary>
    [SkipLocalsInit]
    private static unsafe void DequantTile(byte* quantData, int tileBase, int tileLen,
                                            int kvStride, int quantRowBytes,
                                            int kvHeadIdx, int headDim,
                                            KvCacheDType dtype, float* scratch)
    {
        int headOffset = kvHeadIdx * headDim;

        if (dtype == KvCacheDType.Q8_0)
        {
            int blockSize = 32;
            int blockBytes = 34;
            // Offset within a quantized row to reach the start of this head's data
            int headBlockStart = headOffset / blockSize;
            int headBlocks = headDim / blockSize;
            int headQuantOffset = headBlockStart * blockBytes;

            for (int j = 0; j < tileLen; j++)
            {
                byte* rowStart = quantData + (long)(tileBase + j) * quantRowBytes + headQuantOffset;
                float* dst = scratch + j * headDim;
                KvQuantize.Q8_0ToF32(rowStart, dst, headDim);
            }
        }
        else // Q4_0
        {
            int blockSize = 32;
            int blockBytes = 18;
            int headBlockStart = headOffset / blockSize;
            int headBlocks = headDim / blockSize;
            int headQuantOffset = headBlockStart * blockBytes;

            for (int j = 0; j < tileLen; j++)
            {
                byte* rowStart = quantData + (long)(tileBase + j) * quantRowBytes + headQuantOffset;
                float* dst = scratch + j * headDim;
                KvQuantize.Q4_0ToF32(rowStart, dst, headDim);
            }
        }
    }
}
