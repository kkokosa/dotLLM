using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
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
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                Span<float> output,
                                int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                int positionOffset)
        => Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                   positionOffset, 1.0f / MathF.Sqrt(headDim));

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
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                Span<float> output,
                                int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                int positionOffset, float scale)
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

        // Allocate scores buffer: one head at a time, reused.
        // Follows the MatMul stackalloc/ArrayPool pattern.
        if (scoreSize * sizeof(float) <= StackAllocThreshold)
        {
            Span<float> scores = stackalloc float[scoreSize];
            ExecuteCore(q, k, v, output, scores, seqQ, seqKv, numHeads, headDim,
                        groupSize, scale, qStride, kvStride, positionOffset);
        }
        else
        {
            float[] rented = ArrayPool<float>.Shared.Rent(scoreSize);
            ExecuteCore(q, k, v, output, rented.AsSpan(0, scoreSize), seqQ, seqKv, numHeads, headDim,
                        groupSize, scale, qStride, kvStride, positionOffset);
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    private static void ExecuteCore(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                     Span<float> output, Span<float> scores,
                                     int seqQ, int seqKv, int numHeads, int headDim,
                                     int groupSize, float scale, int qStride, int kvStride,
                                     int positionOffset)
    {
        for (int h = 0; h < numHeads; h++)
        {
            int kvH = h / groupSize;

            // 1. Scaled dot-product scores: Q_h @ K_kvH^T, scaled
            ScaledDotProductScores(q, k, scores, seqQ, seqKv, headDim, scale,
                                   h, kvH, qStride, kvStride);

            // 2. Apply causal mask
            ApplyCausalMask(scores, seqQ, seqKv, positionOffset);

            // 3. Softmax per row
            for (int i = 0; i < seqQ; i++)
            {
                var row = scores.Slice(i * seqKv, seqKv);
                Softmax.Execute(row, row);
            }

            // 4. Weighted sum: weights @ V_kvH → output_h
            WeightedValues(scores, v, output, seqQ, seqKv, headDim,
                           h, kvH, qStride, kvStride);
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
                                      int positionOffset, ComputeThreadPool? pool)
        => Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                   positionOffset, 1.0f / MathF.Sqrt(headDim), pool);

    /// <summary>
    /// Pointer-based attention with caller-provided scale and optional head-parallel execution.
    /// </summary>
    [SkipLocalsInit]
    public static unsafe void Execute(float* q, float* k, float* v, float* output,
                                      int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                      int positionOffset, float scale, ComputeThreadPool? pool)
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
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, scale);
            return;
        }

        // Pre-allocate per-worker scores buffers via pool scratch
        int scoreSize = seqQ * seqKv;
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
            ScratchPtrs = scratchPtrs
        };
        pool.Dispatch((nint)(&ctx), &AttentionWorker);
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

        for (int h = startHead; h < endHead; h++)
        {
            int kvH = h / ctx.GroupSize;

            ScaledDotProductScores(qSpan, kSpan, scoresSpan, ctx.SeqQ, ctx.SeqKv, ctx.HeadDim, ctx.Scale,
                                   h, kvH, ctx.QStride, ctx.KvStride);

            ApplyCausalMask(scoresSpan, ctx.SeqQ, ctx.SeqKv, ctx.PositionOffset);

            for (int i = 0; i < ctx.SeqQ; i++)
            {
                var row = scoresSpan.Slice(i * ctx.SeqKv, ctx.SeqKv);
                Softmax.Execute(row, row);
            }

            WeightedValues(scoresSpan, vSpan, outSpan, ctx.SeqQ, ctx.SeqKv, ctx.HeadDim,
                           h, kvH, ctx.QStride, ctx.KvStride);
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
                                        int positionOffset)
        => ExecuteScalar(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                         positionOffset, 1.0f / MathF.Sqrt(headDim));

    /// <summary>
    /// Scalar reference implementation with caller-provided scale.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                        Span<float> output,
                                        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                        int positionOffset, float scale)
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
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ApplyCausalMask(Span<float> scores, int seqQ, int seqKv, int positionOffset)
    {
        for (int i = 0; i < seqQ; i++)
        {
            for (int j = positionOffset + i + 1; j < seqKv; j++)
            {
                scores[i * seqKv + j] = float.NegativeInfinity;
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
}
