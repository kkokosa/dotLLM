using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Rotary Position Embedding (RoPE) kernel. Applies pair-wise rotation to query and key vectors
/// using pre-computed cos/sin frequency tables.
/// <para>
/// For each dimension pair <c>(2i, 2i+1)</c> in a head:
/// <code>
/// q'[2i]   = q[2i]   * cos(pos * θ_i) - q[2i+1] * sin(pos * θ_i)
/// q'[2i+1] = q[2i]   * sin(pos * θ_i) + q[2i+1] * cos(pos * θ_i)
/// </code>
/// where <c>θ_i = theta^(-2i / headDim)</c>.
/// </para>
/// </summary>
public static class RoPE
{
    /// <summary>Stackalloc threshold in bytes. Above this, use ArrayPool.</summary>
    private const int StackAllocThreshold = 8192;

    /// <summary>
    /// AVX2 deinterleave permutation indices: lower lane [0,2,4,6] gathers evens,
    /// upper lane [1,3,5,7] gathers odds. Stored as RVA static data to avoid
    /// re-materialization at each inlined call site.
    /// </summary>
    private static ReadOnlySpan<int> DeinterleaveIndices => [0, 2, 4, 6, 1, 3, 5, 7];

    /// <summary>
    /// Pre-computes cos/sin frequency tables for RoPE. Tables are indexed as
    /// <c>table[pos * halfDim + i]</c> where <c>halfDim = headDim / 2</c>.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length to pre-compute.</param>
    /// <param name="headDim">Dimension per attention head (must be even).</param>
    /// <param name="theta">Base frequency (e.g., 10000.0 for Llama 2, 500000.0 for Llama 3).</param>
    /// <param name="cosTable">Destination for cosine values. Must have length &gt;= <paramref name="maxSeqLen"/> * <paramref name="headDim"/> / 2.</param>
    /// <param name="sinTable">Destination for sine values. Must have length &gt;= <paramref name="maxSeqLen"/> * <paramref name="headDim"/> / 2.</param>
    [SkipLocalsInit]
    public static void PrecomputeFrequencyTable(int maxSeqLen, int headDim, float theta,
                                                 Span<float> cosTable, Span<float> sinTable)
    {
        if (headDim <= 0 || headDim % 2 != 0)
            throw new ArgumentException($"headDim must be a positive even number, got {headDim}", nameof(headDim));

        int halfDim = headDim / 2;

        // Frequencies depend only on dimension index, not on position.
        // Precompute once to avoid calling MathF.Pow O(maxSeqLen × halfDim) times —
        // for headDim=128 and seqLen=4096 that is 4096× fewer Pow calls.
        if (halfDim * sizeof(float) <= StackAllocThreshold)
        {
            Span<float> freqs = stackalloc float[halfDim];
            FillTables(maxSeqLen, headDim, theta, cosTable, sinTable, freqs);
        }
        else
        {
            float[] rented = ArrayPool<float>.Shared.Rent(halfDim);
            FillTables(maxSeqLen, headDim, theta, cosTable, sinTable, rented.AsSpan(0, halfDim));
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FillTables(int maxSeqLen, int headDim, float theta,
                                    Span<float> cosTable, Span<float> sinTable, Span<float> freqs)
    {
        int halfDim = freqs.Length;
        for (int i = 0; i < halfDim; i++)
            freqs[i] = 1.0f / MathF.Pow(theta, 2.0f * i / headDim);

        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            int tableBase = pos * halfDim;
            for (int i = 0; i < halfDim; i++)
            {
                float angle = pos * freqs[i];
                cosTable[tableBase + i] = MathF.Cos(angle);
                sinTable[tableBase + i] = MathF.Sin(angle);
            }
        }
    }

    /// <summary>
    /// Scalar reference implementation of <see cref="PrecomputeFrequencyTable"/> for correctness verification.
    /// </summary>
    [SkipLocalsInit]
    internal static void PrecomputeFrequencyTableScalar(int maxSeqLen, int headDim, float theta,
                                                         Span<float> cosTable, Span<float> sinTable)
    {
        int halfDim = headDim / 2;

        for (int pos = 0; pos < maxSeqLen; pos++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, 2.0f * i / headDim);
                float angle = pos * freq;
                cosTable[pos * halfDim + i] = MathF.Cos(angle);
                sinTable[pos * halfDim + i] = MathF.Sin(angle);
            }
        }
    }

    /// <summary>
    /// Applies RoPE rotation to a single head vector in-place using pre-computed cos/sin values.
    /// Processes dimension pairs: <c>vec[2i]' = vec[2i]*cos[i] - vec[2i+1]*sin[i]</c>,
    /// <c>vec[2i+1]' = vec[2i]*sin[i] + vec[2i+1]*cos[i]</c>.
    /// </summary>
    /// <param name="vec">Head vector to rotate in-place. Length must equal <paramref name="headDim"/>.</param>
    /// <param name="cos">Cosine table slice for this position. Length must equal headDim / 2.</param>
    /// <param name="sin">Sine table slice for this position. Length must equal headDim / 2.</param>
    /// <param name="headDim">Dimension per head (must be even).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    [SkipLocalsInit]
    public static void ApplyRotation(Span<float> vec, ReadOnlySpan<float> cos,
                                      ReadOnlySpan<float> sin, int headDim)
    {
        int halfDim = headDim / 2;
        int i = 0;

        if (Avx2.IsSupported && halfDim >= 4)
        {
            // Process 4 dimension pairs (8 floats) per iteration.
            // Deinterleave even/odd via a single permute, apply rotation, re-interleave.
            var deinterleaveIdx = Vector256.Create(DeinterleaveIndices);

            for (; i + 4 <= halfDim; i += 4)
            {
                int vecOffset = i * 2;

                // Load 8 interleaved floats: [e0, o0, e1, o1, e2, o2, e3, o3]
                var interleaved = Vector256.LoadUnsafe(ref vec[vecOffset]);

                // Single permute — extract lower (evens) and upper (odds) from same result.
                var permuted = Avx2.PermuteVar8x32(interleaved.AsSingle(), deinterleaveIdx);
                var even = permuted.GetLower(); // [e0, e1, e2, e3]
                var odd  = permuted.GetUpper(); // [o0, o1, o2, o3]

                // Load cos/sin for these 4 pairs
                var cosVec = Vector128.LoadUnsafe(in cos[i]);
                var sinVec = Vector128.LoadUnsafe(in sin[i]);

                // Rotation: even' = even*cos - odd*sin = -(odd*sin) + even*cos
                //           odd'  = even*sin + odd*cos
                Vector128<float> newEven, newOdd;
                if (Fma.IsSupported)
                {
                    // MultiplyAddNegated(a, b, c) = -(a*b) + c = c - a*b
                    // even' = -(odd * sin) + (even * cos)
                    newEven = Fma.MultiplyAddNegated(odd, sinVec, even * cosVec);
                    // odd'  = (even * sin) + (odd * cos)
                    newOdd  = Fma.MultiplyAdd(even, sinVec, odd * cosVec);
                }
                else
                {
                    newEven = even * cosVec - odd * sinVec;
                    newOdd  = even * sinVec + odd * cosVec;
                }

                // Re-interleave: [ne0, no0, ne1, no1, ne2, no2, ne3, no3]
                var lo = Sse.UnpackLow(newEven, newOdd);   // [ne0, no0, ne1, no1]
                var hi = Sse.UnpackHigh(newEven, newOdd);  // [ne2, no2, ne3, no3]
                var result = Vector256.Create(lo, hi);

                result.StoreUnsafe(ref vec[vecOffset]);
            }
        }

        // Scalar remainder / fallback
        for (; i < halfDim; i++)
        {
            float e = vec[2 * i];
            float o = vec[2 * i + 1];
            vec[2 * i] = e * cos[i] - o * sin[i];
            vec[2 * i + 1] = e * sin[i] + o * cos[i];
        }
    }

    /// <summary>
    /// Scalar reference implementation of <see cref="ApplyRotation"/> for correctness verification.
    /// </summary>
    [SkipLocalsInit]
    internal static void ApplyRotationScalar(Span<float> vec, ReadOnlySpan<float> cos,
                                              ReadOnlySpan<float> sin, int headDim)
    {
        int halfDim = headDim / 2;
        for (int i = 0; i < halfDim; i++)
        {
            float e = vec[2 * i];
            float o = vec[2 * i + 1];
            vec[2 * i] = e * cos[i] - o * sin[i];
            vec[2 * i + 1] = e * sin[i] + o * cos[i];
        }
    }

    /// <summary>
    /// Applies RoPE to query and key tensors. Data layout: <c>[seqLen, numHeads * headDim]</c> —
    /// one row per token, all heads concatenated. Convenience overload that rotates all <paramref name="headDim"/> dimensions.
    /// </summary>
    /// <param name="q">Query tensor (modified in-place). Layout: <c>[seqLen, numHeads * headDim]</c>.</param>
    /// <param name="k">Key tensor (modified in-place). Layout: <c>[seqLen, numKvHeads * headDim]</c>.</param>
    /// <param name="positions">Position index per token. Length must equal the sequence length.</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Dimension per head (must be even).</param>
    /// <param name="cosTable">Pre-computed cosine table from <see cref="PrecomputeFrequencyTable"/>.</param>
    /// <param name="sinTable">Pre-computed sine table from <see cref="PrecomputeFrequencyTable"/>.</param>
    [SkipLocalsInit]
    public static void Execute(Span<float> q, Span<float> k, ReadOnlySpan<int> positions,
                                int numHeads, int numKvHeads, int headDim,
                                ReadOnlySpan<float> cosTable, ReadOnlySpan<float> sinTable)
        => Execute(q, k, positions, numHeads, numKvHeads, headDim, headDim, cosTable, sinTable);

    /// <summary>
    /// Applies RoPE to query and key tensors with partial rotation support. Data layout:
    /// <c>[seqLen, numHeads * headDim]</c> — one row per token, all heads concatenated.
    /// When <paramref name="ropeDim"/> &lt; <paramref name="headDim"/>, only the first
    /// <paramref name="ropeDim"/> dimensions of each head are rotated; the rest pass through unchanged.
    /// </summary>
    /// <param name="q">Query tensor (modified in-place). Layout: <c>[seqLen, numHeads * headDim]</c>.</param>
    /// <param name="k">Key tensor (modified in-place). Layout: <c>[seqLen, numKvHeads * headDim]</c>.</param>
    /// <param name="positions">Position index per token. Length must equal the sequence length.</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Full dimension per head (used for stride computation).</param>
    /// <param name="ropeDim">Number of dimensions to rotate (must be even, &lt;= <paramref name="headDim"/>). Cos/sin tables must be sized for this value.</param>
    /// <param name="cosTable">Pre-computed cosine table from <see cref="PrecomputeFrequencyTable"/>.</param>
    /// <param name="sinTable">Pre-computed sine table from <see cref="PrecomputeFrequencyTable"/>.</param>
    [SkipLocalsInit]
    public static void Execute(Span<float> q, Span<float> k, ReadOnlySpan<int> positions,
                                int numHeads, int numKvHeads, int headDim, int ropeDim,
                                ReadOnlySpan<float> cosTable, ReadOnlySpan<float> sinTable)
    {
        int halfRopeDim = ropeDim / 2;
        int qStride = numHeads * headDim;
        int kStride = numKvHeads * headDim;
        int seqLen = positions.Length;

        for (int t = 0; t < seqLen; t++)
        {
            int pos = positions[t];
            var cos = cosTable.Slice(pos * halfRopeDim, halfRopeDim);
            var sin = sinTable.Slice(pos * halfRopeDim, halfRopeDim);

            // Rotate all Q heads for this token.
            for (int h = 0; h < numHeads; h++)
            {
                var headSlice = q.Slice(t * qStride + h * headDim, ropeDim);
                ApplyRotation(headSlice, cos, sin, ropeDim);
            }

            // Rotate all K heads for this token.
            for (int h = 0; h < numKvHeads; h++)
            {
                var headSlice = k.Slice(t * kStride + h * headDim, ropeDim);
                ApplyRotation(headSlice, cos, sin, ropeDim);
            }
        }
    }

    /// <summary>
    /// Scalar reference implementation of <see cref="Execute(Span{float}, Span{float}, ReadOnlySpan{int}, int, int, int, ReadOnlySpan{float}, ReadOnlySpan{float})"/> for correctness verification.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(Span<float> q, Span<float> k, ReadOnlySpan<int> positions,
                                        int numHeads, int numKvHeads, int headDim,
                                        ReadOnlySpan<float> cosTable, ReadOnlySpan<float> sinTable)
        => ExecuteScalar(q, k, positions, numHeads, numKvHeads, headDim, headDim, cosTable, sinTable);

    /// <summary>
    /// Scalar reference implementation of <see cref="Execute(Span{float}, Span{float}, ReadOnlySpan{int}, int, int, int, int, ReadOnlySpan{float}, ReadOnlySpan{float})"/> for correctness verification.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(Span<float> q, Span<float> k, ReadOnlySpan<int> positions,
                                        int numHeads, int numKvHeads, int headDim, int ropeDim,
                                        ReadOnlySpan<float> cosTable, ReadOnlySpan<float> sinTable)
    {
        int halfRopeDim = ropeDim / 2;
        int qStride = numHeads * headDim;
        int kStride = numKvHeads * headDim;
        int seqLen = positions.Length;

        for (int t = 0; t < seqLen; t++)
        {
            int pos = positions[t];
            var cos = cosTable.Slice(pos * halfRopeDim, halfRopeDim);
            var sin = sinTable.Slice(pos * halfRopeDim, halfRopeDim);

            for (int h = 0; h < numHeads; h++)
            {
                var headSlice = q.Slice(t * qStride + h * headDim, ropeDim);
                ApplyRotationScalar(headSlice, cos, sin, ropeDim);
            }

            for (int h = 0; h < numKvHeads; h++)
            {
                var headSlice = k.Slice(t * kStride + h * headDim, ropeDim);
                ApplyRotationScalar(headSlice, cos, sin, ropeDim);
            }
        }
    }
}
