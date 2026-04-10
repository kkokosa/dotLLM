using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Top-K sampling: keeps only the K highest-probability logits, setting the rest to -infinity.
/// </summary>
/// <remarks>
/// Uses a size-K min-heap to find the K-th largest logit in O(N log K) time and K memory,
/// instead of fully sorting the vocabulary. For the common case of K ≪ N (e.g. K=40 on a
/// 128K vocab) this is ~5–15× faster than a full <c>Array.Sort</c> and touches only a
/// stack-resident scratch buffer.
/// </remarks>
public sealed class TopKSampler : ISamplerStep
{
    /// <summary>Stack-allocate the heap scratch for small K; rent from ArrayPool above this.</summary>
    private const int StackHeapThreshold = 512;

    private readonly int? _topK;

    /// <summary>Creates a top-K step that reads from <see cref="SamplerContext"/>.</summary>
    public TopKSampler() { }

    /// <summary>Creates a self-configured top-K step.</summary>
    /// <param name="topK">Number of top tokens to keep (ignores context).</param>
    public TopKSampler(int topK) => _topK = topK;

    /// <inheritdoc/>
    public void Apply(Span<float> logits, SamplerContext context)
    {
        int k = _topK ?? context.TopK;
        if (k <= 0 || k >= logits.Length)
            return;

        float threshold;
        if (k <= StackHeapThreshold)
        {
            Span<float> heap = stackalloc float[k];
            threshold = FindKthLargest(logits, heap);
        }
        else
        {
            float[] rented = ArrayPool<float>.Shared.Rent(k);
            try
            {
                threshold = FindKthLargest(logits, rented.AsSpan(0, k));
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        // Count tokens strictly above threshold to handle ties correctly.
        // Allow only enough ties to fill exactly K slots.
        int aboveCount = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > threshold)
                aboveCount++;
        }

        int tiesAllowed = k - aboveCount;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (v > threshold)
                continue;
            if (v == threshold && tiesAllowed > 0)
            {
                tiesAllowed--;
                continue;
            }
            logits[i] = float.NegativeInfinity;
        }
    }

    /// <summary>
    /// Returns the K-th largest value in <paramref name="logits"/> via a size-K min-heap.
    /// </summary>
    /// <param name="logits">Source logits (read-only, not mutated).</param>
    /// <param name="heap">Scratch buffer of length K.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float FindKthLargest(ReadOnlySpan<float> logits, Span<float> heap)
    {
        int k = heap.Length;

        // Seed the heap with the first K logits, then heapify (Floyd's O(K) build).
        logits[..k].CopyTo(heap);
        for (int i = (k >> 1) - 1; i >= 0; i--)
            SiftDown(heap, i);

        // For each remaining element, if greater than the root (current K-th largest),
        // replace the root and sift down to restore the min-heap property.
        for (int i = k; i < logits.Length; i++)
        {
            float v = logits[i];
            Debug.Assert(!float.IsNaN(v), "TopKSampler: logits must not contain NaN");
            if (v > heap[0])
            {
                heap[0] = v;
                SiftDown(heap, 0);
            }
        }

        return heap[0];
    }

    /// <summary>Sifts <paramref name="i"/> down to restore the min-heap property.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SiftDown(Span<float> heap, int i)
    {
        int n = heap.Length;
        float x = heap[i];
        while (true)
        {
            int left = (i << 1) + 1;
            if (left >= n)
                break;
            int right = left + 1;
            int smaller = (right < n && heap[right] < heap[left]) ? right : left;
            if (heap[smaller] >= x)
                break;
            heap[i] = heap[smaller];
            i = smaller;
        }
        heap[i] = x;
    }
}
