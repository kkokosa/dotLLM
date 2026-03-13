using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Tests for sliding window attention masking.
/// Verifies that positions outside the sliding window are masked to -inf.
/// </summary>
public class SlidingWindowAttentionTests
{
    /// <summary>
    /// Null sliding window = full causal attention (unchanged behavior).
    /// </summary>
    [Fact]
    public void NullWindow_BehavesLikeFullCausal()
    {
        // 4 tokens attending to 4 tokens, no sliding window
        int seqQ = 4, seqKv = 4, numHeads = 1, numKvHeads = 1, headDim = 4;
        float[] q = new float[seqQ * headDim];
        float[] k = new float[seqKv * headDim];
        float[] v = new float[seqKv * headDim];
        float[] outWith = new float[seqQ * headDim];
        float[] outWithout = new float[seqQ * headDim];

        // Fill with simple values
        for (int i = 0; i < q.Length; i++) q[i] = 0.1f * (i + 1);
        for (int i = 0; i < k.Length; i++) k[i] = 0.1f * (i + 1);
        for (int i = 0; i < v.Length; i++) v[i] = 0.1f * (i + 1);

        Attention.ExecuteScalar(q, k, v, outWith, seqQ, seqKv, numHeads, numKvHeads, headDim, 0, slidingWindowSize: null);
        Attention.ExecuteScalar(q, k, v, outWithout, seqQ, seqKv, numHeads, numKvHeads, headDim, 0);

        for (int i = 0; i < outWith.Length; i++)
            Assert.Equal(outWithout[i], outWith[i], precision: 6);
    }

    /// <summary>
    /// Sliding window = 1 means each token can only attend to itself.
    /// </summary>
    [Fact]
    public void Window1_OnlyAttendsSelf()
    {
        int seqQ = 3, seqKv = 3, numHeads = 1, numKvHeads = 1, headDim = 2;
        float[] q = [1, 0, 0, 1, 1, 1];
        float[] k = [1, 0, 0, 1, 1, 1];
        float[] v = [1, 2, 3, 4, 5, 6];
        float[] output = new float[seqQ * headDim];

        // With window=1, each query can only attend to one KV position (itself)
        Attention.ExecuteScalar(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim, 0, slidingWindowSize: 1);

        // Token 0 attends only to pos 0 -> output = V[0] = [1, 2]
        Assert.Equal(1f, output[0], precision: 5);
        Assert.Equal(2f, output[1], precision: 5);

        // Token 1 attends only to pos 1 -> output = V[1] = [3, 4]
        Assert.Equal(3f, output[2], precision: 5);
        Assert.Equal(4f, output[3], precision: 5);

        // Token 2 attends only to pos 2 -> output = V[2] = [5, 6]
        Assert.Equal(5f, output[4], precision: 5);
        Assert.Equal(6f, output[5], precision: 5);
    }

    /// <summary>
    /// Sliding window larger than context has no effect (same as full causal).
    /// </summary>
    [Fact]
    public void WindowLargerThanContext_SameAsFullCausal()
    {
        int seqQ = 3, seqKv = 3, numHeads = 1, numKvHeads = 1, headDim = 2;
        float[] q = [1, 0, 0, 1, 1, 1];
        float[] k = [1, 0, 0, 1, 1, 1];
        float[] v = [1, 2, 3, 4, 5, 6];
        float[] outWindowed = new float[seqQ * headDim];
        float[] outFull = new float[seqQ * headDim];

        Attention.ExecuteScalar(q, k, v, outWindowed, seqQ, seqKv, numHeads, numKvHeads, headDim, 0, slidingWindowSize: 100);
        Attention.ExecuteScalar(q, k, v, outFull, seqQ, seqKv, numHeads, numKvHeads, headDim, 0, slidingWindowSize: null);

        for (int i = 0; i < outWindowed.Length; i++)
            Assert.Equal(outFull[i], outWindowed[i], precision: 6);
    }

    /// <summary>
    /// Sliding window with position offset (KV-cache decode scenario).
    /// </summary>
    [Fact]
    public void WindowWithPositionOffset_MasksCorrectly()
    {
        // Simulate decode: 1 query token at position 5, with 6 KV tokens (positions 0-5), window=3
        // Token at pos 5 with window=3 can see positions 3, 4, 5
        int seqQ = 1, seqKv = 6, numHeads = 1, numKvHeads = 1, headDim = 2;
        float[] q = [1, 0];
        float[] k = new float[seqKv * headDim];
        float[] v = new float[seqKv * headDim];
        // Set all K to same value so dot products are equal
        for (int i = 0; i < k.Length; i++) k[i] = 1;
        // Set V with distinct values per position
        for (int i = 0; i < seqKv; i++)
        {
            v[i * headDim] = i;
            v[i * headDim + 1] = i * 10;
        }

        float[] output = new float[headDim];
        Attention.ExecuteScalar(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim, 5, slidingWindowSize: 3);

        // Positions 0, 1, 2 are masked out. Only positions 3, 4, 5 visible.
        // Equal K values -> uniform attention over visible positions -> average of V[3], V[4], V[5]
        float expectedD0 = (3f + 4f + 5f) / 3f;
        float expectedD1 = (30f + 40f + 50f) / 3f;
        Assert.Equal(expectedD0, output[0], precision: 4);
        Assert.Equal(expectedD1, output[1], precision: 4);
    }

    /// <summary>
    /// Optimized Execute path should match scalar reference with sliding window.
    /// </summary>
    [Fact]
    public void Execute_MatchesScalar_WithSlidingWindow()
    {
        int seqQ = 3, seqKv = 3, numHeads = 2, numKvHeads = 1, headDim = 4;
        float[] q = new float[seqQ * numHeads * headDim];
        float[] k = new float[seqKv * numKvHeads * headDim];
        float[] v = new float[seqKv * numKvHeads * headDim];
        var rng = new Random(42);
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() - 0.5);

        float[] outExec = new float[seqQ * numHeads * headDim];
        float[] outScalar = new float[seqQ * numHeads * headDim];

        Attention.Execute(q, k, v, outExec, seqQ, seqKv, numHeads, numKvHeads, headDim, 0, slidingWindowSize: 2);
        Attention.ExecuteScalar(q, k, v, outScalar, seqQ, seqKv, numHeads, numKvHeads, headDim, 0, slidingWindowSize: 2);

        // Tolerance widened from precision: 5 to account for fast approximate exp in attention.
        for (int i = 0; i < outExec.Length; i++)
            Assert.Equal(outScalar[i], outExec[i], 5e-2f);
    }

    /// <summary>
    /// ApplyCausalMask with sliding window masks correct positions.
    /// </summary>
    [Fact]
    public void ApplyCausalMask_SlidingWindow_MasksCorrectPositions()
    {
        int seqQ = 4, seqKv = 4;
        float[] scores = new float[seqQ * seqKv];
        for (int i = 0; i < scores.Length; i++) scores[i] = 1.0f;

        // Window = 2, positionOffset = 0
        // q_i at position i can attend to positions max(0, i-1) .. i
        Attention.ApplyCausalMask(scores, seqQ, seqKv, 0, slidingWindowSize: 2);

        // Row 0 (pos 0): can see 0, not 1,2,3
        Assert.Equal(1.0f, scores[0]);
        Assert.Equal(float.NegativeInfinity, scores[1]); // future
        Assert.Equal(float.NegativeInfinity, scores[2]);
        Assert.Equal(float.NegativeInfinity, scores[3]);

        // Row 1 (pos 1): can see 0,1, not 2,3
        Assert.Equal(1.0f, scores[4]);
        Assert.Equal(1.0f, scores[5]);
        Assert.Equal(float.NegativeInfinity, scores[6]);
        Assert.Equal(float.NegativeInfinity, scores[7]);

        // Row 2 (pos 2): can see 1,2, not 0,3
        Assert.Equal(float.NegativeInfinity, scores[8]); // outside window
        Assert.Equal(1.0f, scores[9]);
        Assert.Equal(1.0f, scores[10]);
        Assert.Equal(float.NegativeInfinity, scores[11]);

        // Row 3 (pos 3): can see 2,3, not 0,1
        Assert.Equal(float.NegativeInfinity, scores[12]);
        Assert.Equal(float.NegativeInfinity, scores[13]); // outside window
        Assert.Equal(1.0f, scores[14]);
        Assert.Equal(1.0f, scores[15]);
    }
}
