using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed class AttentionTests
{
    [Fact]
    public void Execute_SingleHead_SingleToken_SelfAttention()
    {
        // 1 head, 1 Q token, 1 KV token, headDim=4
        // Q = K = V = [1, 0, 0, 0]
        // Score = dot(Q, K) / sqrt(4) = 1 / 2 = 0.5
        // After softmax of single element → 1.0
        // Output = 1.0 * V = [1, 0, 0, 0]
        const int headDim = 4;
        float[] q = [1f, 0f, 0f, 0f];
        float[] k = [1f, 0f, 0f, 0f];
        float[] v = [1f, 0f, 0f, 0f];
        float[] output = new float[headDim];

        Attention.Execute(q, k, v, output, seqQ: 1, seqKv: 1,
                          numHeads: 1, numKvHeads: 1, headDim: headDim, positionOffset: 0);

        Assert.Equal(1f, output[0], 1e-5f);
        Assert.Equal(0f, output[1], 1e-5f);
        Assert.Equal(0f, output[2], 1e-5f);
        Assert.Equal(0f, output[3], 1e-5f);
    }

    [Fact]
    public void Execute_CausalMask_NoFutureLeakage()
    {
        // Prefill: 3 tokens, 1 head, headDim=2
        // Q = K = identity-like, V has unique values per position
        // Token 0 can only attend to token 0 (causal)
        // Token 1 can attend to tokens 0, 1
        // Token 2 can attend to tokens 0, 1, 2
        const int headDim = 2;
        const int seqLen = 3;

        // Q and K: each token has a distinct direction
        float[] q = [1f, 0f, 0f, 1f, 1f, 1f];
        float[] k = [1f, 0f, 0f, 1f, 1f, 1f];
        // V: distinct per position for tracing
        float[] v = [1f, 0f, 0f, 1f, 0f, 0f];
        float[] output = new float[seqLen * headDim];

        Attention.Execute(q, k, v, output, seqQ: seqLen, seqKv: seqLen,
                          numHeads: 1, numKvHeads: 1, headDim: headDim, positionOffset: 0);

        // Token 0 only sees token 0 → output should be V[0] = [1, 0]
        Assert.Equal(1f, output[0], 1e-4f);
        Assert.Equal(0f, output[1], 1e-4f);
    }

    [Fact]
    public void Execute_CausalMask_SingleDecode_AttendsAllPast()
    {
        // Decode: 1 new Q token, 3 cached KV tokens
        // positionOffset=2 means Q is at position 2, can attend to positions 0,1,2
        const int headDim = 2;
        float[] q = [1f, 0f]; // 1 Q token
        float[] k = [1f, 0f, 1f, 0f, 1f, 0f]; // 3 KV tokens
        float[] v = [1f, 0f, 0f, 1f, 0.5f, 0.5f]; // distinct V values
        float[] output = new float[headDim];

        Attention.Execute(q, k, v, output, seqQ: 1, seqKv: 3,
                          numHeads: 1, numKvHeads: 1, headDim: headDim, positionOffset: 2);

        // All 3 KV positions visible (positions 0,1,2 ≤ positionOffset + 0 = 2)
        // Equal Q·K scores → equal weights → output = mean(V)
        float expectedD0 = (1f + 0f + 0.5f) / 3f;
        float expectedD1 = (0f + 1f + 0.5f) / 3f;
        Assert.Equal(expectedD0, output[0], 1e-4f);
        Assert.Equal(expectedD1, output[1], 1e-4f);
    }

    [Fact]
    public void Execute_MHA_AllHeadsIndependent()
    {
        // 2 heads, numKvHeads=2 (MHA), headDim=2
        // Each head has different Q/K → different outputs
        const int headDim = 2;
        const int numHeads = 2;
        const int numKvHeads = 2;

        // Q: [head0: (1,0), head1: (0,1)]
        float[] q = [1f, 0f, 0f, 1f];
        // K: same
        float[] k = [1f, 0f, 0f, 1f];
        // V: head0 V = (10, 0), head1 V = (0, 20)
        float[] v = [10f, 0f, 0f, 20f];
        float[] output = new float[numHeads * headDim];

        Attention.Execute(q, k, v, output, seqQ: 1, seqKv: 1,
                          numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim, positionOffset: 0);

        // Head 0 attends to KV head 0 → output = V_head0 = [10, 0]
        Assert.Equal(10f, output[0], 1e-4f);
        Assert.Equal(0f, output[1], 1e-4f);
        // Head 1 attends to KV head 1 → output = V_head1 = [0, 20]
        Assert.Equal(0f, output[2], 1e-4f);
        Assert.Equal(20f, output[3], 1e-4f);
    }

    [Fact]
    public void Execute_GQA_HeadBroadcast()
    {
        // numHeads=4, numKvHeads=2 → groupSize=2
        // Q heads 0,1 share KV head 0; Q heads 2,3 share KV head 1
        const int headDim = 2;
        const int numHeads = 4;
        const int numKvHeads = 2;

        // Q: 4 heads, each headDim=2, single token
        float[] q = [1f, 0f, 0f, 1f, 1f, 0f, 0f, 1f];
        // K: 2 KV heads
        float[] k = [1f, 0f, 0f, 1f];
        // V: KV head 0 → (10, 0), KV head 1 → (0, 20)
        float[] v = [10f, 0f, 0f, 20f];
        float[] output = new float[numHeads * headDim];

        Attention.Execute(q, k, v, output, seqQ: 1, seqKv: 1,
                          numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim, positionOffset: 0);

        // Heads 0,1 → KV head 0: V = [10, 0]
        Assert.Equal(10f, output[0], 1e-4f);
        Assert.Equal(0f, output[1], 1e-4f);
        Assert.Equal(10f, output[2], 1e-4f);
        Assert.Equal(0f, output[3], 1e-4f);
        // Heads 2,3 → KV head 1: V = [0, 20]
        Assert.Equal(0f, output[4], 1e-4f);
        Assert.Equal(20f, output[5], 1e-4f);
        Assert.Equal(0f, output[6], 1e-4f);
        Assert.Equal(20f, output[7], 1e-4f);
    }

    [Fact]
    public void Execute_MQA_SingleKVHead()
    {
        // numHeads=4, numKvHeads=1 → all Q heads share single KV head
        const int headDim = 2;
        const int numHeads = 4;
        const int numKvHeads = 1;

        float[] q = [1f, 0f, 0f, 1f, 1f, 1f, -1f, 0f];
        float[] k = [1f, 0f]; // single KV head
        float[] v = [7f, 3f]; // single KV head
        float[] output = new float[numHeads * headDim];

        Attention.Execute(q, k, v, output, seqQ: 1, seqKv: 1,
                          numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim, positionOffset: 0);

        // All heads attend to the single KV → all outputs = V = [7, 3]
        for (int h = 0; h < numHeads; h++)
        {
            Assert.Equal(7f, output[h * headDim + 0], 1e-4f);
            Assert.Equal(3f, output[h * headDim + 1], 1e-4f);
        }
    }

    [Fact]
    public void Execute_ScalarMatchesSIMD()
    {
        const int headDim = 16;
        const int numHeads = 4;
        const int numKvHeads = 2;
        const int seqQ = 3;
        const int seqKv = 5;
        var rng = new Random(42);

        float[] q = RandomArray(rng, seqQ * numHeads * headDim);
        float[] k = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] v = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] outputSimd = new float[seqQ * numHeads * headDim];
        float[] outputScalar = new float[seqQ * numHeads * headDim];

        Attention.Execute(q, k, v, outputSimd, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: 0);
        Attention.ExecuteScalar(q, k, v, outputScalar, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: 0);

        for (int i = 0; i < outputSimd.Length; i++)
            Assert.Equal(outputScalar[i], outputSimd[i], 1e-4f);
    }

    [Fact]
    public void Execute_SoftmaxNormalization_WeightsSumToOne()
    {
        // Verify that softmax weights sum to 1.0 by checking output is convex combination
        const int headDim = 2;
        const int seqKv = 3;

        // All V vectors have same L1 norm, Q·K produces uniform scores
        float[] q = [1f, 1f];
        float[] k = [1f, 1f, 1f, 1f, 1f, 1f]; // all same
        float[] v = [1f, 0f, 0f, 1f, 0.5f, 0.5f];
        float[] output = new float[headDim];

        Attention.Execute(q, k, v, output, seqQ: 1, seqKv: seqKv,
                          numHeads: 1, numKvHeads: 1, headDim: headDim, positionOffset: seqKv - 1);

        // Uniform attention → output = mean(V)
        float expectedD0 = (1f + 0f + 0.5f) / 3f;
        float expectedD1 = (0f + 1f + 0.5f) / 3f;
        Assert.Equal(expectedD0, output[0], 1e-4f);
        Assert.Equal(expectedD1, output[1], 1e-4f);
    }

    [Fact]
    public void Execute_LargeScores_NumericallyStable()
    {
        // Large Q/K values should not produce NaN/Inf thanks to softmax stability
        const int headDim = 4;
        float[] q = [100f, 100f, 100f, 100f];
        float[] k = [100f, 100f, 100f, 100f, -100f, -100f, -100f, -100f];
        float[] v = [1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f];
        float[] output = new float[headDim];

        Attention.Execute(q, k, v, output, seqQ: 1, seqKv: 2,
                          numHeads: 1, numKvHeads: 1, headDim: headDim, positionOffset: 1);

        Assert.All(output, val => Assert.True(float.IsFinite(val), $"Non-finite value: {val}"));
    }

    [Fact]
    public void ApplyCausalMask_CorrectPositions()
    {
        const int seqQ = 3;
        const int seqKv = 5;
        const int positionOffset = 1; // Q positions are 1, 2, 3

        float[] scores = new float[seqQ * seqKv];
        Array.Fill(scores, 1.0f);

        Attention.ApplyCausalMask(scores, seqQ, seqKv, positionOffset);

        // Row 0: Q at pos 1, can attend j ≤ 1 → j=0,1 ok; j=2,3,4 masked
        Assert.Equal(1.0f, scores[0 * seqKv + 0]);
        Assert.Equal(1.0f, scores[0 * seqKv + 1]);
        Assert.Equal(float.NegativeInfinity, scores[0 * seqKv + 2]);
        Assert.Equal(float.NegativeInfinity, scores[0 * seqKv + 3]);
        Assert.Equal(float.NegativeInfinity, scores[0 * seqKv + 4]);

        // Row 1: Q at pos 2, can attend j ≤ 2
        Assert.Equal(1.0f, scores[1 * seqKv + 0]);
        Assert.Equal(1.0f, scores[1 * seqKv + 1]);
        Assert.Equal(1.0f, scores[1 * seqKv + 2]);
        Assert.Equal(float.NegativeInfinity, scores[1 * seqKv + 3]);
        Assert.Equal(float.NegativeInfinity, scores[1 * seqKv + 4]);

        // Row 2: Q at pos 3, can attend j ≤ 3
        Assert.Equal(1.0f, scores[2 * seqKv + 0]);
        Assert.Equal(1.0f, scores[2 * seqKv + 1]);
        Assert.Equal(1.0f, scores[2 * seqKv + 2]);
        Assert.Equal(1.0f, scores[2 * seqKv + 3]);
        Assert.Equal(float.NegativeInfinity, scores[2 * seqKv + 4]);
    }

    [Fact]
    public void ScaledDotProductScores_KnownValues()
    {
        // Q = [[1, 2]], K = [[3, 4], [5, 6]], headDim=2, scale = 1/sqrt(2)
        // score[0,0] = (1*3 + 2*4) * scale = 11 / sqrt(2)
        // score[0,1] = (1*5 + 2*6) * scale = 17 / sqrt(2)
        const int headDim = 2;
        float scale = 1.0f / MathF.Sqrt(headDim);

        float[] q = [1f, 2f];
        float[] k = [3f, 4f, 5f, 6f];
        float[] scores = new float[2]; // seqQ=1, seqKv=2

        Attention.ScaledDotProductScores(q, k, scores,
                                          seqQ: 1, seqKv: 2, headDim: headDim, scale: scale,
                                          headIdx: 0, kvHeadIdx: 0, qStride: headDim, kvStride: headDim);

        Assert.Equal(11f * scale, scores[0], 1e-5f);
        Assert.Equal(17f * scale, scores[1], 1e-5f);
    }

    [Fact]
    public void Execute_Prefill_CausalTriangular()
    {
        // Prefill: seqQ = seqKv = 4, positionOffset = 0
        // Each Q is identical, each K is identical → uniform scores within visible window
        // The output for token i should be mean(V[0..i])
        const int headDim = 2;
        const int seqLen = 4;

        float[] q = new float[seqLen * headDim];
        float[] k = new float[seqLen * headDim];
        float[] v = new float[seqLen * headDim];

        // All Q and K the same → equal dot products
        for (int t = 0; t < seqLen; t++)
        {
            q[t * headDim + 0] = 1f;
            q[t * headDim + 1] = 0f;
            k[t * headDim + 0] = 1f;
            k[t * headDim + 1] = 0f;
        }

        // V: unique per position
        v[0 * headDim + 0] = 1f; v[0 * headDim + 1] = 0f;
        v[1 * headDim + 0] = 0f; v[1 * headDim + 1] = 1f;
        v[2 * headDim + 0] = 1f; v[2 * headDim + 1] = 1f;
        v[3 * headDim + 0] = 0f; v[3 * headDim + 1] = 0f;

        float[] output = new float[seqLen * headDim];
        Attention.Execute(q, k, v, output, seqQ: seqLen, seqKv: seqLen,
                          numHeads: 1, numKvHeads: 1, headDim: headDim, positionOffset: 0);

        // Token 0: sees only V[0] → [1, 0]
        Assert.Equal(1f, output[0 * headDim + 0], 1e-4f);
        Assert.Equal(0f, output[0 * headDim + 1], 1e-4f);

        // Token 1: sees V[0], V[1] → mean = [0.5, 0.5]
        Assert.Equal(0.5f, output[1 * headDim + 0], 1e-4f);
        Assert.Equal(0.5f, output[1 * headDim + 1], 1e-4f);

        // Token 2: sees V[0..2] → mean = [2/3, 2/3]
        Assert.Equal(2f / 3f, output[2 * headDim + 0], 1e-4f);
        Assert.Equal(2f / 3f, output[2 * headDim + 1], 1e-4f);

        // Token 3: sees V[0..3] → mean = [0.5, 0.5]
        Assert.Equal(0.5f, output[3 * headDim + 0], 1e-4f);
        Assert.Equal(0.5f, output[3 * headDim + 1], 1e-4f);
    }

    [Fact]
    public void Execute_ZeroHeadDim_Throws()
    {
        float[] q = [1f];
        float[] k = [1f];
        float[] v = [1f];
        float[] output = new float[1];

        Assert.Throws<ArgumentException>(() =>
            Attention.Execute(q, k, v, output, seqQ: 1, seqKv: 1,
                              numHeads: 1, numKvHeads: 1, headDim: 0, positionOffset: 0));
    }

    [Fact]
    public void Execute_HeadMismatch_Throws()
    {
        float[] q = new float[12]; // 3 heads * headDim=4
        float[] k = new float[8];  // 2 KV heads * headDim=4
        float[] v = new float[8];
        float[] output = new float[12];

        // numHeads=3, numKvHeads=2 → 3 % 2 != 0
        Assert.Throws<ArgumentException>(() =>
            Attention.Execute(q, k, v, output, seqQ: 1, seqKv: 1,
                              numHeads: 3, numKvHeads: 2, headDim: 4, positionOffset: 0));
    }

    [Fact]
    public void ApplyCausalMask_ChunkedPrefill_CorrectBoundary()
    {
        // Second chunk of a prefill: positionOffset=2, seqQ=2.
        // Q token 0 is at absolute position 2 → can attend j ≤ 2 (not j=3,4)
        // Q token 1 is at absolute position 3 → can attend j ≤ 3 (not j=4)
        const int seqQ = 2, seqKv = 5, positionOffset = 2;
        float[] scores = new float[seqQ * seqKv];
        Array.Fill(scores, 1.0f);

        Attention.ApplyCausalMask(scores, seqQ, seqKv, positionOffset);

        // Row 0: visible j=0..2, masked j=3,4
        Assert.Equal(1.0f, scores[0 * seqKv + 2]);
        Assert.Equal(float.NegativeInfinity, scores[0 * seqKv + 3]);
        Assert.Equal(float.NegativeInfinity, scores[0 * seqKv + 4]);

        // Row 1: visible j=0..3, masked j=4
        Assert.Equal(1.0f, scores[1 * seqKv + 3]);
        Assert.Equal(float.NegativeInfinity, scores[1 * seqKv + 4]);
    }

    private static float[] RandomArray(Random rng, int length)
    {
        float[] arr = new float[length];
        for (int i = 0; i < length; i++)
            arr[i] = rng.NextSingle() * 2f - 1f;
        return arr;
    }
}
