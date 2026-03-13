using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Verifies tiled attention (online softmax) produces identical results to the scalar reference.
/// The tiled path is activated when seqQ * seqKv * 4 > StackAllocThreshold (8192), i.e., seqQ * seqKv > 2048.
/// </summary>
public sealed class TiledAttentionTests
{
    private const float Tolerance = 1e-4f;

    [Fact]
    public void TiledMatchesScalar_ShortSequence()
    {
        // seqQ=4, seqKv=4, headDim=16 — small but tests basic correctness
        // scoreSize = 16 * 4 = 64 bytes → below threshold, but we test via Execute which may use naive.
        // Use larger seqKv to force tiled path: seqQ=4, seqKv=600 → 4*600*4 = 9600 > 8192.
        const int seqQ = 4, seqKv = 600, numHeads = 2, numKvHeads = 2, headDim = 16;
        var rng = new Random(42);

        float[] q = RandomArray(rng, seqQ * numHeads * headDim);
        float[] k = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] v = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] outputTiled = new float[seqQ * numHeads * headDim];
        float[] outputScalar = new float[seqQ * numHeads * headDim];

        Attention.Execute(q, k, v, outputTiled, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: 596);
        Attention.ExecuteScalar(q, k, v, outputScalar, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: 596);

        AssertSpansEqual(outputScalar, outputTiled);
    }

    [Fact]
    public void TiledMatchesScalar_LongSequence_Decode()
    {
        // Decode: seqQ=1, seqKv=2100, headDim=64 — crosses tile boundaries
        // scoreSize = 1 * 2100 * 4 = 8400 > 8192 → tiled path
        const int seqQ = 1, seqKv = 2100, numHeads = 4, numKvHeads = 4, headDim = 64;
        var rng = new Random(42);

        float[] q = RandomArray(rng, seqQ * numHeads * headDim);
        float[] k = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] v = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] outputTiled = new float[seqQ * numHeads * headDim];
        float[] outputScalar = new float[seqQ * numHeads * headDim];

        Attention.Execute(q, k, v, outputTiled, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: seqKv - 1);
        Attention.ExecuteScalar(q, k, v, outputScalar, seqQ, seqKv, numHeads, numKvHeads, headDim,
                                positionOffset: seqKv - 1);

        AssertSpansEqual(outputScalar, outputTiled);
    }

    [Fact]
    public void TiledMatchesScalar_Prefill()
    {
        // Prefill: seqQ=seqKv=256, headDim=64 — causal triangular across tiles
        // scoreSize = 256 * 256 * 4 = 262144 > 8192 → tiled path
        const int seqLen = 256, numHeads = 2, numKvHeads = 2, headDim = 64;
        var rng = new Random(42);

        float[] q = RandomArray(rng, seqLen * numHeads * headDim);
        float[] k = RandomArray(rng, seqLen * numKvHeads * headDim);
        float[] v = RandomArray(rng, seqLen * numKvHeads * headDim);
        float[] outputTiled = new float[seqLen * numHeads * headDim];
        float[] outputScalar = new float[seqLen * numHeads * headDim];

        Attention.Execute(q, k, v, outputTiled, seqLen, seqLen, numHeads, numKvHeads, headDim, positionOffset: 0);
        Attention.ExecuteScalar(q, k, v, outputScalar, seqLen, seqLen, numHeads, numKvHeads, headDim, positionOffset: 0);

        AssertSpansEqual(outputScalar, outputTiled);
    }

    [Fact]
    public void TiledMatchesScalar_GQA()
    {
        // GQA: numHeads=8, numKvHeads=2 → groupSize=4
        // seqQ=1, seqKv=2100 → tiled path
        const int seqQ = 1, seqKv = 2100, numHeads = 8, numKvHeads = 2, headDim = 64;
        var rng = new Random(42);

        float[] q = RandomArray(rng, seqQ * numHeads * headDim);
        float[] k = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] v = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] outputTiled = new float[seqQ * numHeads * headDim];
        float[] outputScalar = new float[seqQ * numHeads * headDim];

        Attention.Execute(q, k, v, outputTiled, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: seqKv - 1);
        Attention.ExecuteScalar(q, k, v, outputScalar, seqQ, seqKv, numHeads, numKvHeads, headDim,
                                positionOffset: seqKv - 1);

        AssertSpansEqual(outputScalar, outputTiled);
    }

    [Fact]
    public void TiledMatchesScalar_SlidingWindow()
    {
        // Sliding window: window=128, seqKv=512 — window masking across tile boundaries
        // seqQ=1, scoreSize = 512 * 4 = 2048 → borderline, use seqQ=8 → 8*512*4 = 16384 > 8192
        const int seqQ = 8, seqKv = 512, numHeads = 2, numKvHeads = 2, headDim = 64;
        const int slidingWindow = 128;
        var rng = new Random(42);

        float[] q = RandomArray(rng, seqQ * numHeads * headDim);
        float[] k = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] v = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] outputTiled = new float[seqQ * numHeads * headDim];
        float[] outputScalar = new float[seqQ * numHeads * headDim];

        Attention.Execute(q, k, v, outputTiled, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: seqKv - seqQ, slidingWindowSize: slidingWindow);
        Attention.ExecuteScalar(q, k, v, outputScalar, seqQ, seqKv, numHeads, numKvHeads, headDim,
                                positionOffset: seqKv - seqQ, slidingWindowSize: slidingWindow);

        AssertSpansEqual(outputScalar, outputTiled);
    }

    [Fact]
    public void TiledMatchesScalar_DecodeWithOffset()
    {
        // Decode at position 1023: seqQ=1, seqKv=1024, positionOffset=1023
        // scoreSize = 1024 * 4 = 4096 → below threshold (naive), but let's use seqKv=2100 to force tiled
        const int seqQ = 1, seqKv = 2100, numHeads = 4, numKvHeads = 4, headDim = 64;
        const int positionOffset = 2099;
        var rng = new Random(42);

        float[] q = RandomArray(rng, seqQ * numHeads * headDim);
        float[] k = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] v = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] outputTiled = new float[seqQ * numHeads * headDim];
        float[] outputScalar = new float[seqQ * numHeads * headDim];

        Attention.Execute(q, k, v, outputTiled, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: positionOffset);
        Attention.ExecuteScalar(q, k, v, outputScalar, seqQ, seqKv, numHeads, numKvHeads, headDim,
                                positionOffset: positionOffset);

        AssertSpansEqual(outputScalar, outputTiled);
    }

    [Fact]
    public void OnlineSoftmax_NumericallyStable()
    {
        // Large Q/K values (±100) — no NaN/Inf in output
        const int seqQ = 1, seqKv = 2100, numHeads = 2, numKvHeads = 2, headDim = 64;
        var rng = new Random(42);

        float[] q = RandomArray(rng, seqQ * numHeads * headDim, scale: 100f);
        float[] k = RandomArray(rng, seqKv * numKvHeads * headDim, scale: 100f);
        float[] v = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] output = new float[seqQ * numHeads * headDim];

        Attention.Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                          positionOffset: seqKv - 1);

        Assert.All(output, val => Assert.True(float.IsFinite(val), $"Non-finite value: {val}"));
    }

    [Fact]
    public void CausalMask_AcrossTileBoundary()
    {
        // Choose seqKv so causal boundary falls mid-tile.
        // ComputeTileSize(64) = 256. Prefill: seqQ=seqKv=300.
        // For Q row i=250, visibleEnd=251 which is < 256 (tile 0 covers 0..255).
        // For Q row i=260, visibleEnd=261 which crosses into tile 1 (256..299).
        const int seqLen = 300, numHeads = 1, numKvHeads = 1, headDim = 64;
        var rng = new Random(42);

        float[] q = RandomArray(rng, seqLen * numHeads * headDim);
        float[] k = RandomArray(rng, seqLen * numKvHeads * headDim);
        float[] v = RandomArray(rng, seqLen * numKvHeads * headDim);
        float[] outputTiled = new float[seqLen * numHeads * headDim];
        float[] outputScalar = new float[seqLen * numHeads * headDim];

        Attention.Execute(q, k, v, outputTiled, seqLen, seqLen, numHeads, numKvHeads, headDim, positionOffset: 0);
        Attention.ExecuteScalar(q, k, v, outputScalar, seqLen, seqLen, numHeads, numKvHeads, headDim, positionOffset: 0);

        AssertSpansEqual(outputScalar, outputTiled);
    }

    [Fact]
    public unsafe void Parallel_TiledMatchesSingleThreaded()
    {
        // Pointer overload with pool should produce same results as span-based (which uses tiled internally)
        const int seqQ = 1, seqKv = 2100, numHeads = 8, numKvHeads = 2, headDim = 64;
        var rng = new Random(42);

        int qSize = seqQ * numHeads * headDim;
        int kvSize = seqKv * numKvHeads * headDim;

        float* qPtr = (float*)NativeMemory.AlignedAlloc((nuint)(qSize * sizeof(float)), 64);
        float* kPtr = (float*)NativeMemory.AlignedAlloc((nuint)(kvSize * sizeof(float)), 64);
        float* vPtr = (float*)NativeMemory.AlignedAlloc((nuint)(kvSize * sizeof(float)), 64);
        float* outST = (float*)NativeMemory.AlignedAlloc((nuint)(qSize * sizeof(float)), 64);
        float* outMT = (float*)NativeMemory.AlignedAlloc((nuint)(qSize * sizeof(float)), 64);

        using var pool = new ComputeThreadPool(4);

        try
        {
            FillRandom(qPtr, qSize, rng);
            FillRandom(kPtr, kvSize, rng);
            FillRandom(vPtr, kvSize, rng);

            int positionOffset = seqKv - 1;

            // Single-threaded (span-based, uses tiled internally)
            Attention.Execute(
                new ReadOnlySpan<float>(qPtr, qSize),
                new ReadOnlySpan<float>(kPtr, kvSize),
                new ReadOnlySpan<float>(vPtr, kvSize),
                new Span<float>(outST, qSize),
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);

            // Multi-threaded (pointer-based, uses tiled worker)
            Attention.Execute(qPtr, kPtr, vPtr, outMT,
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, pool);

            for (int i = 0; i < qSize; i++)
            {
                Assert.True(
                    MathF.Abs(outST[i] - outMT[i]) < Tolerance,
                    $"Mismatch at index {i}: expected {outST[i]:G9}, actual {outMT[i]:G9}");
            }
        }
        finally
        {
            NativeMemory.AlignedFree(qPtr);
            NativeMemory.AlignedFree(kPtr);
            NativeMemory.AlignedFree(vPtr);
            NativeMemory.AlignedFree(outST);
            NativeMemory.AlignedFree(outMT);
        }
    }

    [Fact]
    public void NaivePath_StillUsed_ForSmallSequences()
    {
        // Verify that small sequences still produce correct results (naive path)
        // seqQ=2, seqKv=4, scoreSize = 8 * 4 = 32 < 8192 → naive path
        const int seqQ = 2, seqKv = 4, numHeads = 2, numKvHeads = 2, headDim = 16;
        var rng = new Random(42);

        float[] q = RandomArray(rng, seqQ * numHeads * headDim);
        float[] k = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] v = RandomArray(rng, seqKv * numKvHeads * headDim);
        float[] outputExec = new float[seqQ * numHeads * headDim];
        float[] outputScalar = new float[seqQ * numHeads * headDim];

        Attention.Execute(q, k, v, outputExec, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: 2);
        Attention.ExecuteScalar(q, k, v, outputScalar, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: 2);

        AssertSpansEqual(outputScalar, outputExec);
    }

    private static void AssertSpansEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(
                MathF.Abs(expected[i] - actual[i]) < Tolerance,
                $"Mismatch at index {i}: expected {expected[i]:G9}, actual {actual[i]:G9}");
    }

    private static float[] RandomArray(Random rng, int length, float scale = 1f)
    {
        float[] arr = new float[length];
        for (int i = 0; i < length; i++)
            arr[i] = (rng.NextSingle() * 2f - 1f) * scale;
        return arr;
    }

    private static unsafe void FillRandom(float* ptr, int count, Random rng)
    {
        for (int i = 0; i < count; i++)
            ptr[i] = rng.NextSingle() * 2f - 1f;
    }
}
