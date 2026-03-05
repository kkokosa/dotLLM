using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Verifies that head-parallel attention produces identical results to the single-threaded span-based path.
/// </summary>
public sealed unsafe class AttentionParallelTests : IDisposable
{
    private readonly ComputeThreadPool _pool = new(4);

    public void Dispose() => _pool.Dispose();

    [Fact]
    public void Execute_SingleHead_Parallel_MatchesSingleThreaded()
    {
        const int headDim = 4;
        float[] q = [1f, 0f, 0f, 0f];
        float[] k = [1f, 0f, 0f, 0f];
        float[] v = [1f, 0f, 0f, 0f];
        float[] outputST = new float[headDim];
        float* outputMT = (float*)NativeMemory.AlignedAlloc((nuint)(headDim * sizeof(float)), 64);

        try
        {
            Attention.Execute(q, k, v, outputST, seqQ: 1, seqKv: 1,
                numHeads: 1, numKvHeads: 1, headDim: headDim, positionOffset: 0);

            fixed (float* qp = q, kp = k, vp = v)
            {
                // numHeads=1 < 2, so falls back to single-threaded — verify it still produces correct results
                Attention.Execute(qp, kp, vp, outputMT, seqQ: 1, seqKv: 1,
                    numHeads: 1, numKvHeads: 1, headDim: headDim, positionOffset: 0, _pool);
            }

            Assert.Equal(outputST[0], outputMT[0], 1e-5f);
            Assert.Equal(outputST[1], outputMT[1], 1e-5f);
        }
        finally
        {
            NativeMemory.AlignedFree(outputMT);
        }
    }

    [Theory]
    [InlineData(4, 4, 64, 1, 4)]    // MHA: 4 heads, 1 KV head (GQA), seqLen=1, headDim=64 (decode)
    [InlineData(8, 2, 32, 4, 4)]     // GQA: 8 heads, 2 KV heads, seqLen=4 (prefill), headDim=32
    [InlineData(16, 16, 64, 1, 16)]  // MHA: 16 heads, seqLen=1, headDim=64
    [InlineData(8, 8, 64, 8, 8)]     // MHA: prefill 8 tokens, 8 heads (with causal masking)
    public void Execute_MultiHead_Parallel_MatchesSingleThreaded(
        int numHeads, int numKvHeads, int headDim, int seqLen, int seqKv)
    {
        var rng = new Random(42);
        int qSize = seqLen * numHeads * headDim;
        int kvSize = seqKv * numKvHeads * headDim;
        int outSize = seqLen * numHeads * headDim;

        float* qPtr = (float*)NativeMemory.AlignedAlloc((nuint)(qSize * sizeof(float)), 64);
        float* kPtr = (float*)NativeMemory.AlignedAlloc((nuint)(kvSize * sizeof(float)), 64);
        float* vPtr = (float*)NativeMemory.AlignedAlloc((nuint)(kvSize * sizeof(float)), 64);
        float* outST = (float*)NativeMemory.AlignedAlloc((nuint)(outSize * sizeof(float)), 64);
        float* outMT = (float*)NativeMemory.AlignedAlloc((nuint)(outSize * sizeof(float)), 64);

        try
        {
            FillRandom(qPtr, qSize, rng);
            FillRandom(kPtr, kvSize, rng);
            FillRandom(vPtr, kvSize, rng);

            int positionOffset = seqKv - seqLen; // for decode, cache tokens precede query

            // Single-threaded (span-based)
            Attention.Execute(
                new ReadOnlySpan<float>(qPtr, qSize),
                new ReadOnlySpan<float>(kPtr, kvSize),
                new ReadOnlySpan<float>(vPtr, kvSize),
                new Span<float>(outST, outSize),
                seqLen, seqKv, numHeads, numKvHeads, headDim, positionOffset);

            // Multi-threaded (pointer-based)
            Attention.Execute(qPtr, kPtr, vPtr, outMT,
                seqLen, seqKv, numHeads, numKvHeads, headDim, positionOffset, _pool);

            // Verify bit-identical
            for (int i = 0; i < outSize; i++)
            {
                Assert.True(
                    BitConverter.SingleToInt32Bits(outST[i]) == BitConverter.SingleToInt32Bits(outMT[i]),
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
    public void Execute_NullPool_FallsBackToSingleThreaded()
    {
        var rng = new Random(42);
        const int numHeads = 4, numKvHeads = 4, headDim = 32, seqLen = 2, seqKv = 2;
        int qSize = seqLen * numHeads * headDim;
        int kvSize = seqKv * numKvHeads * headDim;
        int outSize = seqLen * numHeads * headDim;

        float* qPtr = (float*)NativeMemory.AlignedAlloc((nuint)(qSize * sizeof(float)), 64);
        float* kPtr = (float*)NativeMemory.AlignedAlloc((nuint)(kvSize * sizeof(float)), 64);
        float* vPtr = (float*)NativeMemory.AlignedAlloc((nuint)(kvSize * sizeof(float)), 64);
        float* outST = (float*)NativeMemory.AlignedAlloc((nuint)(outSize * sizeof(float)), 64);
        float* outNull = (float*)NativeMemory.AlignedAlloc((nuint)(outSize * sizeof(float)), 64);

        try
        {
            FillRandom(qPtr, qSize, rng);
            FillRandom(kPtr, kvSize, rng);
            FillRandom(vPtr, kvSize, rng);

            Attention.Execute(
                new ReadOnlySpan<float>(qPtr, qSize),
                new ReadOnlySpan<float>(kPtr, kvSize),
                new ReadOnlySpan<float>(vPtr, kvSize),
                new Span<float>(outST, outSize),
                seqLen, seqKv, numHeads, numKvHeads, headDim, 0);

            Attention.Execute(qPtr, kPtr, vPtr, outNull,
                seqLen, seqKv, numHeads, numKvHeads, headDim, 0, null);

            for (int i = 0; i < outSize; i++)
            {
                Assert.True(
                    BitConverter.SingleToInt32Bits(outST[i]) == BitConverter.SingleToInt32Bits(outNull[i]),
                    $"Mismatch at index {i}");
            }
        }
        finally
        {
            NativeMemory.AlignedFree(qPtr);
            NativeMemory.AlignedFree(kPtr);
            NativeMemory.AlignedFree(vPtr);
            NativeMemory.AlignedFree(outST);
            NativeMemory.AlignedFree(outNull);
        }
    }

    private static void FillRandom(float* ptr, int count, Random rng)
    {
        for (int i = 0; i < count; i++)
            ptr[i] = rng.NextSingle() * 2f - 1f;
    }
}
