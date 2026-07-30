using System.Runtime.InteropServices;
using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public sealed class EmbeddingLookupTests
{
    // No CPU kernel exists for embedding — scalar references written inline.
    // All three variants follow the same contract:
    //   output[t * hiddenSize .. +hiddenSize] = dequant(embedTable[tokenIds[t]])

    private const int Q8_0BlockSize  = 32;
    private const int Q8_0BlockBytes = 34;

    // ── CPU references ────────────────────────────────────────────────────────

    private static float[] CpuF32(float[] table, int[] ids, int hiddenSize)
    {
        float[] out_ = new float[ids.Length * hiddenSize];
        for (int t = 0; t < ids.Length; t++)
            Array.Copy(table, ids[t] * hiddenSize, out_, t * hiddenSize, hiddenSize);
        return out_;
    }

    private static float[] CpuF16(Half[] table, int[] ids, int hiddenSize)
    {
        float[] out_ = new float[ids.Length * hiddenSize];
        for (int t = 0; t < ids.Length; t++)
        for (int i = 0; i < hiddenSize; i++)
            out_[t * hiddenSize + i] = (float)table[ids[t] * hiddenSize + i];
        return out_;
    }

    private static float[] CpuQ8_0(byte[] table, int[] ids, int hiddenSize)
    {
        int blocksPerRow = hiddenSize / Q8_0BlockSize;
        float[] out_ = new float[ids.Length * hiddenSize];

        for (int t = 0; t < ids.Length; t++)
        {
            int rowOffset = ids[t] * blocksPerRow * Q8_0BlockBytes;
            for (int b = 0; b < blocksPerRow; b++)
            {
                int blockOffset = rowOffset + b * Q8_0BlockBytes;
                // First 2 bytes: half scale
                float d = (float)MemoryMarshal.Read<Half>(table.AsSpan(blockOffset, 2));
                // Next 32 bytes: signed int8 weights
                for (int j = 0; j < Q8_0BlockSize; j++)
                    out_[t * hiddenSize + b * Q8_0BlockSize + j] =
                        d * (float)(sbyte)table[blockOffset + 2 + j];
            }
        }
        return out_;
    }

    // ── F32 table ─────────────────────────────────────────────────────────────

    [MetalTestFact]
    public void F32_SingleToken_MatchesCpu()
    {
        const int vocabSize = 8, hiddenSize = 16, seqLen = 1;
        var rng = new Random(1);

        float[] table  = Rand(rng, vocabSize * hiddenSize);
        int[]   ids    = [3];
        float[] output = new float[seqLen * hiddenSize];

        float[] expected = CpuF32(table, ids, hiddenSize);

        using var ctx = new MetalContext();
        EmbeddingLookup.F32ToF32(ctx, table, ids, output, vocabSize, hiddenSize, seqLen);

        for (int i = 0; i < output.Length; i++) Assert.Equal(expected[i], output[i]);
    }

    [MetalTestFact]
    public void F32_MultipleTokens_MatchesCpu()
    {
        const int vocabSize = 16, hiddenSize = 8, seqLen = 4;
        var rng = new Random(2);

        float[] table  = Rand(rng, vocabSize * hiddenSize);
        int[]   ids    = [0, 5, 3, 15];
        float[] output = new float[seqLen * hiddenSize];

        float[] expected = CpuF32(table, ids, hiddenSize);

        using var ctx = new MetalContext();
        EmbeddingLookup.F32ToF32(ctx, table, ids, output, vocabSize, hiddenSize, seqLen);

        for (int i = 0; i < output.Length; i++) Assert.Equal(expected[i], output[i]);
    }

    [MetalTestFact]
    public void F32_RepeatedTokenId_ProducesSameRow()
    {
        // Looking up the same token twice must produce identical rows.
        const int vocabSize = 4, hiddenSize = 8, seqLen = 2;
        var rng = new Random(3);

        float[] table  = Rand(rng, vocabSize * hiddenSize);
        int[]   ids    = [2, 2];
        float[] output = new float[seqLen * hiddenSize];

        using var ctx = new MetalContext();
        EmbeddingLookup.F32ToF32(ctx, table, ids, output, vocabSize, hiddenSize, seqLen);

        for (int i = 0; i < hiddenSize; i++)
            Assert.Equal(output[i], output[hiddenSize + i]);
    }

    // ── F16 table ─────────────────────────────────────────────────────────────

    [MetalTestFact]
    public void F16_SingleToken_MatchesCpu()
    {
        const int vocabSize = 8, hiddenSize = 16, seqLen = 1;
        var rng = new Random(4);

        Half[]  table  = RandHalf(rng, vocabSize * hiddenSize);
        int[]   ids    = [7];
        float[] output = new float[seqLen * hiddenSize];

        float[] expected = CpuF16(table, ids, hiddenSize);

        using var ctx = new MetalContext();
        EmbeddingLookup.F16ToF32(ctx, table, ids, output, vocabSize, hiddenSize, seqLen);

        // f16 → f32 is lossless: exact comparison
        for (int i = 0; i < output.Length; i++) Assert.Equal(expected[i], output[i]);
    }

    [MetalTestFact]
    public void F16_MultipleTokens_MatchesCpu()
    {
        const int vocabSize = 16, hiddenSize = 8, seqLen = 4;
        var rng = new Random(5);

        Half[]  table  = RandHalf(rng, vocabSize * hiddenSize);
        int[]   ids    = [1, 4, 0, 12];
        float[] output = new float[seqLen * hiddenSize];

        float[] expected = CpuF16(table, ids, hiddenSize);

        using var ctx = new MetalContext();
        EmbeddingLookup.F16ToF32(ctx, table, ids, output, vocabSize, hiddenSize, seqLen);

        for (int i = 0; i < output.Length; i++) Assert.Equal(expected[i], output[i]);
    }

    // ── Q8_0 table ────────────────────────────────────────────────────────────

    [MetalTestFact]
    public void Q8_0_SingleToken_MatchesCpu()
    {
        const int vocabSize = 4, hiddenSize = 32, seqLen = 1;

        byte[] table  = BuildQ8_0Table(vocabSize, hiddenSize, seed: 10);
        int[]  ids    = [2];
        float[] output = new float[seqLen * hiddenSize];

        float[] expected = CpuQ8_0(table, ids, hiddenSize);

        using var ctx = new MetalContext();
        EmbeddingLookup.Q8_0ToF32(ctx, table, ids, output, vocabSize, hiddenSize, seqLen);

        for (int i = 0; i < output.Length; i++) Assert.Equal(expected[i], output[i], 1e-5f);
    }

    [MetalTestFact]
    public void Q8_0_MultipleTokens_MatchesCpu()
    {
        const int vocabSize = 8, hiddenSize = 64, seqLen = 3;

        byte[] table  = BuildQ8_0Table(vocabSize, hiddenSize, seed: 11);
        int[]  ids    = [0, 7, 3];
        float[] output = new float[seqLen * hiddenSize];

        float[] expected = CpuQ8_0(table, ids, hiddenSize);

        using var ctx = new MetalContext();
        EmbeddingLookup.Q8_0ToF32(ctx, table, ids, output, vocabSize, hiddenSize, seqLen);

        for (int i = 0; i < output.Length; i++) Assert.Equal(expected[i], output[i], 1e-5f);
    }

    [MetalTestFact]
    public void Q8_0_ZeroWeights_OutputIsZero()
    {
        // scale = 1.0, all int8 weights = 0 → output must be all zeros
        const int vocabSize = 2, hiddenSize = 32, seqLen = 1;
        int blocksPerRow = hiddenSize / Q8_0BlockSize;

        byte[] table = new byte[vocabSize * blocksPerRow * Q8_0BlockBytes];
        // Write scale = 1.0 (Half) at start of each block, leave weights as 0
        for (int v = 0; v < vocabSize; v++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (v * blocksPerRow + b) * Q8_0BlockBytes;
            MemoryMarshal.Write(table.AsSpan(off, 2), (Half)1.0f);
        }

        int[]   ids    = [0];
        float[] output = new float[seqLen * hiddenSize];

        using var ctx = new MetalContext();
        EmbeddingLookup.Q8_0ToF32(ctx, table, ids, output, vocabSize, hiddenSize, seqLen);

        for (int i = 0; i < output.Length; i++) Assert.Equal(0f, output[i]);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static float[] Rand(Random rng, int n)
        => Enumerable.Range(0, n).Select(_ => rng.NextSingle() * 2f - 1f).ToArray();

    private static Half[] RandHalf(Random rng, int n)
        => Enumerable.Range(0, n).Select(_ => (Half)(rng.NextSingle() * 2f - 1f)).ToArray();

    // Builds a Q8_0 table with deterministic data: each block gets a random
    // positive scale and random int8 weights in [-127, 127].
    private static byte[] BuildQ8_0Table(int vocabSize, int hiddenSize, int seed)
    {
        var rng = new Random(seed);
        int blocksPerRow = hiddenSize / Q8_0BlockSize;
        byte[] table = new byte[vocabSize * blocksPerRow * Q8_0BlockBytes];

        for (int v = 0; v < vocabSize; v++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (v * blocksPerRow + b) * Q8_0BlockBytes;
            Half scale = (Half)(rng.NextSingle() * 0.1f + 0.01f); // small positive scale
            MemoryMarshal.Write(table.AsSpan(off, 2), scale);
            for (int j = 0; j < Q8_0BlockSize; j++)
                table[off + 2 + j] = (byte)(sbyte)(rng.Next(-127, 128));
        }
        return table;
    }
}
