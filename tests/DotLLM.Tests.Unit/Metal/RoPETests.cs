using DotLLM.Core.Configuration;
using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

// Shared tolerance for FP16 RoPE: one rotation with FP16 I/O → ~0.2 % relative error.

namespace DotLLM.Tests.Unit.Metal;

public sealed class RoPETests
{
    // ── Norm variant (Llama/Mistral) ─────────────────────────────────────────

    [MetalTestFact]
    public void NormVariant_MatchesCpu()
    {
        const int seqLen = 4, numHeads = 2, numKvHeads = 2, headDim = 8;
        const float theta = 10000f;
        var rng = new Random(42);
        int halfDim = headDim / 2;

        float[] cosTable = new float[seqLen * halfDim];
        float[] sinTable = new float[seqLen * halfDim];
        DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(seqLen, headDim, theta, cosTable, sinTable);

        float[] q = new float[seqLen * numHeads   * headDim];
        float[] k = new float[seqLen * numKvHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < k.Length; i++) k[i] = rng.NextSingle() * 2f - 1f;
        int[] positions = [0, 1, 2, 3];

        float[] qExpected = (float[])q.Clone();
        float[] kExpected = (float[])k.Clone();
        DotLLM.Cpu.Kernels.RoPE.Execute(qExpected, kExpected, positions,
            numHeads, numKvHeads, headDim, cosTable, sinTable);

        using var ctx = new MetalContext();
        RoPE.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, theta);

        for (int i = 0; i < q.Length; i++) Assert.Equal(qExpected[i], q[i], 1e-4f);
        for (int i = 0; i < k.Length; i++) Assert.Equal(kExpected[i], k[i], 1e-4f);
    }

    // ── NeoX variant (Qwen/Phi) ──────────────────────────────────────────────

    [MetalTestFact]
    public void NeoXVariant_MatchesCpu()
    {
        const int seqLen = 4, numHeads = 2, numKvHeads = 2, headDim = 8;
        const float theta = 10000f;
        var rng = new Random(99);
        int halfDim = headDim / 2;

        float[] cosTable = new float[seqLen * halfDim];
        float[] sinTable = new float[seqLen * halfDim];
        DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(seqLen, headDim, theta, cosTable, sinTable);

        float[] q = new float[seqLen * numHeads   * headDim];
        float[] k = new float[seqLen * numKvHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < k.Length; i++) k[i] = rng.NextSingle() * 2f - 1f;
        int[] positions = [0, 1, 2, 3];

        float[] qExpected = (float[])q.Clone();
        float[] kExpected = (float[])k.Clone();
        DotLLM.Cpu.Kernels.RoPE.Execute(qExpected, kExpected, positions,
            numHeads, numKvHeads, headDim, cosTable, sinTable, RoPEType.NeoX);

        using var ctx = new MetalContext();
        RoPE.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, theta, RoPEType.NeoX);

        for (int i = 0; i < q.Length; i++) Assert.Equal(qExpected[i], q[i], 1e-4f);
        for (int i = 0; i < k.Length; i++) Assert.Equal(kExpected[i], k[i], 1e-4f);
    }

    // ── Position 0 = identité ────────────────────────────────────────────────

    [MetalTestFact]
    public void PositionZero_VecUnchanged()
    {
        // cos(0) = 1, sin(0) = 0 → rotation = identité
        const int seqLen = 1, numHeads = 1, numKvHeads = 1, headDim = 4;

        float[] q = [1f, 2f, 3f, 4f];
        float[] k = [5f, 6f, 7f, 8f];
        float[] qOrig = (float[])q.Clone();
        float[] kOrig = (float[])k.Clone();
        int[] positions = [0];

        using var ctx = new MetalContext();
        RoPE.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, 10000f);

        for (int i = 0; i < q.Length; i++) Assert.Equal(qOrig[i], q[i], 1e-5f);
        for (int i = 0; i < k.Length; i++) Assert.Equal(kOrig[i], k[i], 1e-5f);
    }

    // ── GQA : numKvHeads < numHeads ──────────────────────────────────────────

    [MetalTestFact]
    public void GQA_KvHeadsRotatedCorrectly()
    {
        const int seqLen = 2, numHeads = 4, numKvHeads = 1, headDim = 8;
        const float theta = 10000f;
        var rng = new Random(7);
        int halfDim = headDim / 2;

        float[] cosTable = new float[seqLen * halfDim];
        float[] sinTable = new float[seqLen * halfDim];
        DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(seqLen, headDim, theta, cosTable, sinTable);

        float[] q = new float[seqLen * numHeads   * headDim];
        float[] k = new float[seqLen * numKvHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < k.Length; i++) k[i] = rng.NextSingle() * 2f - 1f;
        int[] positions = [0, 1];

        float[] qExpected = (float[])q.Clone();
        float[] kExpected = (float[])k.Clone();
        DotLLM.Cpu.Kernels.RoPE.Execute(qExpected, kExpected, positions,
            numHeads, numKvHeads, headDim, cosTable, sinTable);

        using var ctx = new MetalContext();
        RoPE.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, theta);

        for (int i = 0; i < q.Length; i++) Assert.Equal(qExpected[i], q[i], 1e-4f);
        for (int i = 0; i < k.Length; i++) Assert.Equal(kExpected[i], k[i], 1e-4f);
    }

    // ── ropeDim partiel ──────────────────────────────────────────────────────

    [MetalTestFact]
    public void PartialRopeDim_MatchesCpu()
    {
        const int seqLen = 2, numHeads = 2, numKvHeads = 1, headDim = 8, ropeDim = 4;
        const float theta = 10000f;
        var rng = new Random(13);
        int halfRopeDim = ropeDim / 2;

        float[] cosTable = new float[seqLen * halfRopeDim];
        float[] sinTable = new float[seqLen * halfRopeDim];
        DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(seqLen, ropeDim, theta, cosTable, sinTable);

        float[] q = new float[seqLen * numHeads   * headDim];
        float[] k = new float[seqLen * numKvHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < k.Length; i++) k[i] = rng.NextSingle() * 2f - 1f;
        int[] positions = [0, 1];

        float[] qExpected = (float[])q.Clone();
        float[] kExpected = (float[])k.Clone();
        DotLLM.Cpu.Kernels.RoPE.Execute(qExpected, kExpected, positions,
            numHeads, numKvHeads, headDim, ropeDim, cosTable, sinTable);

        using var ctx = new MetalContext();
        RoPE.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, ropeDim, theta);

        for (int i = 0; i < q.Length; i++) Assert.Equal(qExpected[i], q[i], 1e-4f);
        for (int i = 0; i < k.Length; i++) Assert.Equal(kExpected[i], k[i], 1e-4f);
    }

    // ── Large input (Llama-like) ─────────────────────────────────────────────

    [MetalTestFact]
    public void LargeInput_MatchesCpu()
    {
        const int seqLen = 8, numHeads = 32, numKvHeads = 8, headDim = 128;
        const float theta = 10000f;
        var rng = new Random(123);
        int halfDim = headDim / 2;

        float[] cosTable = new float[seqLen * halfDim];
        float[] sinTable = new float[seqLen * halfDim];
        DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(seqLen, headDim, theta, cosTable, sinTable);

        float[] q = new float[seqLen * numHeads   * headDim];
        float[] k = new float[seqLen * numKvHeads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < k.Length; i++) k[i] = rng.NextSingle() * 2f - 1f;
        int[] positions = [0, 1, 2, 3, 4, 5, 6, 7];

        float[] qExpected = (float[])q.Clone();
        float[] kExpected = (float[])k.Clone();
        DotLLM.Cpu.Kernels.RoPE.Execute(qExpected, kExpected, positions,
            numHeads, numKvHeads, headDim, cosTable, sinTable);

        using var ctx = new MetalContext();
        RoPE.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, theta);

        for (int i = 0; i < q.Length; i++) Assert.Equal(qExpected[i], q[i], 1e-4f);
        for (int i = 0; i < k.Length; i++) Assert.Equal(kExpected[i], k[i], 1e-4f);
    }
}

// ── RoPE FP16 ─────────────────────────────────────────────────────────────────

public sealed class RoPEF16Tests
{
    // FP16 I/O, FP32 computation: one rotation per element pair → ~0.2 % relative error.
    private const float Tol = 1e-2f;

    /// <summary>
    /// CPU reference: Half→float, run CPU RoPE, compare float vs (float)gpu_half.
    /// </summary>
    private static (float[] qRef, float[] kRef) CpuRef(
        Half[] q, Half[] k, int[] positions,
        int numHeads, int numKvHeads, int headDim,
        int ropeDim, float theta, RoPEType ropeType)
    {
        float[] cosTable = new float[positions.Length * (ropeDim / 2)];
        float[] sinTable = new float[positions.Length * (ropeDim / 2)];
        DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(
            positions.Length, ropeDim, theta, cosTable, sinTable);

        float[] qF = Array.ConvertAll(q, h => (float)h);
        float[] kF = Array.ConvertAll(k, h => (float)h);
        DotLLM.Cpu.Kernels.RoPE.Execute(
            qF, kF, positions, numHeads, numKvHeads, headDim, ropeDim, cosTable, sinTable, ropeType);
        return (qF, kF);
    }

    private static void AssertClose(float[] expected, Half[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], (float)actual[i], Tol);
    }

    // ── Norm variant (Llama/Mistral) ─────────────────────────────────────────

    [MetalTestFact]
    public void NormVariant_MatchesCpu()
    {
        const int seqLen = 4, numHeads = 2, numKvHeads = 2, headDim = 8;
        const float theta = 10000f;
        var rng = new Random(42);

        Half[] q = new Half[seqLen * numHeads    * headDim];
        Half[] k = new Half[seqLen * numKvHeads  * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < k.Length; i++) k[i] = (Half)(rng.NextSingle() * 2f - 1f);
        int[] positions = [0, 1, 2, 3];

        var (qRef, kRef) = CpuRef(q, k, positions, numHeads, numKvHeads, headDim, headDim, theta, RoPEType.Norm);

        using var ctx = new MetalContext();
        RoPEF16.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, theta);

        AssertClose(qRef, q);
        AssertClose(kRef, k);
    }

    // ── NeoX variant (Qwen/Phi) ──────────────────────────────────────────────

    [MetalTestFact]
    public void NeoXVariant_MatchesCpu()
    {
        const int seqLen = 4, numHeads = 2, numKvHeads = 2, headDim = 8;
        const float theta = 10000f;
        var rng = new Random(99);

        Half[] q = new Half[seqLen * numHeads    * headDim];
        Half[] k = new Half[seqLen * numKvHeads  * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < k.Length; i++) k[i] = (Half)(rng.NextSingle() * 2f - 1f);
        int[] positions = [0, 1, 2, 3];

        var (qRef, kRef) = CpuRef(q, k, positions, numHeads, numKvHeads, headDim, headDim, theta, RoPEType.NeoX);

        using var ctx = new MetalContext();
        RoPEF16.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, theta, RoPEType.NeoX);

        AssertClose(qRef, q);
        AssertClose(kRef, k);
    }

    // ── Position 0 = identité ────────────────────────────────────────────────

    [MetalTestFact]
    public void PositionZero_VecUnchanged()
    {
        // cos(0) = 1, sin(0) = 0 → rotation = identité
        const int seqLen = 1, numHeads = 1, numKvHeads = 1, headDim = 4;

        Half[] q = [(Half)1f, (Half)2f, (Half)3f, (Half)4f];
        Half[] k = [(Half)5f, (Half)6f, (Half)7f, (Half)8f];
        Half[] qOrig = (Half[])q.Clone();
        Half[] kOrig = (Half[])k.Clone();
        int[] positions = [0];

        using var ctx = new MetalContext();
        RoPEF16.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, 10000f);

        for (int i = 0; i < q.Length; i++) Assert.Equal((float)qOrig[i], (float)q[i], 1e-3f);
        for (int i = 0; i < k.Length; i++) Assert.Equal((float)kOrig[i], (float)k[i], 1e-3f);
    }

    // ── GQA : numKvHeads < numHeads ──────────────────────────────────────────

    [MetalTestFact]
    public void GQA_KvHeadsRotatedCorrectly()
    {
        const int seqLen = 2, numHeads = 4, numKvHeads = 1, headDim = 8;
        const float theta = 10000f;
        var rng = new Random(7);

        Half[] q = new Half[seqLen * numHeads    * headDim];
        Half[] k = new Half[seqLen * numKvHeads  * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < k.Length; i++) k[i] = (Half)(rng.NextSingle() * 2f - 1f);
        int[] positions = [0, 1];

        var (qRef, kRef) = CpuRef(q, k, positions, numHeads, numKvHeads, headDim, headDim, theta, RoPEType.Norm);

        using var ctx = new MetalContext();
        RoPEF16.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, theta);

        AssertClose(qRef, q);
        AssertClose(kRef, k);
    }

    // ── ropeDim partiel ──────────────────────────────────────────────────────

    [MetalTestFact]
    public void PartialRopeDim_MatchesCpu()
    {
        const int seqLen = 2, numHeads = 2, numKvHeads = 1, headDim = 8, ropeDim = 4;
        const float theta = 10000f;
        var rng = new Random(13);

        Half[] q = new Half[seqLen * numHeads    * headDim];
        Half[] k = new Half[seqLen * numKvHeads  * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < k.Length; i++) k[i] = (Half)(rng.NextSingle() * 2f - 1f);
        int[] positions = [0, 1];

        var (qRef, kRef) = CpuRef(q, k, positions, numHeads, numKvHeads, headDim, ropeDim, theta, RoPEType.Norm);

        using var ctx = new MetalContext();
        RoPEF16.Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, ropeDim, theta);

        AssertClose(qRef, q);
        AssertClose(kRef, k);
    }
}
