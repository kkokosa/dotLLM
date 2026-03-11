using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Verifies that fused decode dispatch produces bit-identical results to individual GEMV calls.
/// Each test runs the fused API and the individual parallel GEMV, then asserts bit-exact match.
/// </summary>
public sealed unsafe class MatMulFusedDecodeTests : IDisposable
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int Q5_0BlockBytes = 22;
    private const int Q4_K_BlockBytes = 144;
    private const int Q5_K_BlockBytes = 176;
    private const int Q6_K_BlockBytes = 210;
    private const int KQuantGroupSize = 256;

    private readonly ComputeThreadPool _pool = new(4);

    public void Dispose() => _pool.Dispose();

    // ──────────────────── FusedDecodeGemv3 Tests ────────────────────

    [Theory]
    [InlineData(256, 256)]
    [InlineData(128, 512)]
    [InlineData(64, 128)]
    public void FusedDecodeGemv3_Q8_0_MatchesIndividual(int m, int k)
    {
        RunFusedDecode3Test(m, m, m, k, QuantizationType.Q8_0, QuantizationType.Q8_0, QuantizationType.Q8_0);
    }

    [Theory]
    [InlineData(256, 256)]
    [InlineData(128, 512)]
    public void FusedDecodeGemv3_Q5_0_MatchesIndividual(int m, int k)
    {
        RunFusedDecode3Test(m, m, m, k, QuantizationType.Q5_0, QuantizationType.Q5_0, QuantizationType.Q5_0);
    }

    [Theory]
    [InlineData(256, 256)]
    [InlineData(128, 512)]
    public void FusedDecodeGemv3_Q4_K_MatchesIndividual(int m, int k)
    {
        RunFusedDecode3Test(m, m, m, k, QuantizationType.Q4_K, QuantizationType.Q4_K, QuantizationType.Q4_K);
    }

    [Theory]
    [InlineData(256, 256)]
    public void FusedDecodeGemv3_MixedQ8_Q5_CrossFamily_MatchesIndividual(int m, int k)
    {
        // Q=Q8_0, K/V=Q5_0 — cross-family (Q8_0 uses Q8_0 input, Q5_0 uses Q8_1 input)
        RunFusedDecode3Test(m, m, m, k, QuantizationType.Q8_0, QuantizationType.Q5_0, QuantizationType.Q5_0);
    }

    [Theory]
    [InlineData(256, 256)]
    public void FusedDecodeGemv3_MixedKQuantFamily_MatchesIndividual(int m, int k)
    {
        // Q4_K + Q5_K + Q6_K — same K-quant family
        RunFusedDecode3Test(m, m, m, k, QuantizationType.Q4_K, QuantizationType.Q5_K, QuantizationType.Q6_K);
    }

    [Theory]
    [InlineData(256, 64, 256)]
    [InlineData(512, 128, 256)]
    public void FusedDecodeGemv3_DifferentOutputDims_MatchesIndividual(int m0, int m1, int k)
    {
        // GQA style: Q has more output rows than K/V
        RunFusedDecode3Test(m0, m1, m1, k, QuantizationType.Q8_0, QuantizationType.Q8_0, QuantizationType.Q8_0);
    }

    [Theory]
    [InlineData(256, 256)]
    public void FusedDecodeGemv3_CrossFamily_MatchesIndividual(int m, int k)
    {
        // Q=Q8_0, K/V=Q4_K — cross-family, dispatches separately
        RunFusedDecode3Test(m, m, m, k, QuantizationType.Q8_0, QuantizationType.Q4_K, QuantizationType.Q4_K);
    }

    // ──────────────────── FusedDecodeGemv2 Tests ────────────────────

    [Theory]
    [InlineData(256, 256)]
    [InlineData(128, 512)]
    public void FusedDecodeGemv2_Q8_0_MatchesIndividual(int m, int k)
    {
        RunFusedDecode2Test(m, m, k, QuantizationType.Q8_0, QuantizationType.Q8_0);
    }

    [Theory]
    [InlineData(256, 256)]
    [InlineData(128, 512)]
    public void FusedDecodeGemv2_Q4_K_MatchesIndividual(int m, int k)
    {
        RunFusedDecode2Test(m, m, k, QuantizationType.Q4_K, QuantizationType.Q4_K);
    }

    [Theory]
    [InlineData(256, 256)]
    public void FusedDecodeGemv2_MixedFamily_MatchesIndividual(int m, int k)
    {
        // Gate=Q8_0, Up=Q4_K — cross-family, dispatches separately
        RunFusedDecode2Test(m, m, k, QuantizationType.Q8_0, QuantizationType.Q4_K);
    }

    // ──────────────────── Test runners ────────────────────

    private void RunFusedDecode3Test(int m0, int m1, int m2, int k,
        QuantizationType qt0, QuantizationType qt1, QuantizationType qt2)
    {
        var rng = new Random(42);

        // Allocate weights
        byte* w0 = AllocWeights(m0, k, qt0, rng);
        byte* w1 = AllocWeights(m1, k, qt1, rng);
        byte* w2 = AllocWeights(m2, k, qt2, rng);

        // Allocate input
        float* input = (float*)NativeMemory.AlignedAlloc((nuint)(k * sizeof(float)), 64);
        FillFloats(input, k, rng);

        // Allocate result buffers: individual vs fused
        float* r0Ind = (float*)NativeMemory.AlignedAlloc((nuint)(m0 * sizeof(float)), 64);
        float* r1Ind = (float*)NativeMemory.AlignedAlloc((nuint)(m1 * sizeof(float)), 64);
        float* r2Ind = (float*)NativeMemory.AlignedAlloc((nuint)(m2 * sizeof(float)), 64);
        float* r0Fused = (float*)NativeMemory.AlignedAlloc((nuint)(m0 * sizeof(float)), 64);
        float* r1Fused = (float*)NativeMemory.AlignedAlloc((nuint)(m1 * sizeof(float)), 64);
        float* r2Fused = (float*)NativeMemory.AlignedAlloc((nuint)(m2 * sizeof(float)), 64);

        // Pre-quantize input: one per family so each projection gets the right format
        byte* preQuant0 = AllocPreQuant(input, k, qt0);
        byte* preQuant1 = IsSameQuantFamily(qt0, qt1) ? preQuant0 : AllocPreQuant(input, k, qt1);
        byte* preQuant2 = IsSameQuantFamily(qt0, qt2) ? preQuant0
                        : IsSameQuantFamily(qt1, qt2) ? preQuant1
                        : AllocPreQuant(input, k, qt2);

        try
        {
            // Individual calls via GemmQ*_* with n=1 and correctly-formatted preQuantized input
            CallIndividualGemm(w0, qt0, input, r0Ind, m0, k, preQuant0);
            CallIndividualGemm(w1, qt1, input, r1Ind, m1, k, preQuant1);
            CallIndividualGemm(w2, qt2, input, r2Ind, m2, k, preQuant2);

            // Fused call (preQuant0 is for qt0's family; FusedDecodeGemv3 handles cross-family)
            MatMul.FusedDecodeGemv3(
                w0, qt0, r0Fused, m0,
                w1, qt1, r1Fused, m1,
                w2, qt2, r2Fused, m2,
                input, preQuant0, k, _pool);

            // Assert bit-identical
            AssertBitIdentical(r0Ind, r0Fused, m0, "Proj0");
            AssertBitIdentical(r1Ind, r1Fused, m1, "Proj1");
            AssertBitIdentical(r2Ind, r2Fused, m2, "Proj2");
        }
        finally
        {
            NativeMemory.AlignedFree(w0);
            NativeMemory.AlignedFree(w1);
            NativeMemory.AlignedFree(w2);
            NativeMemory.AlignedFree(input);
            NativeMemory.AlignedFree(r0Ind);
            NativeMemory.AlignedFree(r1Ind);
            NativeMemory.AlignedFree(r2Ind);
            NativeMemory.AlignedFree(r0Fused);
            NativeMemory.AlignedFree(r1Fused);
            NativeMemory.AlignedFree(r2Fused);
            if (preQuant0 != null) NativeMemory.AlignedFree(preQuant0);
            if (preQuant1 != null && preQuant1 != preQuant0) NativeMemory.AlignedFree(preQuant1);
            if (preQuant2 != null && preQuant2 != preQuant0 && preQuant2 != preQuant1) NativeMemory.AlignedFree(preQuant2);
        }
    }

    private void RunFusedDecode2Test(int m0, int m1, int k,
        QuantizationType qt0, QuantizationType qt1)
    {
        var rng = new Random(42);

        byte* w0 = AllocWeights(m0, k, qt0, rng);
        byte* w1 = AllocWeights(m1, k, qt1, rng);

        float* input = (float*)NativeMemory.AlignedAlloc((nuint)(k * sizeof(float)), 64);
        FillFloats(input, k, rng);

        float* r0Ind = (float*)NativeMemory.AlignedAlloc((nuint)(m0 * sizeof(float)), 64);
        float* r1Ind = (float*)NativeMemory.AlignedAlloc((nuint)(m1 * sizeof(float)), 64);
        float* r0Fused = (float*)NativeMemory.AlignedAlloc((nuint)(m0 * sizeof(float)), 64);
        float* r1Fused = (float*)NativeMemory.AlignedAlloc((nuint)(m1 * sizeof(float)), 64);

        byte* preQuant0 = AllocPreQuant(input, k, qt0);
        byte* preQuant1 = IsSameQuantFamily(qt0, qt1) ? preQuant0 : AllocPreQuant(input, k, qt1);

        try
        {
            CallIndividualGemm(w0, qt0, input, r0Ind, m0, k, preQuant0);
            CallIndividualGemm(w1, qt1, input, r1Ind, m1, k, preQuant1);

            MatMul.FusedDecodeGemv2(
                w0, qt0, r0Fused, m0,
                w1, qt1, r1Fused, m1,
                input, preQuant0, k, _pool);

            AssertBitIdentical(r0Ind, r0Fused, m0, "Proj0");
            AssertBitIdentical(r1Ind, r1Fused, m1, "Proj1");
        }
        finally
        {
            NativeMemory.AlignedFree(w0);
            NativeMemory.AlignedFree(w1);
            NativeMemory.AlignedFree(input);
            NativeMemory.AlignedFree(r0Ind);
            NativeMemory.AlignedFree(r1Ind);
            NativeMemory.AlignedFree(r0Fused);
            NativeMemory.AlignedFree(r1Fused);
            if (preQuant0 != null) NativeMemory.AlignedFree(preQuant0);
            if (preQuant1 != null && preQuant1 != preQuant0) NativeMemory.AlignedFree(preQuant1);
        }
    }

    // ──────────────────── Helpers ────────────────────

    private static void CallIndividualGemm(byte* weights, QuantizationType qt,
        float* input, float* result, int m, int k, byte* preQuant)
    {
        // Use GEMM with n=1 and pre-quantized input — this is what TransformerModel does
        switch (qt)
        {
            case QuantizationType.Q8_0:
                MatMul.GemmQ8_0(weights, input, result, m, k, 1, preQuantizedInput: preQuant);
                break;
            case QuantizationType.Q5_0:
                MatMul.GemmQ5_0(weights, input, result, m, k, 1, preQuantizedInput: preQuant);
                break;
            case QuantizationType.Q4_K:
                MatMul.GemmQ4_K(weights, input, result, m, k, 1, preQuantizedInput: preQuant);
                break;
            case QuantizationType.Q5_K:
                MatMul.GemmQ5_K(weights, input, result, m, k, 1, preQuantizedInput: preQuant);
                break;
            case QuantizationType.Q6_K:
                MatMul.GemmQ6_K(weights, input, result, m, k, 1, preQuantizedInput: preQuant);
                break;
            default:
                throw new NotSupportedException($"Quant type {qt} not supported in test");
        }
    }

    private static byte* AllocWeights(int m, int k, QuantizationType qt, Random rng)
    {
        return qt switch
        {
            QuantizationType.Q8_0 => AllocQ8_0Weights(m, k, rng),
            QuantizationType.Q5_0 => AllocQ5_0Weights(m, k, rng),
            QuantizationType.Q4_K => AllocKQuantWeights(m, k, Q4_K_BlockBytes, rng),
            QuantizationType.Q5_K => AllocKQuantWeights(m, k, Q5_K_BlockBytes, rng),
            QuantizationType.Q6_K => AllocKQuantWeights(m, k, Q6_K_BlockBytes, rng),
            _ => throw new NotSupportedException(),
        };
    }

    private static byte* AllocQ8_0Weights(int m, int k, Random rng)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        int totalBytes = m * rowBytes;
        byte* ptr = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        FillQ8Weights(ptr, m, blocksPerRow, rng);
        return ptr;
    }

    private static byte* AllocQ5_0Weights(int m, int k, Random rng)
    {
        int blocksPerRow = k / Q8_0GroupSize; // Q5_0 uses same group size as Q8_0 (32)
        int rowBytes = blocksPerRow * Q5_0BlockBytes;
        int totalBytes = m * rowBytes;
        byte* ptr = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        FillQ5_0Weights(ptr, m, blocksPerRow, rng);
        return ptr;
    }

    private static byte* AllocKQuantWeights(int m, int k, int blockBytes, Random rng)
    {
        int superBlockCount = k / KQuantGroupSize;
        int rowBytes = superBlockCount * blockBytes;
        int totalBytes = m * rowBytes;
        byte* ptr = (byte*)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        FillKQuantWeights(ptr, totalBytes, rng);
        return ptr;
    }

    private static bool IsSameQuantFamily(QuantizationType a, QuantizationType b)
    {
        if (a == b) return true;

        bool aIsKQuant = a is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;
        bool bIsKQuant = b is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;
        if (aIsKQuant && bIsKQuant) return true;

        // Q8_0 and Q5_0 are no longer in the same family — Q5_0 uses Q8_1 input
        return false;
    }

    private static byte* AllocPreQuant(float* input, int k, QuantizationType qt)
    {
        bool isKQuant = qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;

        if (isKQuant)
        {
            int blockCount = k / KQuantGroupSize;
            int q8kBytes = blockCount * MatMul.Q8_K_BlockBytes;
            byte* preQuant = (byte*)NativeMemory.AlignedAlloc((nuint)q8kBytes, 64);
            MatMul.QuantizeF32ToQ8_K(input, preQuant, k);
            return preQuant;
        }

        if (qt == QuantizationType.Q5_0)
        {
            int blockCount = k / Q8_0GroupSize;
            int q8_1Bytes = blockCount * MatMul.Q8_1BlockBytes;
            byte* preQuant = (byte*)NativeMemory.AlignedAlloc((nuint)q8_1Bytes, 64);
            MatMul.QuantizeF32ToQ8_1(input, preQuant, k);
            return preQuant;
        }

        if (qt == QuantizationType.Q8_0)
        {
            int blockCount = k / Q8_0GroupSize;
            int q8Bytes = blockCount * Q8_0BlockBytes;
            byte* preQuant = (byte*)NativeMemory.AlignedAlloc((nuint)q8Bytes, 64);
            MatMul.QuantizeF32ToQ8_0(input, preQuant, k);
            return preQuant;
        }

        return null;
    }

    private static void FillFloats(float* ptr, int count, Random rng)
    {
        for (int i = 0; i < count; i++)
            ptr[i] = rng.NextSingle() * 2f - 1f;
    }

    private static void FillQ8Weights(byte* ptr, int rows, int blocksPerRow, Random rng)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* block = ptr + (row * blocksPerRow + b) * Q8_0BlockBytes;
                *(Half*)block = (Half)(rng.NextSingle() * 0.1f);
                for (int i = 0; i < Q8_0GroupSize; i++)
                    ((sbyte*)(block + 2))[i] = (sbyte)rng.Next(-127, 128);
            }
        }
    }

    private static void FillQ5_0Weights(byte* ptr, int rows, int blocksPerRow, Random rng)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* block = ptr + (row * blocksPerRow + b) * Q5_0BlockBytes;
                // Q5_0 block: Half d(2) + uint32 qh(4) + byte[16] qs(16) = 22 bytes
                *(Half*)block = (Half)(rng.NextSingle() * 0.1f);
                *(uint*)(block + 2) = (uint)rng.Next(); // qh — high bits
                for (int i = 0; i < 16; i++)
                    block[6 + i] = (byte)rng.Next(256); // qs — low nibbles packed
            }
        }
    }

    private static void FillKQuantWeights(byte* ptr, int totalBytes, Random rng)
    {
        // K-quant blocks have complex structure — fill with random bytes
        // The vec_dot kernels handle any byte pattern correctly
        for (int i = 0; i < totalBytes; i++)
            ptr[i] = (byte)rng.Next(256);
    }

    private static void AssertBitIdentical(float* expected, float* actual, int count, string projName)
    {
        for (int i = 0; i < count; i++)
        {
            Assert.True(
                BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{projName}: Mismatch at index {i}: expected {expected[i]:G9}, actual {actual[i]:G9}");
        }
    }
}
