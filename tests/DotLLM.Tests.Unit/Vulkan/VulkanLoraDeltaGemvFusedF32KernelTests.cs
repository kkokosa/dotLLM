using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity tests for <see cref="LoraDeltaGemvFusedF32Kernel"/>: confirm
/// the single-dispatch fused shader produces the same y as the un-fused
/// <c>matmul(B) → matmul(A) → add</c> chain for the common LoRA ranks
/// (4 / 8 / 16 / 32) at typical TinyLlama / Llama-3 projection shapes.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanLoraDeltaGemvFusedF32KernelTests
{
    private const float AbsTol = 5e-4f;
    private const float RelTol = 1e-4f;

    public static IEnumerable<object[]> ParityCases()
    {
        // (seqLen, inputDim, outputDim, rank)
        // Decode-path shapes (seqLen=1) at TinyLlama hidden=2048.
        yield return new object[] { 1, 2048, 2048, 4 };
        yield return new object[] { 1, 2048, 2048, 8 };
        yield return new object[] { 1, 2048, 2048, 16 };
        yield return new object[] { 1, 2048, 2048, 32 };
        // Down-projection shape: ffn_intermediate=5632 -> 2048.
        yield return new object[] { 1, 5632, 2048, 16 };
        // Asymmetric small case to exercise non-multiple-of-WG outputDim.
        yield return new object[] { 1, 65, 17, 8 };
        // Prefill shape (seqLen > 1) — the fused shader is allowed to be
        // used here too, though MaybeApplyLoraDelta routes prefill through
        // the un-fused path for stability.
        yield return new object[] { 4, 256, 192, 8 };
    }

    [SkippableTheory]
    [MemberData(nameof(ParityCases))]
    public void Launch_MatchesUnfusedChain(int seqLen, int inputDim, int outputDim, int rank)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBEEF + seqLen * 7919 + inputDim * 31 + outputDim * 17 + rank);
        float[] x = RandomFloats(rng, seqLen * inputDim, scale: 0.05f);
        float[] bScaled = RandomFloats(rng, rank * inputDim, scale: 0.05f);  // already alpha/rank-folded
        float[] aWeight = RandomFloats(rng, outputDim * rank, scale: 0.05f);
        float[] yBase = RandomFloats(rng, seqLen * outputDim, scale: 0.5f);

        // Ground truth: scalar reference, computed on the host.
        float[] expected = (float[])yBase.Clone();
        ApplyReference(x, bScaled, aWeight, expected, seqLen, inputDim, outputDim, rank);

        using var device = VulkanDevice.Create();

        // Fused path.
        float[] actualFused;
        using (var fused = LoraDeltaGemvFusedF32Kernel.Create(device, spvDir))
        using (var bufX = device.Allocate((long)x.Length * sizeof(float)))
        using (var bufB = device.Allocate((long)bScaled.Length * sizeof(float)))
        using (var bufA = device.Allocate((long)aWeight.Length * sizeof(float)))
        using (var bufY = device.Allocate((long)yBase.Length * sizeof(float)))
        using (var bufTmp = device.Allocate((long)seqLen * rank * sizeof(float)))
        {
            device.Upload(x, bufX);
            device.Upload(bScaled, bufB);
            device.Upload(aWeight, bufA);
            device.Upload(yBase, bufY);

            fused.Launch(bufX, bufB, bufA, bufY, bufTmp, seqLen, inputDim, outputDim, rank);

            actualFused = new float[yBase.Length];
            device.Download(bufY, actualFused);
        }

        // Un-fused parity path: matmul(B,x) → matmul(A,tmp) → add(yBase, delta).
        // This is the exact dispatch sequence MaybeApplyLoraDelta records today.
        float[] actualUnfused;
        using (var matmul = MatMulF32Kernel.Create(device, spvDir))
        using (var add = AddKernel.Create(device, spvDir))
        using (var bufX = device.Allocate((long)x.Length * sizeof(float)))
        using (var bufB = device.Allocate((long)bScaled.Length * sizeof(float)))
        using (var bufA = device.Allocate((long)aWeight.Length * sizeof(float)))
        using (var bufY = device.Allocate((long)yBase.Length * sizeof(float)))
        using (var bufTmp = device.Allocate((long)seqLen * rank * sizeof(float)))
        using (var bufDelta = device.Allocate((long)seqLen * outputDim * sizeof(float)))
        using (var bufSum = device.Allocate((long)seqLen * outputDim * sizeof(float)))
        {
            device.Upload(x, bufX);
            device.Upload(bScaled, bufB);
            device.Upload(aWeight, bufA);
            device.Upload(yBase, bufY);

            // tmp[seqLen, rank] = matmul(B[rank, inputDim], x[seqLen, inputDim])
            //   weights = B (M=rank, K=inputDim), input = x (N=seqLen, K=inputDim)
            matmul.Launch(bufB, bufX, bufTmp, m: rank, k: inputDim, n: seqLen);
            // delta[seqLen, outputDim] = matmul(A[outputDim, rank], tmp[seqLen, rank])
            matmul.Launch(bufA, bufTmp, bufDelta, m: outputDim, k: rank, n: seqLen);
            // sum = y + delta
            add.Launch(bufY, bufDelta, bufSum, seqLen * outputDim);

            actualUnfused = new float[yBase.Length];
            device.Download(bufSum, actualUnfused);
        }

        // The fused shader and the un-fused chain do their floating-point
        // accumulations in slightly different orders (fused: per-thread A
        // sum-of-products with rank-sized tmp; un-fused: full matmul reduction
        // per cell + element-wise add). Compare each separately to the host
        // scalar reference, then to each other within float tolerance.
        AssertClose(expected, actualFused,   "fused vs reference",   AbsTol, RelTol);
        AssertClose(expected, actualUnfused, "unfused vs reference", AbsTol, RelTol);
        AssertClose(actualUnfused, actualFused, "fused vs unfused",  AbsTol, RelTol);
    }

    private static void ApplyReference(
        float[] x, float[] b, float[] a, float[] y,
        int seqLen, int inputDim, int outputDim, int rank)
    {
        Span<float> tmp = stackalloc float[rank];
        for (int t = 0; t < seqLen; t++)
        {
            for (int r = 0; r < rank; r++)
            {
                float acc = 0f;
                for (int k = 0; k < inputDim; k++)
                    acc += b[r * inputDim + k] * x[t * inputDim + k];
                tmp[r] = acc;
            }
            for (int m = 0; m < outputDim; m++)
            {
                float delta = 0f;
                for (int r = 0; r < rank; r++)
                    delta += a[m * rank + r] * tmp[r];
                y[t * outputDim + m] += delta;
            }
        }
    }

    private static float[] RandomFloats(Random rng, int n, float scale)
    {
        var arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual, string label, float absTol, float relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float tol = absTol + relTol * MathF.Abs(expected[i]);
            Assert.True(diff <= tol,
                $"{label}: i={i} expected={expected[i]} actual={actual[i]} diff={diff} tol={tol}");
        }
    }
}
