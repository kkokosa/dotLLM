using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan FP32 RoPE kernel.
/// </summary>
/// <remarks>
/// The Vulkan kernel mirrors <c>rope_f32.cu</c> — frequencies reconstructed on
/// the GPU from <c>theta</c>, not from pre-computed tables. Compared against
/// the scalar CPU reference <see cref="RoPE.ExecuteScalar"/> driven by a
/// <see cref="RoPE.PrecomputeFrequencyTableScalar"/> table; any tolerance
/// consumption comes from <c>cos/sin/pow</c> backend drift on the GPU.
/// Only <c>Norm</c> (interleaved) variant is validated here — that is the
/// convention used by Llama-family, SmolLM, and the CUDA reference kernel's
/// default (<c>rope_type != 1</c>).
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanRopeF32KernelTests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    // (seqLen, numHeads, numKvHeads, headDim, theta)
    [InlineData(4, 2, 2, 64, 10000f)]       // short, MHA
    [InlineData(4, 9, 3, 64, 10000f)]       // short, GQA (SmolLM shape fewer-tokens)
    [InlineData(256, 9, 3, 64, 10000f)]     // long, GQA — SmolLM-135M prefill shape
    [InlineData(1, 32, 8, 128, 500000f)]    // decode, Llama-3 style theta
    public void Launch_MatchesCpuReference_Norm(int seqLen, int numHeads, int numKvHeads, int headDim, float theta)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int ropeDim = headDim; // rotate the full head
        int halfDim = ropeDim / 2;

        var rng = new Random(0xABC + seqLen * 13 + numHeads * 7 + headDim);
        float[] q = RandomFloats(rng, seqLen * numHeads * headDim);
        float[] k = RandomFloats(rng, seqLen * numKvHeads * headDim);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        // CPU reference via scalar path using pre-computed tables — matches
        // the CUDA per-thread formula up to backend rounding.
        float[] cosTable = new float[seqLen * halfDim];
        float[] sinTable = new float[seqLen * halfDim];
        RoPE.PrecomputeFrequencyTableScalar(seqLen, headDim, theta, cosTable, sinTable);

        float[] qExpected = (float[])q.Clone();
        float[] kExpected = (float[])k.Clone();
        RoPE.ExecuteScalar(
            qExpected.AsSpan(), kExpected.AsSpan(), positions,
            numHeads, numKvHeads, headDim, ropeDim,
            cosTable, sinTable);

        // GPU path.
        using var device = VulkanDevice.Create();
        using var kernel = RopeF32Kernel.Create(device, spvDir);

        using var bufQ = device.Allocate(q.Length * sizeof(float));
        using var bufK = device.Allocate(k.Length * sizeof(float));
        using var bufPos = device.Allocate((long)positions.Length * sizeof(int));

        device.Upload(q.AsSpan(), bufQ);
        device.Upload(k.AsSpan(), bufK);
        device.Upload(MemoryMarshal.AsBytes(positions.AsSpan()), bufPos);

        kernel.Launch(bufQ, bufK, bufPos,
            seqLen, numHeads, numKvHeads, headDim, ropeDim, theta, RopeF32Kernel.Variant.Norm);

        float[] qActual = new float[q.Length];
        float[] kActual = new float[k.Length];
        device.Download(bufQ, qActual);
        device.Download(bufK, kActual);

        AssertClose(qExpected, qActual, "Q");
        AssertClose(kExpected, kActual, "K");
    }

    [SkippableTheory]
    // NeoX / rotate-half variant — pair (i, i+halfRope). This is the HF
    // safetensors convention (Llama-family via HF, Qwen2, Phi-3); the shader
    // already supports it via is_neox push-constant but only Norm had an
    // explicit parity test. The end-to-end VulkanTransformerModel calls this
    // variant only when loading from HF safetensors (GGUF Llama uses Norm),
    // but we validate it here to unblock non-GGUF model paths.
    [InlineData(4, 2, 2, 64, 10000f)]
    [InlineData(4, 9, 3, 64, 10000f)]
    [InlineData(256, 9, 3, 64, 10000f)]
    [InlineData(1, 32, 8, 128, 500000f)]
    public void Launch_MatchesCpuReference_NeoX(int seqLen, int numHeads, int numKvHeads, int headDim, float theta)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int ropeDim = headDim;
        int halfDim = ropeDim / 2;

        var rng = new Random(0xBEEF + seqLen * 13 + numHeads * 7 + headDim);
        float[] q = RandomFloats(rng, seqLen * numHeads * headDim);
        float[] k = RandomFloats(rng, seqLen * numKvHeads * headDim);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        float[] cosTable = new float[seqLen * halfDim];
        float[] sinTable = new float[seqLen * halfDim];
        RoPE.PrecomputeFrequencyTableScalar(seqLen, headDim, theta, cosTable, sinTable);

        float[] qExpected = (float[])q.Clone();
        float[] kExpected = (float[])k.Clone();
        RoPE.Execute(
            qExpected.AsSpan(), kExpected.AsSpan(), positions,
            numHeads, numKvHeads, headDim, ropeDim,
            cosTable, sinTable,
            DotLLM.Core.Configuration.RoPEType.NeoX);

        using var device = VulkanDevice.Create();
        using var kernel = RopeF32Kernel.Create(device, spvDir);

        using var bufQ = device.Allocate(q.Length * sizeof(float));
        using var bufK = device.Allocate(k.Length * sizeof(float));
        using var bufPos = device.Allocate((long)positions.Length * sizeof(int));

        device.Upload(q.AsSpan(), bufQ);
        device.Upload(k.AsSpan(), bufK);
        device.Upload(MemoryMarshal.AsBytes(positions.AsSpan()), bufPos);

        kernel.Launch(bufQ, bufK, bufPos,
            seqLen, numHeads, numKvHeads, headDim, ropeDim, theta, RopeF32Kernel.Variant.NeoX);

        float[] qActual = new float[q.Length];
        float[] kActual = new float[k.Length];
        device.Download(bufQ, qActual);
        device.Download(bufK, kActual);

        AssertClose(qExpected, qActual, "Q");
        AssertClose(kExpected, kActual, "K");
    }

    // ─────────────────────────────────────────────────────────────

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0); // [-1, 1]
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual, string tensorName)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"{tensorName}: Numerical drift exceeded tolerance: " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
