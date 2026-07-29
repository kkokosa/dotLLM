using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q8_0 GEMV kernel.
/// </summary>
/// <remarks>
/// <para>
/// Validation strategy: generate random FP32 weights, quantize them to Q8_0
/// via the CPU kernel (<c>MatMul.QuantizeF32ToQ8_0</c>) — this produces the
/// exact byte blob the GPU shader must read. Reference result is from the CPU
/// scalar Q8_0 GEMV (<c>MatMul.VecDotQ8_0Scalar</c>) run against the
/// *same* quantized bytes with a Q8_0-quantized copy of <c>x</c>.
/// </para>
/// <para>
/// This is a stricter test than "quantize → compare to FP32": by comparing
/// Q8_0-GPU vs Q8_0-CPU on byte-identical weights we catch bugs in bit-unpack,
/// sign-extension, and block-stride arithmetic that a FP32-reference would mask.
/// </para>
/// <para>
/// Tolerance mandated: relative 1e-3 / absolute 1e-4. The GPU kernel uses a
/// different reduction order (workgroup tree reduce vs. CPU block-sequential)
/// which produces small but nonzero drift at K=576+; 1e-3 rel / 1e-4 abs
/// is comfortably above that noise floor on AMD Radeon 8060S.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanMatMulQ8_0KernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 32)]                   // minimum: 1 block
    [InlineData(8, 64)]                   // 2 blocks per row
    [InlineData(4, 128)]                  // 4 blocks per row, odd row-byte alignment (4*34=136)
    [InlineData(49152, 576)]              // SmolLM lm_head / vocab-size output
    [InlineData(576, 576)]                // SmolLM q/k/v projection shape
    [InlineData(1536, 576)]               // SmolLM gate/up projection
    [InlineData(576, 1536)]               // SmolLM down projection
    // Regression for issue #1 — latent stride bug at K=32 with M>1. Row stride
    // = blocksPerRow*34 = 34 bytes, but rowUints*4 = 36 (rounded up). Earlier
    // shader read at the rowUints stride and silently returned garbage for
    // rows beyond the first. Repeats for any K where blocksPerRow*34 % 4 != 0
    // (i.e., blocksPerRow odd: K = 32, 96, 160, 224, ...).
    [InlineData(8, 32)]                   // K=32, M>1 — historical bug
    [InlineData(4, 96)]                   // K=96, blocksPerRow=3 (odd) — same family
    [InlineData(2, 160)]                  // K=160, blocksPerRow=5 (odd) — same family
    public void Launch_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBEEF + m * 7 + k);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);
        float[] x = RandomFloats(rng, k, range: 1.0f);

        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ8.Length);

        // CPU reference uses the bytes exactly as the GPU sees them.
        float[] expected = CpuGemvQ8_0(weightsQ8, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ8_0Kernel.Create(device, spvDir);

        // Round buffer size up to 4-byte multiple — the shader reads the weights
        // buffer as a uint array.
        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        AssertClose(expected, actual, m, k);
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    /// <summary>
    /// Quantize an <c>[m,k]</c> row-major FP32 matrix to the Q8_0 byte blob
    /// expected by both the CPU <c>GemvQ8_0</c> path and the Vulkan kernel.
    /// </summary>
    private static unsafe byte[] QuantizeRows(float[] src, int m, int k)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        var dst = new byte[m * rowBytes];
        fixed (float* srcPtr = src)
        fixed (byte* dstPtr = dst)
        {
            for (int row = 0; row < m; row++)
            {
                MatMul.QuantizeF32ToQ8_0(srcPtr + (long)row * k, dstPtr + (long)row * rowBytes, k);
            }
        }
        return dst;
    }

    /// <summary>
    /// Scalar CPU reference: reads the same Q8_0 byte blob the GPU sees,
    /// dequantizes on the fly, dots against FP32 <c>x</c>. Block-sequential
    /// reduction matches <c>MatMul.VecDotQ8_0Scalar</c> semantics (not the
    /// quantized-input path — Vulkan kernel reads x in FP32).
    /// </summary>
    private static unsafe float[] CpuGemvQ8_0(byte[] weightsQ8, float[] x, int m, int k)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        var result = new float[m];

        fixed (byte* wPtr = weightsQ8)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = rowBase + b * Q8_0BlockBytes;
                    float d = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(block);
                    sbyte* qs = (sbyte*)(block + 2);

                    float blockSum = 0;
                    for (int j = 0; j < Q8_0GroupSize; j++)
                        blockSum += (float)qs[j] * x[b * Q8_0GroupSize + j];
                    sum += d * blockSum;
                }
                result[row] = sum;
            }
        }
        return result;
    }

    private static void AssertClose(float[] expected, float[] actual, int m, int k)
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
            $"Numerical drift exceeded tolerance (m={m},k={k}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
