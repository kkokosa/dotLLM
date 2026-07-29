using System.Diagnostics;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q8_0 batched GEMM (prefill path).
/// </summary>
/// <remarks>
/// <para>
/// Validation strategy mirrors <see cref="VulkanMatMulQ8_0KernelTests"/>: we
/// quantize random FP32 weights to Q8_0 via the CPU kernel so both sides see
/// byte-identical weights, and compare the Vulkan kernel output against a
/// scalar CPU reference run against the <em>same</em> Q8_0 bytes. This is a
/// stricter test than "quantize + compare to FP32" — it catches block-stride
/// / sign-extension / fp16-scale-straddle bugs that a FP32-reference would
/// mask.
/// </para>
/// <para>
/// Shapes:
/// <list type="bullet">
///   <item>Tiny sanity: <c>N=2, M=4, K=32</c> (one block per row).</item>
///   <item>SmolLM-135M QKV/O projection: <c>N=64, M=576, K=576</c>.</item>
///   <item>SmolLM-135M Gate/Up projection: <c>N=64, M=1536, K=576</c>.</item>
///   <item>Llama-3-8B projection: <c>N=64, M=4096, K=4096</c>.</item>
/// </list>
/// Tolerance mandated: absolute 1e-4, relative 1e-3.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanMatMulQ8_0GemmKernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    private readonly ITestOutputHelper _output;

    public VulkanMatMulQ8_0GemmKernelTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableTheory]
    [InlineData(2, 4, 32)]           // tiny sanity: one block per row
    [InlineData(1, 1, 32)]           // single-cell output (bounds check)
    [InlineData(17, 33, 64)]         // non-multiple-of-tile sizes, odd row alignment
    [InlineData(64, 576, 576)]       // SmolLM-135M QKV/O projection (prefill batch)
    [InlineData(64, 1536, 576)]      // SmolLM-135M Gate/Up projection (prefill batch)
    [InlineData(64, 4096, 4096)]     // Llama-3-8B projection (prefill batch)
    public void Launch_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xFEED + n * 31 + m * 17 + k * 3);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = RandomFloats(rng, n * k, range: 1.0f);

        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ8.Length);

        // CPU reference uses the exact same Q8_0 bytes the GPU sees.
        float[] expected = CpuGemmQ8_0(weightsQ8, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ8_0GemmKernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(inputB, bufB);

        // Single timed dispatch so the test doubles as a perf smoke signal.
        var sw = Stopwatch.StartNew();
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        sw.Stop();

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        AssertClose(expected, actual, m, k, n, sw.Elapsed);
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
    /// Quantize an <c>[m, k]</c> row-major FP32 matrix to the Q8_0 byte blob
    /// expected by both the CPU <c>GemmQ8_0</c> path and the Vulkan kernel.
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
    /// Scalar CPU reference: <c>C[N,M] = B[N,K] @ W_q8[M,K]^T</c>, reading the
    /// same Q8_0 byte blob the GPU sees, dequantizing on the fly, block-
    /// sequential reduction. Matches the per-row loop in
    /// <see cref="VulkanMatMulQ8_0KernelTests"/> extended over the N-batch.
    /// </summary>
    private static unsafe float[] CpuGemmQ8_0(byte[] weightsQ8, float[] b, int m, int k, int n)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        var result = new float[n * m];

        fixed (byte* wPtr = weightsQ8)
        fixed (float* bPtr = b)
        {
            for (int t = 0; t < n; t++)
            {
                float* bRow = bPtr + (long)t * k;
                for (int row = 0; row < m; row++)
                {
                    byte* rowBase = wPtr + (long)row * rowBytes;
                    float sum = 0;
                    for (int blk = 0; blk < blocksPerRow; blk++)
                    {
                        byte* block = rowBase + blk * Q8_0BlockBytes;
                        float d = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(block);
                        sbyte* qs = (sbyte*)(block + 2);

                        float blockSum = 0;
                        for (int j = 0; j < Q8_0GroupSize; j++)
                            blockSum += (float)qs[j] * bRow[blk * Q8_0GroupSize + j];
                        sum += d * blockSum;
                    }
                    result[t * m + row] = sum;
                }
            }
        }
        return result;
    }

    private void AssertClose(float[] expected, float[] actual, int m, int k, int n, TimeSpan elapsed)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        double sumAbs = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            sumAbs += diff;
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        double meanAbs = sumAbs / expected.Length;
        _output.WriteLine(
            $"Q8_0 GEMM n={n} m={m} k={k}: elapsed={elapsed.TotalMilliseconds:F2} ms, " +
            $"maxAbs={maxAbs:G6}, meanAbs={meanAbs:G6}, maxRel={maxRel:G6}");
        Assert.True(errors == 0,
            $"Numerical drift exceeded tolerance (n={n},m={m},k={k}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
