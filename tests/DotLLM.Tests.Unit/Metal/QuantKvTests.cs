using DotLLM.Cpu.Kernels;
using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

/// <summary>
/// Tests for <see cref="QuantKv"/>.
///
/// Strategy: round-trip comparison.
///   1. Convert FP16 input to FP32 (lossless).
///   2. Quantize on CPU via <see cref="KvQuantize"/> (FP32 path).
///   3. Quantize on GPU via <see cref="QuantKv"/> (FP16 path, Metal kernel).
///   4. Dequantize both outputs back to FP32 via <see cref="KvQuantize"/>.
///   5. Compare dequantized floats with a tolerance of one quantization step.
///
/// This avoids byte-exact comparison, which would require matching the rounding
/// convention of Metal's <c>round()</c> (away-from-zero) in the CPU reference.
/// A ±1 difference in a quantized weight maps to at most <c>d = max_abs / scale</c>
/// in float, which the tolerance absorbs.
/// </summary>
public sealed class QuantKvTests
{
    // ── Helpers ───────────────────────────────────────────────────────────────

    private static Half[] BuildSrc(int totalBlocks, int seed)
    {
        const int blockSize = 32;
        var rng = new Random(seed);
        var src = new Half[totalBlocks * blockSize];
        for (int i = 0; i < src.Length; i++)
            src[i] = (Half)(rng.NextSingle() * 2f - 1f);
        return src;
    }

    private static float[] HalfToFloat(Half[] src)
    {
        var dst = new float[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (float)src[i];
        return dst;
    }

    // Dequantize Q8_0 bytes → float32 using the existing CPU kernel.
    private static unsafe float[] DequantQ8_0(byte[] src, int elementCount)
    {
        float[] dst = new float[elementCount];
        fixed (byte*  p = src)
        fixed (float* q = dst)
            KvQuantize.Q8_0ToF32(p, q, elementCount);
        return dst;
    }

    // Dequantize Q4_0 bytes → float32 using the existing CPU kernel.
    private static unsafe float[] DequantQ4_0(byte[] src, int elementCount)
    {
        float[] dst = new float[elementCount];
        fixed (byte*  p = src)
        fixed (float* q = dst)
            KvQuantize.Q4_0ToF32(p, q, elementCount);
        return dst;
    }

    // CPU quantize float32 → Q8_0 bytes.
    private static unsafe byte[] CpuQuantQ8_0(float[] src, int totalBlocks)
    {
        byte[] dst = new byte[totalBlocks * KvQuantize.Q8_0BlockBytes];
        fixed (float* p = src)
        fixed (byte*  q = dst)
            KvQuantize.F32ToQ8_0(p, q, src.Length);
        return dst;
    }

    // CPU quantize float32 → Q4_0 bytes.
    private static unsafe byte[] CpuQuantQ4_0(float[] src, int totalBlocks)
    {
        byte[] dst = new byte[totalBlocks * KvQuantize.Q4_0BlockBytes];
        fixed (float* p = src)
        fixed (byte*  q = dst)
            KvQuantize.F32ToQ4_0(p, q, src.Length);
        return dst;
    }

    private static void AssertEqual(float[] cpuDequant, float[] gpuDequant, float tol)
    {
        Assert.Equal(cpuDequant.Length, gpuDequant.Length);
        for (int i = 0; i < cpuDequant.Length; i++)
            Assert.Equal(cpuDequant[i], gpuDequant[i], tol);
    }

    // ── Q8_0 tests ────────────────────────────────────────────────────────────

    // Q8_0 tolerance: one quantization step = max_abs / 127.
    // For inputs in [-1, 1], max_abs ≤ 1, so tol = 1/127 ≈ 0.008.
    private const float Q8_0Tol = 0.01f;

    [MetalTestFact]
    public void Q8_0_SingleBlock_MatchesCpu()
    {
        const int totalBlocks = 1, n = totalBlocks * 32;
        Half[]  src      = BuildSrc(totalBlocks, seed: 1);
        float[] srcFloat = HalfToFloat(src);

        byte[]  cpuBytes = CpuQuantQ8_0(srcFloat, totalBlocks);
        byte[]  gpuBytes = new byte[totalBlocks * KvQuantize.Q8_0BlockBytes];
        using var ctx = new MetalContext();
        QuantKv.F16ToQ8_0(ctx, src, gpuBytes, totalBlocks);

        AssertEqual(DequantQ8_0(cpuBytes, n), DequantQ8_0(gpuBytes, n), Q8_0Tol);
    }

    [MetalTestFact]
    public void Q8_0_MultipleBlocks_MatchesCpu()
    {
        const int totalBlocks = 8, n = totalBlocks * 32;
        Half[]  src      = BuildSrc(totalBlocks, seed: 2);
        float[] srcFloat = HalfToFloat(src);

        byte[]  cpuBytes = CpuQuantQ8_0(srcFloat, totalBlocks);
        byte[]  gpuBytes = new byte[totalBlocks * KvQuantize.Q8_0BlockBytes];
        using var ctx = new MetalContext();
        QuantKv.F16ToQ8_0(ctx, src, gpuBytes, totalBlocks);

        AssertEqual(DequantQ8_0(cpuBytes, n), DequantQ8_0(gpuBytes, n), Q8_0Tol);
    }

    [MetalTestFact]
    public void Q8_0_ManyBlocks_MatchesCpu()
    {
        const int totalBlocks = 128, n = totalBlocks * 32;
        Half[]  src      = BuildSrc(totalBlocks, seed: 3);
        float[] srcFloat = HalfToFloat(src);

        byte[]  cpuBytes = CpuQuantQ8_0(srcFloat, totalBlocks);
        byte[]  gpuBytes = new byte[totalBlocks * KvQuantize.Q8_0BlockBytes];
        using var ctx = new MetalContext();
        QuantKv.F16ToQ8_0(ctx, src, gpuBytes, totalBlocks);

        AssertEqual(DequantQ8_0(cpuBytes, n), DequantQ8_0(gpuBytes, n), Q8_0Tol);
    }

    [MetalTestFact]
    public void Q8_0_ZeroInput_ScaleZeroAndWeightsZero()
    {
        const int totalBlocks = 4, n = totalBlocks * 32;
        Half[] src    = new Half[n]; // all zero
        byte[] actual = new byte[totalBlocks * KvQuantize.Q8_0BlockBytes];

        using var ctx = new MetalContext();
        QuantKv.F16ToQ8_0(ctx, src, actual, totalBlocks);

        float[] dequant = DequantQ8_0(actual, n);
        foreach (float v in dequant) Assert.Equal(0f, v, 1e-6f);
    }

    // ── Q4_0 tests ────────────────────────────────────────────────────────────

    // Q4_0 tolerance: one quantization step = max_abs / 7.
    // For inputs in [-1, 1], max_abs ≤ 1, so tol = 1/7 ≈ 0.143.
    private const float Q4_0Tol = 0.15f;

    [MetalTestFact]
    public void Q4_0_SingleBlock_MatchesCpu()
    {
        const int totalBlocks = 1, n = totalBlocks * 32;
        Half[]  src      = BuildSrc(totalBlocks, seed: 4);
        float[] srcFloat = HalfToFloat(src);

        byte[]  cpuBytes = CpuQuantQ4_0(srcFloat, totalBlocks);
        byte[]  gpuBytes = new byte[totalBlocks * KvQuantize.Q4_0BlockBytes];
        using var ctx = new MetalContext();
        QuantKv.F16ToQ4_0(ctx, src, gpuBytes, totalBlocks);

        AssertEqual(DequantQ4_0(cpuBytes, n), DequantQ4_0(gpuBytes, n), Q4_0Tol);
    }

    [MetalTestFact]
    public void Q4_0_MultipleBlocks_MatchesCpu()
    {
        const int totalBlocks = 8, n = totalBlocks * 32;
        Half[]  src      = BuildSrc(totalBlocks, seed: 5);
        float[] srcFloat = HalfToFloat(src);

        byte[]  cpuBytes = CpuQuantQ4_0(srcFloat, totalBlocks);
        byte[]  gpuBytes = new byte[totalBlocks * KvQuantize.Q4_0BlockBytes];
        using var ctx = new MetalContext();
        QuantKv.F16ToQ4_0(ctx, src, gpuBytes, totalBlocks);

        AssertEqual(DequantQ4_0(cpuBytes, n), DequantQ4_0(gpuBytes, n), Q4_0Tol);
    }

    [MetalTestFact]
    public void Q4_0_ManyBlocks_MatchesCpu()
    {
        const int totalBlocks = 128, n = totalBlocks * 32;
        Half[]  src      = BuildSrc(totalBlocks, seed: 6);
        float[] srcFloat = HalfToFloat(src);

        byte[]  cpuBytes = CpuQuantQ4_0(srcFloat, totalBlocks);
        byte[]  gpuBytes = new byte[totalBlocks * KvQuantize.Q4_0BlockBytes];
        using var ctx = new MetalContext();
        QuantKv.F16ToQ4_0(ctx, src, gpuBytes, totalBlocks);

        AssertEqual(DequantQ4_0(cpuBytes, n), DequantQ4_0(gpuBytes, n), Q4_0Tol);
    }

    [MetalTestFact]
    public void Q4_0_ZeroInput_ScaleZeroAndWeightsCentered()
    {
        const int totalBlocks = 4, n = totalBlocks * 32;
        Half[] src    = new Half[n]; // all zero
        byte[] actual = new byte[totalBlocks * KvQuantize.Q4_0BlockBytes];

        using var ctx = new MetalContext();
        QuantKv.F16ToQ4_0(ctx, src, actual, totalBlocks);

        float[] dequant = DequantQ4_0(actual, n);
        foreach (float v in dequant) Assert.Equal(0f, v, 1e-6f);
    }
}
