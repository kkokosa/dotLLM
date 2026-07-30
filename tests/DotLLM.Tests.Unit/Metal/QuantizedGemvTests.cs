using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

// ── Shared helpers ────────────────────────────────────────────────────────────

file static class GemvF16Helpers
{
    /// <summary>
    /// Asserts that two half-precision vectors are equal, with a tolerance exprimed in percent (base 100 : min = 0, max = 100).
    /// </summary>
    /// <param name="expected"></param>
    /// <param name="actual"></param>
    /// <param name="tolerancePercent"></param>
    internal static void AssertEqual(float[] expected, Half[] actual, float tolerancePercent = 0.2f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e   = expected[i];
            float a   = (float)actual[i];

            float maxAbs = MathF.Max(MathF.Abs(e), MathF.Abs(a));
            float relTol = maxAbs * (tolerancePercent / 100f);
            float absTol = 1e-4f;
            float tol = MathF.Max(relTol, absTol);
            Assert.Equal(e, a, tol);
        }
    }

    // Dequantize to float32, then scalar dot product.
    internal static float[] CpuGemv(byte[] weight, Half[] x, int n, int k, QuantizationType qt)
    {
        float[] wF = new float[n * k];
        unsafe
        {
            fixed (byte* p = weight)
                Dequantize.ToFloat32((nint)p, n * k, qt, wF);
        }
        float[] y = new float[n];
        for (int i = 0; i < n; i++)
        {
            float acc = 0f;
            for (int j = 0; j < k; j++)
                acc += wF[i * k + j] * (float)x[j];
            y[i] = acc;
        }
        return y;
    }

    internal static Half[] BuildX(int k, int seed)
    {
        var rng = new Random(seed);
        Half[] x = new Half[k];
        for (int i = 0; i < k; i++) x[i] = (Half)(rng.NextSingle() * 2f - 1f);
        return x;
    }
}

/// <summary>
/// Tests for <see cref="QuantizedGemv.Q8_0F32In"/>.
/// CPU reference: dequantize weight matrix with <see cref="Dequantize.ToFloat32"/>,
/// then compute the dot product in float32.
/// </summary>
public sealed class QuantizedGemvTests
{
    // ── CPU reference ─────────────────────────────────────────────────────────

    /// <summary>
    /// Dequantizes the Q8_0 weight matrix to float32, then computes y = W * x.
    /// </summary>
    private static float[] CpuGemv(byte[] weight, float[] x, int n, int k)
    {
        // Step 1: dequantize W → float32
        float[] wFloat = new float[n * k];
        unsafe
        {
            fixed (byte* p = weight)
                Dequantize.ToFloat32((nint)p, n * k, QuantizationType.Q8_0, wFloat);
        }

        // Step 2: y[i] = dot(W[i,:], x)
        float[] y = new float[n];
        for (int i = 0; i < n; i++)
        {
            float acc = 0f;
            for (int j = 0; j < k; j++)
                acc += wFloat[i * k + j] * x[j];
            y[i] = acc;
        }
        return y;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Build a Q8_0 weight matrix: n rows × (k/32) blocks × 34 bytes.
    private static byte[] BuildWeight(int n, int k, int seed)
    {
        const int BlockSize = 32, BlockBytes = 34;
        int blocksPerRow = k / BlockSize;
        byte[] buf = new byte[n * blocksPerRow * BlockBytes];
        var rng = new Random(seed);

        for (int row = 0; row < n; row++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (row * blocksPerRow + b) * BlockBytes;
            MemoryMarshal.Write(buf.AsSpan(off, 2), (Half)(rng.NextSingle() * 0.1f + 0.01f));
            for (int j = 0; j < BlockSize; j++)
                buf[off + 2 + j] = (byte)(sbyte)rng.Next(-127, 128);
        }
        return buf;
    }

    private static float[] BuildX(int k, int seed)
    {
        var rng = new Random(seed);
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;
        return x;
    }

    // The GPU accumulates in float32 with a different summation order than the
    // CPU reference, so we allow a small relative tolerance.
    private const float Tol = 1e-3f;

    private static void AssertEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], Tol);
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    [MetalTestFact]
    public void Q8_0_SingleRow_MatchesCpu()
    {
        const int n = 1, k = 32;
        byte[]  weight   = BuildWeight(n, k, seed: 1);
        float[] x        = BuildX(k, seed: 2);
        float[] expected = CpuGemv(weight, x, n, k);
        float[] y        = new float[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0F32In(ctx, weight, x, y, n, k);

        AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void Q8_0_MultipleRows_MatchesCpu()
    {
        const int n = 8, k = 128;
        byte[]  weight   = BuildWeight(n, k, seed: 3);
        float[] x        = BuildX(k, seed: 4);
        float[] expected = CpuGemv(weight, x, n, k);
        float[] y        = new float[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0F32In(ctx, weight, x, y, n, k);

        AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void Q8_0_LargerMatrix_MatchesCpu()
    {
        // Realistic hidden-dim size to stress the grid-stride loop.
        const int n = 32, k = 512;
        byte[]  weight   = BuildWeight(n, k, seed: 5);
        float[] x        = BuildX(k, seed: 6);
        float[] expected = CpuGemv(weight, x, n, k);
        float[] y        = new float[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0F32In(ctx, weight, x, y, n, k);

        AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void Q8_0_ZeroWeights_OutputIsZero()
    {
        const int n = 4, k = 64;
        int blocksPerRow = k / 32;
        byte[] weight = new byte[n * blocksPerRow * 34];
        // Write scale = 1.0 at each block header, leave int8 weights as 0.
        for (int row = 0; row < n; row++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (row * blocksPerRow + b) * 34;
            MemoryMarshal.Write(weight.AsSpan(off, 2), (Half)1.0f);
        }

        float[] x = BuildX(k, seed: 7);
        float[] y = new float[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0F32In(ctx, weight, x, y, n, k);

        foreach (float v in y) Assert.Equal(0f, v, 1e-6f);
    }

    [MetalTestFact]
    public void Q8_0_ZeroInput_OutputIsZero()
    {
        const int n = 4, k = 64;
        byte[]  weight = BuildWeight(n, k, seed: 8);
        float[] x      = new float[k]; // all zeros
        float[] y      = new float[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0F32In(ctx, weight, x, y, n, k);

        foreach (float v in y) Assert.Equal(0f, v, 1e-6f);
    }
}

// ── Q8_0 FP16 I/O ─────────────────────────────────────────────────────────────

public sealed class QuantizedGemvQ8_0Tests
{
    private static byte[] BuildWeight(int n, int k, int seed)
    {
        const int BlockSize = 32, BlockBytes = 34;
        int bpr = k / BlockSize;
        byte[] buf = new byte[n * bpr * BlockBytes];
        var rng = new Random(seed);
        for (int row = 0; row < n; row++)
        for (int b = 0; b < bpr; b++)
        {
            int off = (row * bpr + b) * BlockBytes;
            MemoryMarshal.Write(buf.AsSpan(off, 2), (Half)(rng.NextSingle() * 0.1f + 0.01f));
            for (int j = 0; j < BlockSize; j++)
                buf[off + 2 + j] = (byte)(sbyte)rng.Next(-127, 128);
        }
        return buf;
    }

    [MetalTestFact]
    public void SingleRow_MatchesCpu()
    {
        const int n = 1, k = 32;
        byte[]   weight = BuildWeight(n, k, seed: 10);
        Half[]   x      = GemvF16Helpers.BuildX(k, seed: 11);
        Half[]   y      = new Half[n];
        float[]  expected = GemvF16Helpers.CpuGemv(weight, x, n, k, QuantizationType.Q8_0);

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0(ctx, weight, x, y, n, k);

        GemvF16Helpers.AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void MultipleRows_MatchesCpu()
    {
        const int n = 8, k = 128;
        byte[]  weight   = BuildWeight(n, k, seed: 12);
        Half[]  x        = GemvF16Helpers.BuildX(k, seed: 13);
        Half[]  y        = new Half[n];
        float[] expected = GemvF16Helpers.CpuGemv(weight, x, n, k, QuantizationType.Q8_0);

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0(ctx, weight, x, y, n, k);

        GemvF16Helpers.AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void ZeroInput_OutputIsZero()
    {
        const int n = 4, k = 64;
        byte[] weight = BuildWeight(n, k, seed: 14);
        Half[] x      = new Half[k];
        Half[] y      = new Half[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0(ctx, weight, x, y, n, k);

        foreach (Half v in y) Assert.Equal(0f, (float)v, 1e-4f);
    }
}

// ── Q5_0 FP16 I/O ─────────────────────────────────────────────────────────────

public sealed class QuantizedGemvQ5_0Tests
{
    // Q5_0: 2-byte d + 4-byte qh + 16 packed nibbles = 22 bytes per 32 values.
    private static byte[] BuildWeight(int n, int k, int seed)
    {
        const int BlockSize = 32, BlockBytes = 22;
        int bpr = k / BlockSize;
        byte[] buf = new byte[n * bpr * BlockBytes];
        var rng = new Random(seed);
        for (int row = 0; row < n; row++)
        for (int b = 0; b < bpr; b++)
        {
            int off = (row * bpr + b) * BlockBytes;
            MemoryMarshal.Write(buf.AsSpan(off, 2), (Half)(rng.NextSingle() * 0.1f + 0.01f));
            // qh (4 bytes) + 16 nibble bytes
            for (int j = 2; j < BlockBytes; j++)
                buf[off + j] = (byte)rng.Next(256);
        }
        return buf;
    }

    [MetalTestFact]
    public void SingleRow_MatchesCpu()
    {
        const int n = 1, k = 32;
        byte[]  weight   = BuildWeight(n, k, seed: 20);
        Half[]  x        = GemvF16Helpers.BuildX(k, seed: 21);
        Half[]  y        = new Half[n];
        float[] expected = GemvF16Helpers.CpuGemv(weight, x, n, k, QuantizationType.Q5_0);

        using var ctx = new MetalContext();
        QuantizedGemv.Q5_0(ctx, weight, x, y, n, k);

        GemvF16Helpers.AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void MultipleRows_MatchesCpu()
    {
        const int n = 8, k = 128;
        byte[]  weight   = BuildWeight(n, k, seed: 22);
        Half[]  x        = GemvF16Helpers.BuildX(k, seed: 23);
        Half[]  y        = new Half[n];
        float[] expected = GemvF16Helpers.CpuGemv(weight, x, n, k, QuantizationType.Q5_0);

        using var ctx = new MetalContext();
        QuantizedGemv.Q5_0(ctx, weight, x, y, n, k);

        GemvF16Helpers.AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void ZeroInput_OutputIsZero()
    {
        const int n = 4, k = 64;
        byte[] weight = BuildWeight(n, k, seed: 24);
        Half[] x      = new Half[k];
        Half[] y      = new Half[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q5_0(ctx, weight, x, y, n, k);

        foreach (Half v in y) Assert.Equal(0f, (float)v, 1e-4f);
    }
}

// ── Q4_K FP16 I/O ─────────────────────────────────────────────────────────────

public sealed class QuantizedGemvQ4_KTests
{
    // Q4_K: 2d + 2dmin + 12 scales + 128 nibbles = 144 bytes per 256 values.
    private static byte[] BuildWeight(int n, int k, int seed)
    {
        const int SuperSize = 256, SuperBytes = 144;
        int sbpr = k / SuperSize;
        byte[] buf = new byte[n * sbpr * SuperBytes];
        var rng = new Random(seed);
        for (int row = 0; row < n; row++)
        for (int sb = 0; sb < sbpr; sb++)
        {
            int off = (row * sbpr + sb) * SuperBytes;
            MemoryMarshal.Write(buf.AsSpan(off,     2), (Half)(rng.NextSingle() * 0.1f + 0.01f));
            MemoryMarshal.Write(buf.AsSpan(off + 2, 2), (Half)(rng.NextSingle() * 0.02f));
            for (int j = 4; j < SuperBytes; j++)
                buf[off + j] = (byte)rng.Next(256);
        }
        return buf;
    }

    [MetalTestFact]
    public void SingleRow_MatchesCpu()
    {
        const int n = 1, k = 256;
        byte[]  weight   = BuildWeight(n, k, seed: 30);
        Half[]  x        = GemvF16Helpers.BuildX(k, seed: 31);
        Half[]  y        = new Half[n];
        float[] expected = GemvF16Helpers.CpuGemv(weight, x, n, k, QuantizationType.Q4_K);

        using var ctx = new MetalContext();
        QuantizedGemv.Q4_K(ctx, weight, x, y, n, k);

        GemvF16Helpers.AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void MultipleRows_MatchesCpu()
    {
        const int n = 4, k = 512;
        byte[]  weight   = BuildWeight(n, k, seed: 32);
        Half[]  x        = GemvF16Helpers.BuildX(k, seed: 33);
        Half[]  y        = new Half[n];
        float[] expected = GemvF16Helpers.CpuGemv(weight, x, n, k, QuantizationType.Q4_K);

        using var ctx = new MetalContext();
        QuantizedGemv.Q4_K(ctx, weight, x, y, n, k);

        GemvF16Helpers.AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void ZeroInput_OutputIsZero()
    {
        const int n = 2, k = 256;
        byte[] weight = BuildWeight(n, k, seed: 34);
        Half[] x      = new Half[k];
        Half[] y      = new Half[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q4_K(ctx, weight, x, y, n, k);

        foreach (Half v in y) Assert.Equal(0f, (float)v, 1e-4f);
    }
}

// ── Q5_K FP16 I/O ─────────────────────────────────────────────────────────────

public sealed class QuantizedGemvQ5_KTests
{
    // Q5_K: 2d + 2dmin + 12 scales + 32 qh + 128 qs = 176 bytes per 256 values.
    private static byte[] BuildWeight(int n, int k, int seed)
    {
        const int SuperSize = 256, SuperBytes = 176;
        int sbpr = k / SuperSize;
        byte[] buf = new byte[n * sbpr * SuperBytes];
        var rng = new Random(seed);
        for (int row = 0; row < n; row++)
        for (int sb = 0; sb < sbpr; sb++)
        {
            int off = (row * sbpr + sb) * SuperBytes;
            MemoryMarshal.Write(buf.AsSpan(off,     2), (Half)(rng.NextSingle() * 0.1f + 0.01f));
            MemoryMarshal.Write(buf.AsSpan(off + 2, 2), (Half)(rng.NextSingle() * 0.02f));
            for (int j = 4; j < SuperBytes; j++)
                buf[off + j] = (byte)rng.Next(256);
        }
        return buf;
    }

    [MetalTestFact]
    public void SingleRow_MatchesCpu()
    {
        const int n = 1, k = 256;
        byte[]  weight   = BuildWeight(n, k, seed: 40);
        Half[]  x        = GemvF16Helpers.BuildX(k, seed: 41);
        Half[]  y        = new Half[n];
        float[] expected = GemvF16Helpers.CpuGemv(weight, x, n, k, QuantizationType.Q5_K);

        using var ctx = new MetalContext();
        QuantizedGemv.Q5_K(ctx, weight, x, y, n, k);

        GemvF16Helpers.AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void MultipleRows_MatchesCpu()
    {
        const int n = 4, k = 512;
        byte[]  weight   = BuildWeight(n, k, seed: 42);
        Half[]  x        = GemvF16Helpers.BuildX(k, seed: 43);
        Half[]  y        = new Half[n];
        float[] expected = GemvF16Helpers.CpuGemv(weight, x, n, k, QuantizationType.Q5_K);

        using var ctx = new MetalContext();
        QuantizedGemv.Q5_K(ctx, weight, x, y, n, k);

        GemvF16Helpers.AssertEqual(expected, y, 5);
    }

    [MetalTestFact]
    public void ZeroInput_OutputIsZero()
    {
        const int n = 2, k = 256;
        byte[] weight = BuildWeight(n, k, seed: 44);
        Half[] x      = new Half[k];
        Half[] y      = new Half[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q5_K(ctx, weight, x, y, n, k);

        foreach (Half v in y) Assert.Equal(0f, (float)v, 1e-4f);
    }
}

// ── Q6_K FP16 I/O ─────────────────────────────────────────────────────────────

public sealed class QuantizedGemvQ6_KTests
{
    // Q6_K: 128 ql + 64 qh + 16 int8 scales + 2 d = 210 bytes per 256 values.
    private static byte[] BuildWeight(int n, int k, int seed)
    {
        const int SuperSize = 256, SuperBytes = 210;
        int sbpr = k / SuperSize;
        byte[] buf = new byte[n * sbpr * SuperBytes];
        var rng = new Random(seed);
        for (int row = 0; row < n; row++)
        for (int sb = 0; sb < sbpr; sb++)
        {
            int off = (row * sbpr + sb) * SuperBytes;
            // ql[0..127], qh[128..191]: random nibble/2-bit data
            for (int j = 0; j < 192; j++)
                buf[off + j] = (byte)rng.Next(256);
            // scales[192..207]: small int8 values
            for (int j = 192; j < 208; j++)
                buf[off + j] = (byte)(sbyte)rng.Next(-8, 8);
            // d at bytes 208-209
            MemoryMarshal.Write(buf.AsSpan(off + 208, 2), (Half)(rng.NextSingle() * 0.1f + 0.01f));
        }
        return buf;
    }

    [MetalTestFact]
    public void SingleRow_MatchesCpu()
    {
        const int n = 1, k = 256;
        byte[]  weight   = BuildWeight(n, k, seed: 50);
        Half[]  x        = GemvF16Helpers.BuildX(k, seed: 51);
        Half[]  y        = new Half[n];
        float[] expected = GemvF16Helpers.CpuGemv(weight, x, n, k, QuantizationType.Q6_K);

        using var ctx = new MetalContext();
        QuantizedGemv.Q6_K(ctx, weight, x, y, n, k);

        GemvF16Helpers.AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void MultipleRows_MatchesCpu()
    {
        const int n = 4, k = 512;
        byte[]  weight   = BuildWeight(n, k, seed: 52);
        Half[]  x        = GemvF16Helpers.BuildX(k, seed: 53);
        Half[]  y        = new Half[n];
        float[] expected = GemvF16Helpers.CpuGemv(weight, x, n, k, QuantizationType.Q6_K);

        using var ctx = new MetalContext();
        QuantizedGemv.Q6_K(ctx, weight, x, y, n, k);

        GemvF16Helpers.AssertEqual(expected, y);
    }

    [MetalTestFact]
    public void ZeroInput_OutputIsZero()
    {
        const int n = 2, k = 256;
        byte[] weight = BuildWeight(n, k, seed: 54);
        Half[] x      = new Half[k];
        Half[] y      = new Half[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q6_K(ctx, weight, x, y, n, k);

        foreach (Half v in y) Assert.Equal(0f, (float)v, 1e-4f);
    }
}
