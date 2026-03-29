using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class KvQuantizeTests
{
    private const int BlockSize = 32;

    [Fact]
    public void Q4_0_Scalar_RoundTrip_WithinTolerance()
    {
        float[] input = GenerateTestData(BlockSize * 4);
        byte[] quantized = new byte[KvQuantize.Q4_0BlockBytes * 4];
        float[] output = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* op = output)
        {
            KvQuantize.F32ToQ4_0Scalar(ip, qp, input.Length);
            KvQuantize.Q4_0ToF32Scalar(qp, op, input.Length);
        }

        float maxErr = 0;
        for (int i = 0; i < input.Length; i++)
        {
            float err = MathF.Abs(input[i] - output[i]);
            if (err > maxErr) maxErr = err;
        }

        // Q4_0 has 4-bit precision — expect ~1/7 of max value error per element
        float maxVal = input.Max(MathF.Abs);
        float expectedMaxErr = maxVal / 7.0f; // one quantization step
        Assert.True(maxErr <= expectedMaxErr * 1.1f,
            $"Max round-trip error {maxErr} exceeds expected {expectedMaxErr}");
    }

    [Fact]
    public void Q4_0_Avx2_MatchesScalar()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            return; // Skip on non-AVX2 hardware

        float[] input = GenerateTestData(BlockSize * 8);
        byte[] quantScalar = new byte[KvQuantize.Q4_0BlockBytes * 8];
        byte[] quantAvx2 = new byte[KvQuantize.Q4_0BlockBytes * 8];

        fixed (float* ip = input)
        fixed (byte* sp = quantScalar)
        fixed (byte* ap = quantAvx2)
        {
            KvQuantize.F32ToQ4_0Scalar(ip, sp, input.Length);
            KvQuantize.F32ToQ4_0Avx2(ip, ap, input.Length);
        }

        Assert.Equal(quantScalar, quantAvx2);
    }

    [Fact]
    public void Q4_0_ZeroInput_AllZeroOutput()
    {
        float[] input = new float[BlockSize];
        byte[] quantized = new byte[KvQuantize.Q4_0BlockBytes];
        float[] output = new float[BlockSize];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* op = output)
        {
            KvQuantize.F32ToQ4_0(ip, qp, BlockSize);
            KvQuantize.Q4_0ToF32(qp, op, BlockSize);
        }

        for (int i = 0; i < BlockSize; i++)
            Assert.Equal(0f, output[i]);
    }

    [Fact]
    public void Q4_0_Dequant_Avx2_MatchesScalar()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            return;

        float[] input = GenerateTestData(BlockSize * 4);
        byte[] quantized = new byte[KvQuantize.Q4_0BlockBytes * 4];
        float[] outScalar = new float[input.Length];
        float[] outAvx2 = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        {
            KvQuantize.F32ToQ4_0Scalar(ip, qp, input.Length);
        }

        fixed (byte* qp = quantized)
        fixed (float* sp = outScalar)
        fixed (float* ap = outAvx2)
        {
            KvQuantize.Q4_0ToF32Scalar(qp, sp, input.Length);
            KvQuantize.Q4_0ToF32Avx2(qp, ap, input.Length);
        }

        for (int i = 0; i < input.Length; i++)
            Assert.Equal(outScalar[i], outAvx2[i], 5);
    }

    [Fact]
    public void Q8_0_RoundTrip_WithinTolerance()
    {
        float[] input = GenerateTestData(BlockSize * 4);
        byte[] quantized = new byte[KvQuantize.Q8_0BlockBytes * 4];
        float[] output = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        fixed (float* op = output)
        {
            KvQuantize.F32ToQ8_0(ip, qp, input.Length);
            KvQuantize.Q8_0ToF32(qp, op, input.Length);
        }

        float maxErr = 0;
        for (int i = 0; i < input.Length; i++)
        {
            float err = MathF.Abs(input[i] - output[i]);
            if (err > maxErr) maxErr = err;
        }

        // Q8_0 has 8-bit precision — much tighter tolerance
        float maxVal = input.Max(MathF.Abs);
        float expectedMaxErr = maxVal / 127.0f;
        Assert.True(maxErr <= expectedMaxErr * 1.1f,
            $"Max round-trip error {maxErr} exceeds expected {expectedMaxErr}");
    }

    [Fact]
    public void Q8_0_Dequant_Avx2_MatchesScalar()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            return;

        float[] input = GenerateTestData(BlockSize * 4);
        byte[] quantized = new byte[KvQuantize.Q8_0BlockBytes * 4];
        float[] outScalar = new float[input.Length];
        float[] outAvx2 = new float[input.Length];

        fixed (float* ip = input)
        fixed (byte* qp = quantized)
        {
            KvQuantize.F32ToQ8_0(ip, qp, input.Length);
        }

        fixed (byte* qp = quantized)
        fixed (float* sp = outScalar)
        fixed (float* ap = outAvx2)
        {
            KvQuantize.Q8_0ToF32Scalar(qp, sp, input.Length);
            KvQuantize.Q8_0ToF32Avx2(qp, ap, input.Length);
        }

        for (int i = 0; i < input.Length; i++)
            Assert.Equal(outScalar[i], outAvx2[i], 5);
    }

    [Fact]
    public void QuantizedRowBytes_ReturnsCorrectSize()
    {
        int kvStride = 1024;
        Assert.Equal(kvStride / BlockSize * KvQuantize.Q8_0BlockBytes,
            KvQuantize.QuantizedRowBytes(kvStride, Core.Configuration.KvCacheDType.Q8_0));
        Assert.Equal(kvStride / BlockSize * KvQuantize.Q4_0BlockBytes,
            KvQuantize.QuantizedRowBytes(kvStride, Core.Configuration.KvCacheDType.Q4_0));
    }

    private static float[] GenerateTestData(int count)
    {
        var rng = new Random(42);
        var data = new float[count];
        for (int i = 0; i < count; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1) * 5.0f; // range [-5, 5]
        return data;
    }
}
