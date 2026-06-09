using System.Buffers.Binary;
using DotLLM.Core.Lora;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lora;

/// <summary>
/// Phase 4d.1 — Quantised LoRA adapter weight parity. Verifies that
/// dispatching <see cref="LoraDelta"/> with F16 / BF16 buffers matches the
/// F32 reference within (abs 5e-3, rel 1e-3).
/// </summary>
public sealed unsafe class LoraDeltaQuantizedDtypeTests
{
    private const int SeqLen = 4;
    private const int InputDim = 32;
    private const int OutputDim = 24;
    private const int Rank = 8;
    private const float Scale = 0.5f;

    [Fact]
    public void F16_MatchesF32Reference()
    {
        var (x, bF32, aF32, y0) = BuildVectors();

        // Reference: F32 path.
        var yRef = (float[])y0.Clone();
        ApplyF32(x, bF32, aF32, yRef);

        // Quantised: F16 storage, dequant on read.
        var bF16 = ToHalf(bF32);
        var aF16 = ToHalf(aF32);
        var yKernel = (float[])y0.Clone();
        unsafe
        {
            fixed (float* xp = x)
            fixed (Half* bp = bF16)
            fixed (Half* ap = aF16)
            fixed (float* yp = yKernel)
            {
                LoraDelta.Apply(xp, (void*)bp, (void*)ap, yp,
                    SeqLen, InputDim, OutputDim, Rank, Scale,
                    LoraWeightDType.F16, LoraWeightDType.F16);
            }
        }

        AssertClose(yRef, yKernel, absTol: 5e-3f, relTol: 1e-3f);
    }

    [Fact]
    public void BF16_MatchesF32Reference()
    {
        var (x, bF32, aF32, y0) = BuildVectors();

        var yRef = (float[])y0.Clone();
        ApplyF32(x, bF32, aF32, yRef);

        var bBf16 = ToBF16(bF32);
        var aBf16 = ToBF16(aF32);
        var yKernel = (float[])y0.Clone();
        unsafe
        {
            fixed (float* xp = x)
            fixed (byte* bp = bBf16)
            fixed (byte* ap = aBf16)
            fixed (float* yp = yKernel)
            {
                LoraDelta.Apply(xp, (void*)bp, (void*)ap, yp,
                    SeqLen, InputDim, OutputDim, Rank, Scale,
                    LoraWeightDType.BF16, LoraWeightDType.BF16);
            }
        }

        // BF16 has only ~3 decimal digits of mantissa precision, so we
        // relax the absolute tolerance vs F16.
        AssertClose(yRef, yKernel, absTol: 5e-2f, relTol: 5e-2f);
    }

    [Fact]
    public void F32DispatchedThroughDTypeOverload_IsByteEquivalent()
    {
        // Pure F32 should be bit-equivalent to the legacy overload — the new
        // dtype-aware overload short-circuits to the same kernel.
        var (x, bF32, aF32, y0) = BuildVectors();

        var yLegacy = (float[])y0.Clone();
        ApplyF32(x, bF32, aF32, yLegacy);

        var yNew = (float[])y0.Clone();
        unsafe
        {
            fixed (float* xp = x)
            fixed (float* bp = bF32)
            fixed (float* ap = aF32)
            fixed (float* yp = yNew)
            {
                LoraDelta.Apply(xp, (void*)bp, (void*)ap, yp,
                    SeqLen, InputDim, OutputDim, Rank, Scale,
                    LoraWeightDType.F32, LoraWeightDType.F32);
            }
        }

        for (int i = 0; i < yLegacy.Length; i++)
            Assert.Equal(yLegacy[i], yNew[i]);
    }

    private static (float[] X, float[] B, float[] A, float[] Y) BuildVectors()
    {
        var rng = new Random(7);
        var x = RandomVec(rng, SeqLen * InputDim, 0.4f);
        var b = RandomVec(rng, Rank * InputDim, 0.4f);
        var a = RandomVec(rng, OutputDim * Rank, 0.4f);
        var y = RandomVec(rng, SeqLen * OutputDim, 0.4f);
        return (x, b, a, y);
    }

    private static void ApplyF32(float[] x, float[] b, float[] a, float[] y)
    {
        unsafe
        {
            fixed (float* xp = x)
            fixed (float* bp = b)
            fixed (float* ap = a)
            fixed (float* yp = y)
            {
                LoraDelta.Apply(xp, bp, ap, yp, SeqLen, InputDim, OutputDim, Rank, Scale);
            }
        }
    }

    private static float[] RandomVec(Random rng, int n, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
        return v;
    }

    private static Half[] ToHalf(float[] src)
    {
        var dst = new Half[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (Half)src[i];
        return dst;
    }

    private static byte[] ToBF16(float[] src)
    {
        var dst = new byte[src.Length * 2];
        for (int i = 0; i < src.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(src[i]);
            ushort top = (ushort)(bits >> 16);
            BinaryPrimitives.WriteUInt16LittleEndian(dst.AsSpan(i * 2, 2), top);
        }
        return dst;
    }

    private static void AssertClose(float[] expected, float[] actual, float absTol, float relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float tol = absTol + relTol * MathF.Abs(expected[i]);
            Assert.True(diff <= tol,
                $"Mismatch at [{i}]: expected={expected[i]:G6} actual={actual[i]:G6} diff={diff:G6} tol={tol:G6}");
        }
    }
}
