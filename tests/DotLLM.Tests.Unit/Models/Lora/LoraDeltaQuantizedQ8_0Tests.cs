using DotLLM.Core.Lora;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lora;

/// <summary>
/// Phase 4d.4 — Q8_0 LoRA-B parity tests. Verifies that dispatching
/// <see cref="LoraDelta"/> with a Q8_0 B buffer matches the F32 reference
/// within Q8_0 quantisation tolerance.
/// </summary>
/// <remarks>
/// <para>
/// Q8_0 stores 32 elements per block with one shared F16 scale, so each
/// element has at most <c>scale × 1</c> rounding error (sub-1% of the
/// block max-abs). For LoRA-B with row max-abs ~0.4 in a small synthetic
/// adapter, the worst-case absolute error per output element is bounded
/// by <c>scale × inputDim × max|x|</c> — a few percent of typical y
/// magnitudes. We therefore relax the absolute tolerance vs the F16 test:
/// 5e-2 abs / 5e-2 rel.
/// </para>
/// <para>
/// We also exercise the Q8_0-B + F16-A combination (the production case
/// per the spike design) and the Q8_0-B + F32-A combination (the
/// dispatch fast-path that avoids per-call A dequant).
/// </para>
/// </remarks>
public sealed unsafe class LoraDeltaQuantizedQ8_0Tests
{
    private const int SeqLen = 4;
    private const int InputDim = 128;     // multiple of 32 — required for Q8_0
    private const int OutputDim = 24;
    private const int Rank = 8;
    private const float Scale = 0.5f;

    [Fact]
    public void Q8_0B_F32A_MatchesF32Reference()
    {
        var (x, bF32, aF32, y0) = BuildVectors();

        var yRef = (float[])y0.Clone();
        ApplyF32(x, bF32, aF32, yRef);

        // Quantise B to Q8_0 (rank rows of inputDim elements each).
        var bQ8 = new byte[Rank * (InputDim / 32) * 34];
        fixed (float* bp = bF32)
        fixed (byte* bqp = bQ8)
        {
            LoraDelta.Quantize_F32_To_Q8_0(bp, bqp, Rank, InputDim);
        }

        var yKernel = (float[])y0.Clone();
        fixed (float* xp = x)
        fixed (byte* bqp = bQ8)
        fixed (float* ap = aF32)
        fixed (float* yp = yKernel)
        {
            LoraDelta.Apply(xp, (void*)bqp, (void*)ap, yp,
                SeqLen, InputDim, OutputDim, Rank, Scale,
                LoraWeightDType.Q8_0, LoraWeightDType.F32);
        }

        AssertClose(yRef, yKernel, absTol: 5e-2f, relTol: 5e-2f);
    }

    [Fact]
    public void Q8_0B_F16A_MatchesF32Reference()
    {
        var (x, bF32, aF32, y0) = BuildVectors();

        var yRef = (float[])y0.Clone();
        ApplyF32(x, bF32, aF32, yRef);

        var bQ8 = new byte[Rank * (InputDim / 32) * 34];
        fixed (float* bp = bF32)
        fixed (byte* bqp = bQ8)
        {
            LoraDelta.Quantize_F32_To_Q8_0(bp, bqp, Rank, InputDim);
        }

        var aF16 = new Half[aF32.Length];
        for (int i = 0; i < aF32.Length; i++) aF16[i] = (Half)aF32[i];

        var yKernel = (float[])y0.Clone();
        fixed (float* xp = x)
        fixed (byte* bqp = bQ8)
        fixed (Half* ap = aF16)
        fixed (float* yp = yKernel)
        {
            LoraDelta.Apply(xp, (void*)bqp, (void*)ap, yp,
                SeqLen, InputDim, OutputDim, Rank, Scale,
                LoraWeightDType.Q8_0, LoraWeightDType.F16);
        }

        AssertClose(yRef, yKernel, absTol: 5e-2f, relTol: 5e-2f);
    }

    [Fact]
    public void Quantize_RoundTrip_PreservesRowsWithinTolerance()
    {
        // Round-trip: F32 row -> Q8_0 -> F32 should match within scale-bounded error.
        var rng = new Random(11);
        var src = RandomVec(rng, InputDim, 0.4f);

        var q8 = new byte[(InputDim / 32) * 34];
        var dst = new float[InputDim];

        fixed (float* sp = src)
        fixed (byte* qp = q8)
        fixed (float* dp = dst)
        {
            LoraDelta.Quantize_F32_To_Q8_0(sp, qp, rows: 1, elementsPerRow: InputDim);
            LoraDelta.DequantizeRowToF32(qp, dp, InputDim);
        }

        // Per-block Q8_0 max error is scale * 0.5 ≈ (max_abs_block / 127) * 0.5.
        // For src in [-0.4, 0.4], max scale ≈ 3.15e-3, so per-element abs error
        // < ~1.6e-3. Add headroom for occasional worst-case: 5e-3.
        for (int i = 0; i < InputDim; i++)
        {
            float diff = MathF.Abs(src[i] - dst[i]);
            Assert.True(diff <= 5e-3f,
                $"Q8_0 round-trip mismatch at [{i}]: src={src[i]:G6} dst={dst[i]:G6} diff={diff:G6}");
        }
    }

    [Fact]
    public void Quantize_RejectsNonMultipleOf32()
    {
        var src = new float[33];
        var dst = new byte[64];
        Assert.Throws<ArgumentException>(() =>
        {
            unsafe
            {
                fixed (float* sp = src)
                fixed (byte* dp = dst)
                {
                    LoraDelta.Quantize_F32_To_Q8_0(sp, dp, rows: 1, elementsPerRow: 33);
                }
            }
        });
    }

    [Fact]
    public void ApplyQ8_0B_RejectsNonMultipleOf32_InputDim()
    {
        // Hot-path defensive check — Q8_0 B requires inputDim multiple of 32.
        var x = new float[SeqLen * 30];
        var bQ8 = new byte[8 * 34];
        var aF32 = new float[OutputDim * Rank];
        var y = new float[SeqLen * OutputDim];

        Assert.Throws<ArgumentException>(() =>
        {
            unsafe
            {
                fixed (float* xp = x)
                fixed (byte* bqp = bQ8)
                fixed (float* ap = aF32)
                fixed (float* yp = y)
                {
                    LoraDelta.Apply(xp, (void*)bqp, (void*)ap, yp,
                        SeqLen, inputDim: 30, OutputDim, Rank, Scale,
                        LoraWeightDType.Q8_0, LoraWeightDType.F32);
                }
            }
        });
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
