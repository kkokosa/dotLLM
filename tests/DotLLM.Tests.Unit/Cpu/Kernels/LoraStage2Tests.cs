using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Phase 4d.6 — outer-product stage-2 fast path tests.
///
/// <para>
/// <see cref="LoraStage2.ApplyF32_R16"/> computes
/// <c>y[t, o] += scale × sum_r A[o, r] × tmp[t, r]</c> for <c>rank = 16</c>
/// using a transposed-A layout (<c>[16, outputDim]</c>) and an
/// outer-product accumulator. We compare it against the canonical
/// per-token GEMV+MultiplyAdd reference (the legacy path). FMA reordering
/// is allowed within a tight absolute tolerance.
/// </para>
/// </summary>
public sealed unsafe class LoraStage2Tests
{
    private const int Rank = 16;

    public static TheoryData<int, int> Shapes => new()
    {
        // (seqLen, outputDim) — covers k/v_proj (512), q/o_proj (2048),
        // FFN gate/up_proj (5632) on Llama-3.2-1B and a few stress sizes.
        { 1, 16 },
        { 1, 512 },
        { 4, 32 },
        { 16, 2048 },
        { 64, 1024 },
        { 128, 5632 },
        // outputDim NOT a multiple of 16 — exercises the scalar tail.
        { 8, 23 },
        { 4, 257 },
    };

    [SkipUnlessAvx512Theory]
    [MemberData(nameof(Shapes))]
    public void ApplyF32_R16_MatchesPerTokenGemvReference(int seqLen, int outputDim)
    {
        // Build inputs in F32 native memory.
        long aElems = (long)outputDim * Rank;
        long aTElems = (long)Rank * outputDim;
        long tmpElems = (long)seqLen * Rank;
        long yElems = (long)seqLen * outputDim;

        float* aRowMajor = (float*)NativeMemory.AlignedAlloc((nuint)(aElems * sizeof(float)), 64);
        float* tmp = (float*)NativeMemory.AlignedAlloc((nuint)(tmpElems * sizeof(float)), 64);
        float* yRef = (float*)NativeMemory.AlignedAlloc((nuint)(yElems * sizeof(float)), 64);
        float* yKernel = (float*)NativeMemory.AlignedAlloc((nuint)(yElems * sizeof(float)), 64);
        nint aT = 0;
        try
        {
            var rng = new Random(13 + seqLen * 31 + outputDim);
            for (long i = 0; i < aElems; i++) aRowMajor[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.4f;
            for (long i = 0; i < tmpElems; i++) tmp[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.5f;
            // Non-zero starting y so the read-modify-write semantics are exercised.
            for (long i = 0; i < yElems; i++) yRef[i] = yKernel[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.3f;

            float scale = 0.5f;

            // Reference: per-token GemvF32 + scaled MultiplyAdd into y.
            ApplyReference(aRowMajor, tmp, yRef, seqLen, outputDim, Rank, scale);

            // Build transposed A and dispatch the fast path.
            aT = LoraStage2.BuildATransposedF32(aRowMajor, outputDim, Rank);
            LoraStage2.ApplyF32_R16((float*)aT, tmp, yKernel, seqLen, outputDim, scale);

            // Tight tolerance — both paths use F32 FMA, only multiply ordering differs.
            for (long i = 0; i < yElems; i++)
            {
                float expected = yRef[i];
                float actual = yKernel[i];
                float diff = MathF.Abs(expected - actual);
                float tol = 1e-3f + 1e-4f * MathF.Abs(expected);
                Assert.True(diff <= tol,
                    $"Mismatch at flat-idx {i}: expected={expected:G6} actual={actual:G6} diff={diff:G6} tol={tol:G6}");
            }
        }
        finally
        {
            if (aT != 0) NativeMemory.AlignedFree((void*)aT);
            NativeMemory.AlignedFree(yKernel);
            NativeMemory.AlignedFree(yRef);
            NativeMemory.AlignedFree(tmp);
            NativeMemory.AlignedFree(aRowMajor);
        }
    }

    /// <summary>
    /// Verifies the dtype-aware transposed-A builder for F16 / BF16 sources
    /// produces a layout that yields the same kernel result (within dtype
    /// tolerance) as F32-direct.
    /// </summary>
    [SkipUnlessAvx512Fact]
    public void BuildATransposedF32FromDType_F16_MatchesF32WithinTolerance()
    {
        const int seqLen = 8;
        const int outputDim = 1024;
        long aElems = (long)outputDim * Rank;

        float* aF32 = (float*)NativeMemory.AlignedAlloc((nuint)(aElems * sizeof(float)), 64);
        Half* aF16 = (Half*)NativeMemory.AlignedAlloc((nuint)(aElems * sizeof(Half)), 64);
        float* tmp = (float*)NativeMemory.AlignedAlloc((nuint)((long)seqLen * Rank * sizeof(float)), 64);
        float* yF32 = (float*)NativeMemory.AlignedAlloc((nuint)((long)seqLen * outputDim * sizeof(float)), 64);
        float* yF16 = (float*)NativeMemory.AlignedAlloc((nuint)((long)seqLen * outputDim * sizeof(float)), 64);
        nint aT_F32 = 0, aT_F16 = 0;
        try
        {
            var rng = new Random(2026);
            for (long i = 0; i < aElems; i++)
            {
                float v = ((float)rng.NextDouble() * 2f - 1f) * 0.4f;
                aF32[i] = v;
                aF16[i] = (Half)v;
            }
            for (long i = 0; i < seqLen * Rank; i++) tmp[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.5f;
            for (long i = 0; i < seqLen * outputDim; i++) yF32[i] = yF16[i] = 0;

            aT_F32 = LoraStage2.BuildATransposedF32FromDType(
                (nint)aF32, DotLLM.Core.Lora.LoraWeightDType.F32, outputDim, Rank);
            aT_F16 = LoraStage2.BuildATransposedF32FromDType(
                (nint)aF16, DotLLM.Core.Lora.LoraWeightDType.F16, outputDim, Rank);

            LoraStage2.ApplyF32_R16((float*)aT_F32, tmp, yF32, seqLen, outputDim, scale: 0.5f);
            LoraStage2.ApplyF32_R16((float*)aT_F16, tmp, yF16, seqLen, outputDim, scale: 0.5f);

            // F16 → F32 round-trip introduces ~1e-3 relative error on each
            // weight; accumulated over rank=16 sums that's ~5e-3 abs at
            // typical magnitudes. Tolerance bound matches the existing
            // F16 LoRA dtype tests.
            long elems = (long)seqLen * outputDim;
            for (long i = 0; i < elems; i++)
            {
                float diff = MathF.Abs(yF32[i] - yF16[i]);
                float tol = 1e-2f + 1e-2f * MathF.Abs(yF32[i]);
                Assert.True(diff <= tol,
                    $"F16 vs F32 mismatch at idx {i}: f32={yF32[i]:G6} f16={yF16[i]:G6} diff={diff:G6}");
            }
        }
        finally
        {
            if (aT_F16 != 0) NativeMemory.AlignedFree((void*)aT_F16);
            if (aT_F32 != 0) NativeMemory.AlignedFree((void*)aT_F32);
            NativeMemory.AlignedFree(yF16);
            NativeMemory.AlignedFree(yF32);
            NativeMemory.AlignedFree(tmp);
            NativeMemory.AlignedFree(aF16);
            NativeMemory.AlignedFree(aF32);
        }
    }

    [Fact]
    public void BuildATransposedF32_LayoutIsCorrect()
    {
        // Tiny shape — verify the byte-level layout matches the contract:
        // dst[r, o] = src[o, r] for r ∈ [0, rank), o ∈ [0, outputDim).
        const int rank = 16;
        const int outputDim = 5;
        long elems = (long)outputDim * rank;
        float* aRowMajor = (float*)NativeMemory.AlignedAlloc((nuint)(elems * sizeof(float)), 64);
        nint aT = 0;
        try
        {
            for (long i = 0; i < elems; i++) aRowMajor[i] = i + 0.5f;

            aT = LoraStage2.BuildATransposedF32(aRowMajor, outputDim, rank);
            float* d = (float*)aT;

            for (int r = 0; r < rank; r++)
                for (int o = 0; o < outputDim; o++)
                {
                    float expected = aRowMajor[o * rank + r];
                    float actual = d[r * outputDim + o];
                    Assert.Equal(expected, actual);
                }
        }
        finally
        {
            if (aT != 0) NativeMemory.AlignedFree((void*)aT);
            NativeMemory.AlignedFree(aRowMajor);
        }
    }

    private static void ApplyReference(
        float* aRowMajor, float* tmp, float* y,
        int seqLen, int outputDim, int rank, float scale)
    {
        // Production legacy stage-2: per token GemvF32(A, tmp_t, delta) then
        // y += scale * delta. We inline the GemvF32 with TensorPrimitives.Dot
        // to stay independent of MatMul.cs internals.
        float[] deltaBuf = ArrayPool<float>.Shared.Rent(outputDim);
        try
        {
            for (int t = 0; t < seqLen; t++)
            {
                fixed (float* delta = deltaBuf)
                {
                    var xSpan = new ReadOnlySpan<float>(tmp + t * rank, rank);
                    for (int o = 0; o < outputDim; o++)
                    {
                        var rowSpan = new ReadOnlySpan<float>(aRowMajor + o * rank, rank);
                        delta[o] = TensorPrimitives.Dot(rowSpan, xSpan);
                    }
                    var deltaSpan = new ReadOnlySpan<float>(delta, outputDim);
                    var ySpan = new Span<float>(y + t * outputDim, outputDim);
                    TensorPrimitives.MultiplyAdd(deltaSpan, scale, ySpan, ySpan);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(deltaBuf);
        }
    }
}

/// <summary>Skip the test class on hardware without AVX-512.</summary>
internal sealed class SkipUnlessAvx512FactAttribute : FactAttribute
{
    public SkipUnlessAvx512FactAttribute()
    {
        if (!System.Runtime.Intrinsics.X86.Avx512F.IsSupported)
            Skip = "Requires AVX-512F.";
    }
}

internal sealed class SkipUnlessAvx512TheoryAttribute : TheoryAttribute
{
    public SkipUnlessAvx512TheoryAttribute()
    {
        if (!System.Runtime.Intrinsics.X86.Avx512F.IsSupported)
            Skip = "Requires AVX-512F.";
    }
}
