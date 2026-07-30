using DotLLM.Metal;
using Xunit;
using Convert = DotLLM.Metal.Kernels.Convert;

namespace DotLLM.Tests.Unit.Metal;

public sealed class ConvertTests
{
    // ── F16 → F32 ────────────────────────────────────────────────────────────

    [MetalTestFact]
    public void F16ToF32_KnownValues_ExactMatch()
    {
        // float16 → float32 is lossless: every half value is representable as float.
        Half[] src = [(Half)1.0f, (Half)(-2.5f), (Half)0.0f, (Half)100.0f];
        float[] dst = new float[src.Length];

        using var ctx = new MetalContext();
        Convert.F16ToF32(ctx, src, dst, src.Length);

        for (int i = 0; i < src.Length; i++)
            Assert.Equal((float)src[i], dst[i]);
    }

    [MetalTestFact]
    public void F16ToF32_RandomValues_MatchCpuCast()
    {
        var rng = new Random(1);
        Half[]  src = new Half[64];
        float[] dst = new float[64];
        for (int i = 0; i < src.Length; i++)
            src[i] = (Half)(rng.NextSingle() * 10f - 5f);

        using var ctx = new MetalContext();
        Convert.F16ToF32(ctx, src, dst, src.Length);

        for (int i = 0; i < src.Length; i++)
            Assert.Equal((float)src[i], dst[i]); // exact — no tolerance needed
    }

    // ── F32 → F16 ────────────────────────────────────────────────────────────

    [MetalTestFact]
    public void F32ToF16_KnownValues_WithinHalfPrecision()
    {
        // float32 → float16 loses precision. Round-trip via CPU cast is the reference.
        float[] src = [1.0f, -2.5f, 0.0f, 100.0f, 3.14159f];
        Half[]  dst = new Half[src.Length];

        using var ctx = new MetalContext();
        Convert.F32ToF16(ctx, src, dst, src.Length);

        for (int i = 0; i < src.Length; i++)
            Assert.Equal((Half)src[i], dst[i]); // Metal and CPU use the same rounding
    }

    [MetalTestFact]
    public void F32ToF16_RandomValues_MatchCpuCast()
    {
        var rng = new Random(2);
        float[] src = new float[64];
        Half[]  dst = new Half[64];
        for (int i = 0; i < src.Length; i++)
            src[i] = rng.NextSingle() * 10f - 5f;

        using var ctx = new MetalContext();
        Convert.F32ToF16(ctx, src, dst, src.Length);

        for (int i = 0; i < src.Length; i++)
            Assert.Equal((Half)src[i], dst[i]);
    }

    // ── Round-trip F32 → F16 → F32 ───────────────────────────────────────────

    [MetalTestFact]
    public void RoundTrip_F32ToF16ToF32_WithinHalfPrecision()
    {
        // After a round-trip, the error is at most the float16 precision (~1/1000).
        var rng = new Random(3);
        float[] src    = new float[64];
        Half[]  mid    = new Half[64];
        float[] result = new float[64];
        for (int i = 0; i < src.Length; i++)
            src[i] = rng.NextSingle() * 2f - 1f; // stay within safe half range

        using var ctx = new MetalContext();
        Convert.F32ToF16(ctx, src, mid,    src.Length);
        Convert.F16ToF32(ctx, mid, result, src.Length);

        for (int i = 0; i < src.Length; i++)
            Assert.Equal((float)(Half)src[i], result[i]); // reference: CPU round-trip
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    [MetalTestFact]
    public void F32ToF16_Zero_IsExact()
    {
        float[] src = [0f, -0f];
        Half[]  dst = new Half[2];

        using var ctx = new MetalContext();
        Convert.F32ToF16(ctx, src, dst, 2);

        Assert.Equal((Half)0f,  dst[0]);
        Assert.Equal((Half)(-0f), dst[1]);
    }

    [MetalTestFact]
    public void F16ToF32_LargeArray_MatchesCpuCast()
    {
        const int n = 1024;
        var rng = new Random(42);
        Half[]  src = new Half[n];
        float[] dst = new float[n];
        for (int i = 0; i < n; i++)
            src[i] = (Half)(rng.NextSingle() * 20f - 10f);

        using var ctx = new MetalContext();
        Convert.F16ToF32(ctx, src, dst, n);

        for (int i = 0; i < n; i++)
            Assert.Equal((float)src[i], dst[i]);
    }
}
