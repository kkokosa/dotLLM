using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public sealed unsafe class DequantizeTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    // ──────────────────── FP16 ────────────────────

    [Fact]
    public void Fp16_KnownValues_MatchExpected()
    {
        Half[] input = [Half.Zero, (Half)1.0f, (Half)(-2.5f), Half.MaxValue];
        float[] expected = input.Select(h => (float)h).ToArray();

        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(input.Length * sizeof(Half)), 32);
        try
        {
            input.AsSpan().CopyTo(new Span<Half>((void*)ptr, input.Length));
            float[] dest = new float[input.Length];

            Dequantize.ToFloat32(ptr, input.Length, QuantizationType.F16, dest);

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], dest[i]);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Fp16_AllZeros_ProducesZeros()
    {
        const int count = 64;
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(Half)), 32);
        try
        {
            NativeMemory.Clear((void*)ptr, (nuint)(count * sizeof(Half)));
            float[] dest = new float[count];

            Dequantize.ToFloat32(ptr, count, QuantizationType.F16, dest);

            Assert.All(dest, v => Assert.Equal(0f, v));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Fp16_NegativeValues_Correct()
    {
        Half[] input = [(Half)(-1.0f), (Half)(-0.5f), (Half)(-100.0f)];

        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(input.Length * sizeof(Half)), 32);
        try
        {
            input.AsSpan().CopyTo(new Span<Half>((void*)ptr, input.Length));
            float[] dest = new float[input.Length];

            Dequantize.ToFloat32(ptr, input.Length, QuantizationType.F16, dest);

            for (int i = 0; i < input.Length; i++)
                Assert.Equal((float)input[i], dest[i]);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── Q8_0 ────────────────────

    [Fact]
    public void Q8_0_SingleBlock_HandCalculated()
    {
        // scale = 0.5, qs = [0, 1, 2, ..., 31]
        // expected: [0.0, 0.5, 1.0, 1.5, ..., 15.5]
        nint ptr = AllocQ8_0Block(scale: (Half)0.5f, fillQs: i => (sbyte)i);
        try
        {
            float[] dest = new float[Q8_0GroupSize];
            Dequantize.ToFloat32(ptr, Q8_0GroupSize, QuantizationType.Q8_0, dest);

            for (int i = 0; i < Q8_0GroupSize; i++)
                Assert.Equal(0.5f * i, dest[i], 1e-3f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q8_0_ScaleZero_AllZeros()
    {
        nint ptr = AllocQ8_0Block(scale: Half.Zero, fillQs: i => (sbyte)(i + 1));
        try
        {
            float[] dest = new float[Q8_0GroupSize];
            Dequantize.ToFloat32(ptr, Q8_0GroupSize, QuantizationType.Q8_0, dest);

            Assert.All(dest, v => Assert.Equal(0f, v));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q8_0_MultipleBlocks_DifferentScales()
    {
        const int blockCount = 4;
        const int totalElements = blockCount * Q8_0GroupSize;
        nuint totalBytes = (nuint)(blockCount * Q8_0BlockBytes);

        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 32);
        try
        {
            byte* p = (byte*)ptr;
            for (int b = 0; b < blockCount; b++)
            {
                Half scale = (Half)(b + 1.0f); // scales: 1, 2, 3, 4
                *(Half*)p = scale;
                for (int i = 0; i < Q8_0GroupSize; i++)
                    ((sbyte*)(p + 2))[i] = 1; // all qs = 1
                p += Q8_0BlockBytes;
            }

            float[] dest = new float[totalElements];
            Dequantize.ToFloat32(ptr, totalElements, QuantizationType.Q8_0, dest);

            for (int b = 0; b < blockCount; b++)
            {
                float expectedScale = b + 1.0f;
                for (int i = 0; i < Q8_0GroupSize; i++)
                    Assert.Equal(expectedScale, dest[b * Q8_0GroupSize + i], 1e-3f);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q8_0_MaxValues_NoOverflow()
    {
        // scale = 1.0, qs = 127 (sbyte max) → output = 127.0
        nint ptr = AllocQ8_0Block(scale: (Half)1.0f, fillQs: _ => sbyte.MaxValue);
        try
        {
            float[] dest = new float[Q8_0GroupSize];
            Dequantize.ToFloat32(ptr, Q8_0GroupSize, QuantizationType.Q8_0, dest);

            Assert.All(dest, v => Assert.Equal(127f, v, 1e-3f));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q8_0_NegativeQs_Correct()
    {
        // scale = 2.0, qs = -1 → output = -2.0
        nint ptr = AllocQ8_0Block(scale: (Half)2.0f, fillQs: _ => (sbyte)-1);
        try
        {
            float[] dest = new float[Q8_0GroupSize];
            Dequantize.ToFloat32(ptr, Q8_0GroupSize, QuantizationType.Q8_0, dest);

            Assert.All(dest, v => Assert.Equal(-2.0f, v, 1e-3f));
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q8_0_ScalarMatchesSimd_RandomBlocks()
    {
        const int blockCount = 16;
        const int totalElements = blockCount * Q8_0GroupSize;
        nuint totalBytes = (nuint)(blockCount * Q8_0BlockBytes);

        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            // Fill with pseudo-random data.
            var rng = new Random(42);
            byte* p = (byte*)ptr;
            for (int b = 0; b < blockCount; b++)
            {
                *(Half*)p = (Half)(rng.NextSingle() * 2.0f - 1.0f);
                for (int i = 0; i < Q8_0GroupSize; i++)
                    ((sbyte*)(p + 2))[i] = (sbyte)rng.Next(-128, 128);
                p += Q8_0BlockBytes;
            }

            float[] scalarDest = new float[totalElements];
            float[] simdDest = new float[totalElements];

            Dequantize.DequantizeQ8_0Scalar(ptr, totalElements, scalarDest);

            if (Avx2.IsSupported)
            {
                Dequantize.DequantizeQ8_0Avx2(ptr, totalElements, simdDest);

                for (int i = 0; i < totalElements; i++)
                    Assert.Equal(scalarDest[i], simdDest[i], 1e-5f);
            }

            // Also verify dispatch path matches scalar.
            float[] dispatchDest = new float[totalElements];
            Dequantize.ToFloat32(ptr, totalElements, QuantizationType.Q8_0, dispatchDest);

            for (int i = 0; i < totalElements; i++)
                Assert.Equal(scalarDest[i], dispatchDest[i], 1e-5f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── F32 ────────────────────

    [Fact]
    public void F32_CopiesDirectly()
    {
        float[] input = [1.0f, -2.5f, 0f, float.MaxValue];
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(input.Length * sizeof(float)), 32);
        try
        {
            input.AsSpan().CopyTo(new Span<float>((void*)ptr, input.Length));
            float[] dest = new float[input.Length];

            Dequantize.ToFloat32(ptr, input.Length, QuantizationType.F32, dest);

            for (int i = 0; i < input.Length; i++)
                Assert.Equal(input[i], dest[i]);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    // ──────────────────── Dispatch ────────────────────

    [Fact]
    public void UnsupportedType_Throws()
    {
        float[] dest = new float[32];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Dequantize.ToFloat32(nint.Zero, 32, QuantizationType.Q4_0, dest));
    }

    [Fact]
    public void DestTooSmall_Throws()
    {
        float[] dest = new float[1];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 32, QuantizationType.F32, dest));
    }

    // ──────────────────── Helpers ────────────────────

    private static nint AllocQ8_0Block(Half scale, Func<int, sbyte> fillQs)
    {
        nint ptr = (nint)NativeMemory.AlignedAlloc(Q8_0BlockBytes, 32);
        byte* p = (byte*)ptr;
        *(Half*)p = scale;
        for (int i = 0; i < Q8_0GroupSize; i++)
            ((sbyte*)(p + 2))[i] = fillQs(i);
        return ptr;
    }
}
