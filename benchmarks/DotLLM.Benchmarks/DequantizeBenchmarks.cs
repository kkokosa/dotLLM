using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Benchmarks;

/// <summary>
/// Benchmarks for <see cref="Dequantize"/> kernels.
/// <para>
/// The <see cref="MemoryDiagnoserAttribute"/> column <c>Allocated</c> must read <c>0 B</c>
/// for all benchmarks — dequantization has zero managed allocations on the hot path.
/// </para>
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class DequantizeBenchmarks
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int Q5_0BlockBytes = 22;
    private const int Q5_0GroupSize = 32;

    private nint _fp16Src;
    private nint _q8Src;
    private nint _q5_0Src;
    private nint _f32Src;
    private float[] _dest = null!;

    /// <summary>Number of elements to dequantize per benchmark iteration.</summary>
    [Params(4096, 32768)]
    public int ElementCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        // FP16 source: ElementCount × 2 bytes
        _fp16Src = (nint)NativeMemory.AlignedAlloc((nuint)(ElementCount * sizeof(Half)), 64);
        var fp16Span = new Span<Half>((void*)_fp16Src, ElementCount);
        for (int i = 0; i < ElementCount; i++)
            fp16Span[i] = (Half)(rng.NextSingle() * 2f - 1f);

        // Q8_0 source: (ElementCount / 32) blocks × 34 bytes
        int blockCount = ElementCount / Q8_0GroupSize;
        nuint q8Bytes = (nuint)(blockCount * Q8_0BlockBytes);
        _q8Src = (nint)NativeMemory.AlignedAlloc(q8Bytes, 64);
        byte* p = (byte*)_q8Src;
        for (int b = 0; b < blockCount; b++)
        {
            *(Half*)p = (Half)(rng.NextSingle() * 0.1f);
            for (int i = 0; i < Q8_0GroupSize; i++)
                ((sbyte*)(p + 2))[i] = (sbyte)rng.Next(-128, 128);
            p += Q8_0BlockBytes;
        }

        // Q5_0 source: (ElementCount / 32) blocks × 22 bytes
        int q5BlockCount = ElementCount / Q5_0GroupSize;
        nuint q5Bytes = (nuint)(q5BlockCount * Q5_0BlockBytes);
        _q5_0Src = (nint)NativeMemory.AlignedAlloc(q5Bytes, 64);
        byte* q5p = (byte*)_q5_0Src;
        for (int b = 0; b < q5BlockCount; b++)
        {
            *(Half*)q5p = (Half)(rng.NextSingle() * 0.1f);
            *(uint*)(q5p + 2) = (uint)rng.Next() ^ ((uint)rng.Next() << 16);
            for (int i = 0; i < 16; i++)
                (q5p + 6)[i] = (byte)rng.Next(0, 256);
            q5p += Q5_0BlockBytes;
        }

        // F32 source: ElementCount × 4 bytes
        _f32Src = (nint)NativeMemory.AlignedAlloc((nuint)(ElementCount * sizeof(float)), 64);
        var f32Span = new Span<float>((void*)_f32Src, ElementCount);
        for (int i = 0; i < ElementCount; i++)
            f32Span[i] = rng.NextSingle();

        _dest = new float[ElementCount];
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.AlignedFree((void*)_fp16Src);
        NativeMemory.AlignedFree((void*)_q8Src);
        NativeMemory.AlignedFree((void*)_q5_0Src);
        NativeMemory.AlignedFree((void*)_f32Src);
    }

    [Benchmark]
    public void FP16_ToFloat32()
        => Dequantize.ToFloat32(_fp16Src, ElementCount, QuantizationType.F16, _dest);

    [Benchmark]
    public void Q8_0_ToFloat32()
        => Dequantize.ToFloat32(_q8Src, ElementCount, QuantizationType.Q8_0, _dest);

    [Benchmark]
    public void Q8_0_Scalar()
        => Dequantize.DequantizeQ8_0Scalar(_q8Src, ElementCount, _dest);

    [Benchmark]
    public void Q5_0_ToFloat32()
        => Dequantize.ToFloat32(_q5_0Src, ElementCount, QuantizationType.Q5_0, _dest);

    [Benchmark]
    public void Q5_0_Scalar()
        => Dequantize.DequantizeQ5_0Scalar(_q5_0Src, ElementCount, _dest);

    [Benchmark]
    public void Q5_0_Avx2()
        => Dequantize.DequantizeQ5_0Avx2(_q5_0Src, ElementCount, _dest);

    [Benchmark]
    public void F32_Passthrough()
        => Dequantize.ToFloat32(_f32Src, ElementCount, QuantizationType.F32, _dest);
}
