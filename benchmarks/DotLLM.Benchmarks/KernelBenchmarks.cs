using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Benchmarks;

/// <summary>
/// Benchmarks for CPU compute kernels (Add, Multiply, SiLU, RmsNorm, Softmax, MatMul).
/// <para>
/// The <see cref="MemoryDiagnoserAttribute"/> column <c>Allocated</c> must read <c>0 B</c>
/// for all benchmarks — kernels have zero managed allocations on the hot path.
/// </para>
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class KernelBenchmarks
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    private float[] _a = null!;
    private float[] _b = null!;
    private float[] _result = null!;
    private float[] _weight = null!;

    // MatMul data
    private nint _weightsQ8;
    private nint _weightsF32;
    private float[] _x = null!;
    private float[] _gemvResult = null!;

    // Pre-built Q8_0 activation buffer for VecDot-level benchmarks
    private nint _xQ8;

    private const int GemvM = 64;

    /// <summary>Inner dimension for all kernels.</summary>
    [Params(4096, 11008)]
    public int K { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);

        _a = new float[K];
        _b = new float[K];
        _result = new float[K];
        _weight = new float[K];
        _x = new float[K];
        _gemvResult = new float[GemvM];

        for (int i = 0; i < K; i++)
        {
            _a[i] = rng.NextSingle() * 2f - 1f;
            _b[i] = rng.NextSingle() * 2f - 1f;
            _weight[i] = rng.NextSingle() * 2f;
            _x[i] = rng.NextSingle() * 2f - 1f;
        }

        // Q8_0 weights: GemvM rows × K/32 blocks × 34 bytes
        int blocksPerRow = K / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        _weightsQ8 = (nint)NativeMemory.AlignedAlloc((nuint)(GemvM * rowBytes), 64);
        byte* p = (byte*)_weightsQ8;
        for (int row = 0; row < GemvM; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                *(Half*)p = (Half)(rng.NextSingle() * 0.1f);
                for (int i = 0; i < Q8_0GroupSize; i++)
                    ((sbyte*)(p + 2))[i] = (sbyte)rng.Next(-127, 128);
                p += Q8_0BlockBytes;
            }
        }

        // F32 weights: GemvM × K
        _weightsF32 = (nint)NativeMemory.AlignedAlloc((nuint)(GemvM * K * sizeof(float)), 64);
        float* fp = (float*)_weightsF32;
        for (int i = 0; i < GemvM * K; i++)
            fp[i] = rng.NextSingle() * 2f - 1f;

        // Pre-quantize activation vector for VecDot-level benchmarks
        _xQ8 = (nint)NativeMemory.AlignedAlloc((nuint)(blocksPerRow * Q8_0BlockBytes), 64);
        fixed (float* xp = _x)
            MatMul.QuantizeF32ToQ8_0(xp, (byte*)_xQ8, K);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.AlignedFree((void*)_weightsQ8);
        NativeMemory.AlignedFree((void*)_weightsF32);
        NativeMemory.AlignedFree((void*)_xQ8);
    }

    // ──────────────────── Element-wise ops ────────────────────

    [Benchmark]
    public void Add()
        => Cpu.Kernels.Add.Execute(_a, _b, _result);

    [Benchmark]
    public void Multiply()
        => Cpu.Kernels.Multiply.Execute(_a, _b, _result);

    [Benchmark]
    public void SiLu()
        => Cpu.Kernels.SiLu.Execute(_a, _result);

    // ──────────────────── Reduction ops ────────────────────

    [Benchmark]
    public void RmsNorm()
        => Cpu.Kernels.RmsNorm.Execute(_a, _weight, 1e-5f, _result);

    [Benchmark]
    public void Softmax()
        => Cpu.Kernels.Softmax.Execute(_a, _result);

    // ──────────────────── MatMul ────────────────────

    [Benchmark]
    public void GemvF32()
    {
        fixed (float* xp = _x, rp = _gemvResult)
            MatMul.GemvF32((float*)_weightsF32, xp, rp, GemvM, K);
    }

    [Benchmark(Baseline = true)]
    public void GemvQ8_0()
    {
        fixed (float* xp = _x, rp = _gemvResult)
            MatMul.GemvQ8_0((byte*)_weightsQ8, xp, rp, GemvM, K);
    }

    // ──────────────────── VecDot micro-benchmarks ────────────────────

    [Benchmark]
    public float VecDotQ8_0_Scalar()
    {
        int blockCount = K / Q8_0GroupSize;
        return MatMul.VecDotQ8_0Scalar((byte*)_weightsQ8, (byte*)_xQ8, blockCount);
    }

    [Benchmark]
    public float VecDotQ8_0_Avx2()
    {
        if (!Avx2.IsSupported) return 0f;
        int blockCount = K / Q8_0GroupSize;
        return MatMul.VecDotQ8_0Avx2((byte*)_weightsQ8, (byte*)_xQ8, blockCount);
    }

    [Benchmark]
    public void QuantizeF32ToQ8_0()
    {
        fixed (float* xp = _x)
            MatMul.QuantizeF32ToQ8_0(xp, (byte*)_xQ8, K);
    }

    [Benchmark]
    public void QuantizeF32ToQ8_0_Scalar()
    {
        fixed (float* xp = _x)
            MatMul.QuantizeF32ToQ8_0Scalar(xp, (byte*)_xQ8, K);
    }
}

/// <summary>
/// Separate benchmark class for GEMV scaling by M (output rows).
/// Uses a fixed K=4096 to isolate the impact of multi-row optimization.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class GemvScalingBenchmarks
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int FixedK = 4096;

    private nint _weightsQ8;
    private float[] _x = null!;
    private float[] _result = null!;

    [Params(1, 64, 1024, 4096)]
    public int M { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int blocksPerRow = FixedK / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;

        _weightsQ8 = (nint)NativeMemory.AlignedAlloc((nuint)(M * rowBytes), 64);
        _x = new float[FixedK];
        _result = new float[M];

        for (int i = 0; i < FixedK; i++)
            _x[i] = rng.NextSingle() * 2f - 1f;

        byte* p = (byte*)_weightsQ8;
        for (int row = 0; row < M; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                *(Half*)p = (Half)(rng.NextSingle() * 0.1f);
                for (int i = 0; i < Q8_0GroupSize; i++)
                    ((sbyte*)(p + 2))[i] = (sbyte)rng.Next(-127, 128);
                p += Q8_0BlockBytes;
            }
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.AlignedFree((void*)_weightsQ8);
    }

    [Benchmark]
    public void GemvQ8_0_ByM()
    {
        fixed (float* xp = _x, rp = _result)
            MatMul.GemvQ8_0((byte*)_weightsQ8, xp, rp, M, FixedK);
    }
}
