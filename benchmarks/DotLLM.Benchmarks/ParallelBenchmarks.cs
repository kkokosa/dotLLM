using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;

namespace DotLLM.Benchmarks;

/// <summary>
/// Thread-scaling benchmarks for parallel GEMV, GEMM, and Attention kernels.
/// Measures throughput at 1, 2, 4, and 8 threads to verify near-linear scaling.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class ParallelGemvBenchmarks : IDisposable
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int M = 4096;
    private const int K = 4096;

    private nint _weightsQ8;
    private nint _weightsF32;
    private float[] _x = null!;
    private float[] _result = null!;
    private ComputeThreadPool? _pool;

    [Params(1, 2, 4, 8)]
    public int Threads { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int blocksPerRow = K / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;

        _weightsQ8 = (nint)NativeMemory.AlignedAlloc((nuint)((long)M * rowBytes), 64);
        _weightsF32 = (nint)NativeMemory.AlignedAlloc((nuint)((long)M * K * sizeof(float)), 64);
        _x = new float[K];
        _result = new float[M];

        for (int i = 0; i < K; i++)
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

        float* fp = (float*)_weightsF32;
        for (long i = 0; i < (long)M * K; i++)
            fp[i] = rng.NextSingle() * 2f - 1f;

        _pool = Threads >= 2 ? new ComputeThreadPool(Threads) : null;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.AlignedFree((void*)_weightsQ8);
        NativeMemory.AlignedFree((void*)_weightsF32);
        _pool?.Dispose();
    }

    public void Dispose() => Cleanup();

    [Benchmark(Baseline = true)]
    public void GemvQ8_0()
    {
        fixed (float* xp = _x, rp = _result)
            MatMul.GemvQ8_0((byte*)_weightsQ8, xp, rp, M, K, _pool);
    }

    [Benchmark]
    public void GemvF32()
    {
        fixed (float* xp = _x, rp = _result)
            MatMul.GemvF32((float*)_weightsF32, xp, rp, M, K, _pool);
    }
}

/// <summary>
/// Thread-scaling benchmarks for parallel GEMM (batched MatMul for prefill).
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class ParallelGemmBenchmarks : IDisposable
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int M = 4096;
    private const int K = 4096;
    private const int N = 64;

    private nint _weightsQ8;
    private nint _weightsF32;
    private float[] _input = null!;
    private float[] _result = null!;
    private ComputeThreadPool? _pool;

    [Params(1, 2, 4, 8)]
    public int Threads { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int blocksPerRow = K / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;

        _weightsQ8 = (nint)NativeMemory.AlignedAlloc((nuint)((long)M * rowBytes), 64);
        _weightsF32 = (nint)NativeMemory.AlignedAlloc((nuint)((long)M * K * sizeof(float)), 64);
        _input = new float[N * K];
        _result = new float[N * M];

        for (int i = 0; i < _input.Length; i++)
            _input[i] = rng.NextSingle() * 2f - 1f;

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

        float* fp = (float*)_weightsF32;
        for (long i = 0; i < (long)M * K; i++)
            fp[i] = rng.NextSingle() * 2f - 1f;

        _pool = Threads >= 2 ? new ComputeThreadPool(Threads) : null;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.AlignedFree((void*)_weightsQ8);
        NativeMemory.AlignedFree((void*)_weightsF32);
        _pool?.Dispose();
    }

    public void Dispose() => Cleanup();

    [Benchmark(Baseline = true)]
    public void GemmQ8_0()
    {
        fixed (float* inp = _input, res = _result)
            MatMul.GemmQ8_0((byte*)_weightsQ8, inp, res, M, K, N, _pool);
    }

    [Benchmark]
    public void GemmF32()
    {
        fixed (float* inp = _input, res = _result)
            MatMul.GemmF32((float*)_weightsF32, inp, res, M, K, N, _pool);
    }
}

/// <summary>
/// Thread-scaling benchmarks for parallel Attention.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class ParallelAttentionBenchmarks : IDisposable
{
    private const int NumHeads = 32;
    private const int NumKvHeads = 8;
    private const int HeadDim = 64;
    private const int SeqQ = 1;
    private const int SeqKv = 128;

    private nint _q, _k, _v, _output;
    private ComputeThreadPool? _pool;

    [Params(1, 2, 4, 8)]
    public int Threads { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int qSize = SeqQ * NumHeads * HeadDim;
        int kvSize = SeqKv * NumKvHeads * HeadDim;
        int outSize = SeqQ * NumHeads * HeadDim;

        _q = (nint)NativeMemory.AlignedAlloc((nuint)(qSize * sizeof(float)), 64);
        _k = (nint)NativeMemory.AlignedAlloc((nuint)(kvSize * sizeof(float)), 64);
        _v = (nint)NativeMemory.AlignedAlloc((nuint)(kvSize * sizeof(float)), 64);
        _output = (nint)NativeMemory.AlignedAlloc((nuint)(outSize * sizeof(float)), 64);

        FillRandom((float*)_q, qSize, rng);
        FillRandom((float*)_k, kvSize, rng);
        FillRandom((float*)_v, kvSize, rng);

        _pool = Threads >= 2 ? new ComputeThreadPool(Threads) : null;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.AlignedFree((void*)_q);
        NativeMemory.AlignedFree((void*)_k);
        NativeMemory.AlignedFree((void*)_v);
        NativeMemory.AlignedFree((void*)_output);
        _pool?.Dispose();
    }

    public void Dispose() => Cleanup();

    [Benchmark]
    public void Attention_HeadParallel()
    {
        Cpu.Kernels.Attention.Execute(
            (float*)_q, (float*)_k, (float*)_v, (float*)_output,
            SeqQ, SeqKv, NumHeads, NumKvHeads, HeadDim, SeqKv - SeqQ, _pool);
    }

    private static void FillRandom(float* ptr, int count, Random rng)
    {
        for (int i = 0; i < count; i++)
            ptr[i] = rng.NextSingle() * 2f - 1f;
    }
}
