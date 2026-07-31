using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Benchmarks;

/// <summary>
/// Compares the new <see cref="OuterProductGemm.OuterProductGemmF32"/> against the
/// production <see cref="MatMul.GemmF32(float*, float*, float*, int, int, int)"/>
/// at prefill-shaped workloads (multi-token, contraction along K).
///
/// Convention: <c>C[N,M] = B[N,K] × A[M,K]^T</c> — N is the batch (token count),
/// M is the output dim (e.g. hidden size), K is the contraction dim (e.g. K-proj
/// from the attention block input).
///
/// Run with:
///   dotnet run -c Release -- --filter '*OuterProductGemmF32Benchmark*'
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 5, iterationCount: 15)]
public unsafe class OuterProductGemmF32Benchmark : IDisposable
{
    // Three prefill profiles spanning typical attention-projection shapes.
    // K=4096 mirrors Llama-3-8B's hidden_size = 4096 and 32-head q_proj output.
    [Params(128, 512, 2048)]
    public int M { get; set; }

    public int K { get; set; } = 4096;

    public int N { get; set; } = 32;

    private float* _a;
    private float* _b;
    private float* _c;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        long aLen = (long)M * K;
        long bLen = (long)N * K;
        long cLen = (long)N * M;

        _a = (float*)NativeMemory.AlignedAlloc((nuint)(aLen * sizeof(float)), 64);
        _b = (float*)NativeMemory.AlignedAlloc((nuint)(bLen * sizeof(float)), 64);
        _c = (float*)NativeMemory.AlignedAlloc((nuint)(cLen * sizeof(float)), 64);

        for (long i = 0; i < aLen; i++) _a[i] = rng.NextSingle() * 2f - 1f;
        for (long i = 0; i < bLen; i++) _b[i] = rng.NextSingle() * 2f - 1f;
    }

    public void Dispose()
    {
        if (_a != null) { NativeMemory.AlignedFree(_a); _a = null; }
        if (_b != null) { NativeMemory.AlignedFree(_b); _b = null; }
        if (_c != null) { NativeMemory.AlignedFree(_c); _c = null; }
        GC.SuppressFinalize(this);
    }

    /// <summary>Baseline: production tiled GEMM path used by the engine today.</summary>
    [Benchmark(Baseline = true)]
    public void GemmF32_Baseline()
    {
        MatMul.GemmF32(_a, _b, _c, M, K, N);
    }

    /// <summary>Candidate: new outer-product 4×3 AVX2 microkernel.</summary>
    [Benchmark]
    public void OuterProductGemmF32_Avx2()
    {
        OuterProductGemm.OuterProductGemmF32(_a, _b, _c, M, K, N);
    }
}
