using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using DotLLM.Core.Lora;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Benchmarks.Lora;

/// <summary>
/// Phase 4d.3 — Measures LoRA delta overhead vs the bare base projection.
/// Baseline = a single F32 GEMM at TinyLlama-1.1B q_proj shapes
/// (hidden=2048, q_out=2048, seq=128 typical prefill chunk).
/// LoRA path = baseline + <c>scale × (x · B) · A</c> at r=16.
/// Target: &lt;5% overhead on the bare projection.
/// </summary>
/// <remarks>
/// We benchmark at the kernel level (no model load) because the spec target
/// is the additional cost of the delta itself, and a kernel bench is fully
/// reproducible without checkpoint download. The macro-bench against a real
/// TinyLlama checkpoint is tracked as a follow-up — once a public checkpoint
/// path is wired into the bench harness, replace this file with a
/// model-level forward-pass bench.
/// </remarks>
[MemoryDiagnoser]
[ShortRunJob]
public unsafe class LoraDeltaOverheadBenchmark
{
    /// <summary>Sequence length (prefill chunk size).</summary>
    [Params(1, 128)]
    public int SeqLen { get; set; }

    /// <summary>LoRA rank.</summary>
    [Params(16)]
    public int Rank { get; set; }

    // TinyLlama q_proj shape.
    private const int HiddenSize = 2048;
    private const int OutputDim = 2048;

    private nint _xPtr;
    private nint _yBasePtr;
    private nint _yLoraPtr;
    private nint _wPtr;     // base weight [OutputDim, HiddenSize]
    private nint _bPtr;     // LoRA B [Rank, HiddenSize]
    private nint _aPtr;     // LoRA A [OutputDim, Rank]

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(123);

        _xPtr = AllocAligned(SeqLen * HiddenSize);
        _yBasePtr = AllocAligned(SeqLen * OutputDim);
        _yLoraPtr = AllocAligned(SeqLen * OutputDim);
        _wPtr = AllocAligned(OutputDim * HiddenSize);
        _bPtr = AllocAligned(Rank * HiddenSize);
        _aPtr = AllocAligned(OutputDim * Rank);

        FillRandom((float*)_xPtr, SeqLen * HiddenSize, rng, 0.05f);
        FillRandom((float*)_wPtr, OutputDim * HiddenSize, rng, 0.05f);
        FillRandom((float*)_bPtr, Rank * HiddenSize, rng, 0.05f);
        FillRandom((float*)_aPtr, OutputDim * Rank, rng, 0.05f);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        FreeAligned(_xPtr);
        FreeAligned(_yBasePtr);
        FreeAligned(_yLoraPtr);
        FreeAligned(_wPtr);
        FreeAligned(_bPtr);
        FreeAligned(_aPtr);
    }

    /// <summary>Baseline: only the base GEMM projection.</summary>
    [Benchmark(Baseline = true)]
    public void BaseProjectionOnly()
    {
        // C[N, M] = B[N, K] × A[M, K]^T, so y = x · w^T.
        MatMul.GemmF32((float*)_wPtr, (float*)_xPtr, (float*)_yBasePtr,
            OutputDim, HiddenSize, SeqLen);
    }

    /// <summary>Base GEMM + F32 LoRA delta (Phase 4a path).</summary>
    [Benchmark]
    public void BasePlusLoraF32()
    {
        MatMul.GemmF32((float*)_wPtr, (float*)_xPtr, (float*)_yLoraPtr,
            OutputDim, HiddenSize, SeqLen);
        LoraDelta.Apply(
            (float*)_xPtr, (float*)_bPtr, (float*)_aPtr, (float*)_yLoraPtr,
            SeqLen, HiddenSize, OutputDim, Rank, scale: 0.5f);
    }

    /// <summary>Base GEMM + F16 LoRA delta (Phase 4d.1 path).</summary>
    [Benchmark]
    public void BasePlusLoraF16()
    {
        MatMul.GemmF32((float*)_wPtr, (float*)_xPtr, (float*)_yLoraPtr,
            OutputDim, HiddenSize, SeqLen);
        // Reinterpret existing F32 buffers as F16 for the dispatch test —
        // we measure dispatch + dequant overhead, not the math (the test
        // suite already verifies numerical parity).
        LoraDelta.Apply(
            (float*)_xPtr, (void*)_bPtr, (void*)_aPtr, (float*)_yLoraPtr,
            SeqLen, HiddenSize, OutputDim, Rank, scale: 0.5f,
            LoraWeightDType.F16, LoraWeightDType.F16);
    }

    private static nint AllocAligned(long elementCount)
        => (nint)NativeMemory.AlignedAlloc((nuint)(elementCount * sizeof(float)), 64);

    private static void FreeAligned(nint p)
    {
        if (p != 0) NativeMemory.AlignedFree((void*)p);
    }

    private static void FillRandom(float* p, long n, Random rng, float scale)
    {
        for (long i = 0; i < n; i++)
            p[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
    }
}
