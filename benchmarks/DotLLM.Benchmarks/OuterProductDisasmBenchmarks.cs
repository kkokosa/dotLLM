using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Benchmarks;

/// <summary>
/// Disassembly comparison: existing 4-row VecDot (called per token) vs outer-product 4×3 microkernel.
/// Run with: dotnet run -c Release -- --filter '*OuterProductDisasm*' --disasm
/// </summary>
[DisassemblyDiagnoser(maxDepth: 2, printSource: false)]
[SimpleJob(warmupCount: 2, iterationCount: 5)]
public unsafe class OuterProductDisasmBenchmarks : IDisposable
{
    // SmolLM-like dimensions: K=576 → 18 blocks per row
    private const int K = 576;
    private const int Q8_0GroupSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int BlockCount = K / Q8_0GroupSize; // 18
    private const int M = 16; // 4 R4 groups — enough to exercise the kernel

    // R4 repacked weights
    private nint _repackedWeights;
    // Original row-major weights (for baseline comparison)
    private nint _originalWeights;
    // 3 pre-quantized Q8_0 input rows (tokens)
    private nint _inputQ8_0;
    private nint _inputQ8_1;
    private nint _inputQ8_2;
    // Output buffer
    private float* _output;

    private int _fullGroups;
    private int _tailRows;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int rowBytes = BlockCount * Q8_0BlockBytes;

        // Allocate original row-major weights
        _originalWeights = (nint)NativeMemory.AlignedAlloc((nuint)((long)M * rowBytes), 64);
        byte* p = (byte*)_originalWeights;
        for (int row = 0; row < M; row++)
        {
            for (int b = 0; b < BlockCount; b++)
            {
                *(Half*)p = (Half)(rng.NextSingle() * 0.1f);
                for (int i = 0; i < Q8_0GroupSize; i++)
                    ((sbyte*)(p + 2))[i] = (sbyte)rng.Next(-127, 128);
                p += Q8_0BlockBytes;
            }
        }

        // Repack to R4
        var rw = WeightRepacking.RepackR4(_originalWeights, QuantizationType.Q8_0, M, K);
        _repackedWeights = rw.Ptr;
        _fullGroups = rw.FullGroupCount;
        _tailRows = rw.TailRows;

        // Allocate 3 Q8_0 input rows
        _inputQ8_0 = (nint)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
        _inputQ8_1 = (nint)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
        _inputQ8_2 = (nint)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
        FillRandomQ8((byte*)_inputQ8_0, rng);
        FillRandomQ8((byte*)_inputQ8_1, rng);
        FillRandomQ8((byte*)_inputQ8_2, rng);

        // Output buffer: 3 tokens × M rows
        _output = (float*)NativeMemory.AlignedAlloc((nuint)(3 * M * sizeof(float)), 64);
    }

    private void FillRandomQ8(byte* ptr, Random rng)
    {
        for (int b = 0; b < BlockCount; b++)
        {
            *(Half*)ptr = (Half)(rng.NextSingle() * 0.1f);
            for (int i = 0; i < Q8_0GroupSize; i++)
                ((sbyte*)(ptr + 2))[i] = (sbyte)rng.Next(-127, 128);
            ptr += Q8_0BlockBytes;
        }
    }

    /// <summary>
    /// Baseline: existing 4-row VecDot called 3 times (once per token) on first R4 group.
    /// This is what the inner-product GEMM does per group.
    /// </summary>
    [Benchmark(Baseline = true)]
    public void VecDot4Rows_x3Tokens()
    {
        byte* groupBase = (byte*)_repackedWeights;
        MatMul.VecDotQ8_0Avx2_4RowsR4(groupBase, (byte*)_inputQ8_0, BlockCount, _output);
        MatMul.VecDotQ8_0Avx2_4RowsR4(groupBase, (byte*)_inputQ8_1, BlockCount, _output + M);
        MatMul.VecDotQ8_0Avx2_4RowsR4(groupBase, (byte*)_inputQ8_2, BlockCount, _output + 2 * M);
    }

    /// <summary>
    /// Outer-product 4×3: processes 4 rows × 3 tokens in one microkernel call.
    /// Expected to be faster due to weight block reuse and more ILP.
    /// </summary>
    [Benchmark]
    public void OuterProduct4x3()
    {
        byte* groupBase = (byte*)_repackedWeights;
        MatMul.OuterProductQ8_0Avx2_4x3(
            groupBase,
            (byte*)_inputQ8_0,
            (byte*)_inputQ8_1,
            (byte*)_inputQ8_2,
            _output, BlockCount, M);
    }

    /// <summary>
    /// Full outer-product GEMM: all M rows × 3 tokens.
    /// </summary>
    [Benchmark]
    public void OuterProductGemm_3Tokens()
    {
        // Pack 3 token inputs contiguously as OuterProductGemmQ8_0 expects
        int rowBytes = BlockCount * Q8_0BlockBytes;
        byte* packed = stackalloc byte[3 * rowBytes];
        Buffer.MemoryCopy((void*)_inputQ8_0, packed, rowBytes, rowBytes);
        Buffer.MemoryCopy((void*)_inputQ8_1, packed + rowBytes, rowBytes, rowBytes);
        Buffer.MemoryCopy((void*)_inputQ8_2, packed + 2 * rowBytes, rowBytes, rowBytes);

        MatMul.OuterProductGemmQ8_0((byte*)_repackedWeights, packed, _output,
            _fullGroups, _tailRows, BlockCount, M, 3);
    }

    /// <summary>
    /// Baseline full GEMM: existing tiled path for 3 tokens.
    /// </summary>
    [Benchmark]
    public void TiledGemm_3Tokens()
    {
        int rowBytes = BlockCount * Q8_0BlockBytes;
        byte* packed = stackalloc byte[3 * rowBytes];
        Buffer.MemoryCopy((void*)_inputQ8_0, packed, rowBytes, rowBytes);
        Buffer.MemoryCopy((void*)_inputQ8_1, packed + rowBytes, rowBytes, rowBytes);
        Buffer.MemoryCopy((void*)_inputQ8_2, packed + 2 * rowBytes, rowBytes, rowBytes);

        MatMul.GemmQ8_0((byte*)_originalWeights, (float*)null, _output, M, K, 3, packed);
    }

    public void Dispose()
    {
        NativeMemory.AlignedFree((void*)_repackedWeights);
        NativeMemory.AlignedFree((void*)_originalWeights);
        NativeMemory.AlignedFree((void*)_inputQ8_0);
        NativeMemory.AlignedFree((void*)_inputQ8_1);
        NativeMemory.AlignedFree((void*)_inputQ8_2);
        NativeMemory.AlignedFree(_output);
    }
}
