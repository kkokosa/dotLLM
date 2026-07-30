using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Benchmarks;

/// <summary>
/// AVX-VNNI vs AVX2 for the R4-interleaved Q8_0 dot product that <c>GemmR4TiledQ8_0</c> drives
/// during prefill (#414). The AVX2 kernel is the baseline, so the VNNI row reads directly as
/// "N times the speed of what shipped".
/// </summary>
/// <remarks>
/// <para>
/// Run with: <c>dotnet run -c Release -- --filter '*R4Vnni*'</c>
/// </para>
/// <para>
/// Single-threaded and deliberately so: this isolates the microkernel from the L2 tiling and
/// thread partitioning in <c>GemmR4TiledQ8_0</c>. K=4096 matches the shape used in the #322
/// Zen5 measurements so the numbers are directly comparable to that table.
/// </para>
/// </remarks>
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class R4VnniMicroBenchmarks : IDisposable
{
    private const int Q8_0GroupSize = 32;
    private const int Q8_0BlockBytes = 34;

    private nint _originalWeights;
    private WeightRepacking.RepackedWeight _repacked;
    private nint _inputQ8;
    private float* _output;
    private int _blockCount;

    /// <summary>Reduction length. 4096 matches #322; 512 is a typical attention projection.</summary>
    [Params(512, 4096)]
    public int K { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        if (!Avx512BW.IsSupported || !AvxVnni.IsSupported)
        {
            throw new PlatformNotSupportedException(
                "R4VnniMicroBenchmarks requires AVX-512BW + AVX-VNNI; without both, the VNNI row " +
                "would not be measuring the kernel under test.");
        }

        var rng = new Random(42);
        _blockCount = K / Q8_0GroupSize;
        int rowBytes = _blockCount * Q8_0BlockBytes;

        // One R4 group is 4 rows — the microkernel's unit of work.
        _originalWeights = (nint)NativeMemory.AlignedAlloc((nuint)(4L * rowBytes), 64);
        FillRandomQ8((byte*)_originalWeights, rng, 4, _blockCount);

        _repacked = WeightRepacking.RepackR4(_originalWeights, QuantizationType.Q8_0, 4, K);

        _inputQ8 = (nint)NativeMemory.AlignedAlloc((nuint)rowBytes, 64);
        FillRandomQ8((byte*)_inputQ8, rng, 1, _blockCount);

        _output = (float*)NativeMemory.AlignedAlloc(4 * sizeof(float), 64);
    }

    [Benchmark(Baseline = true, Description = "AVX2 (vpmaddubsw + vpmaddwd)")]
    public void Avx2_4RowsR4() =>
        MatMul.VecDotQ8_0Avx2_4RowsR4((byte*)_repacked.Ptr, (byte*)_inputQ8, _blockCount, _output);

    [Benchmark(Description = "AVX-VNNI (vpdpbusd)")]
    public void Vnni_4RowsR4() =>
        MatMul.VecDotQ8_0Vnni_4RowsR4((byte*)_repacked.Ptr, (byte*)_inputQ8, _blockCount, _output);

    internal static void FillRandomQ8(byte* ptr, Random rng, int rows, int blockCount)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int b = 0; b < blockCount; b++)
            {
                *(Half*)ptr = (Half)(rng.NextSingle() * 0.1f);
                // [-127, 127]: the range Q8_0 quantization actually produces. -128 would break the
                // abs/sign idiom the vectorized reductions rely on.
                for (int i = 0; i < Q8_0GroupSize; i++)
                    ((sbyte*)(ptr + 2))[i] = (sbyte)rng.Next(-127, 128);
                ptr += Q8_0BlockBytes;
            }
        }
    }

    public void Dispose()
    {
        _repacked.Dispose();
        NativeMemory.AlignedFree((void*)_originalWeights);
        NativeMemory.AlignedFree((void*)_inputQ8);
        NativeMemory.AlignedFree(_output);
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// The three row-major 4-row tiers against each other. This is not the R4 prefill path — it is the
/// <c>ComputeRows</c> dispatch that #399 changed — and it is here to check that change's premise:
/// that <c>vpdpbusd</c> beats the integer-reduction pair in the kernel it actually shipped into.
/// </summary>
/// <remarks>
/// The AVX-512 maddubs tier is the baseline because it is what the VNNI tier displaced on hosts
/// that have both. Same shapes as <see cref="R4VnniMicroBenchmarks"/> so the two tables line up.
/// </remarks>
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class RowMajorVnniBenchmarks : IDisposable
{
    private const int Q8_0GroupSize = 32;
    private const int Q8_0BlockBytes = 34;

    private nint _weights;
    private nint _inputQ8;
    private float* _output;
    private int _blockCount;
    private int _rowBytes;

    [Params(512, 4096)]
    public int K { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        if (!Avx512BW.IsSupported || !AvxVnni.IsSupported)
            throw new PlatformNotSupportedException("Requires AVX-512BW + AVX-VNNI.");

        var rng = new Random(42);
        _blockCount = K / Q8_0GroupSize;
        _rowBytes = _blockCount * Q8_0BlockBytes;

        _weights = (nint)NativeMemory.AlignedAlloc((nuint)(4L * _rowBytes), 64);
        R4VnniMicroBenchmarks.FillRandomQ8((byte*)_weights, rng, 4, _blockCount);

        _inputQ8 = (nint)NativeMemory.AlignedAlloc((nuint)_rowBytes, 64);
        R4VnniMicroBenchmarks.FillRandomQ8((byte*)_inputQ8, rng, 1, _blockCount);

        _output = (float*)NativeMemory.AlignedAlloc(4 * sizeof(float), 64);
    }

    [Benchmark(Baseline = true, Description = "AVX-512 maddubs (pre-#399)")]
    public void Avx512_4Rows() => MatMul.VecDotQ8_0Avx512_4Rows(
        W(0), W(1), W(2), W(3), (byte*)_inputQ8, _blockCount, _output);

    [Benchmark(Description = "AVX-VNNI vpdpbusd (#399, shipped)")]
    public void Vnni_4Rows() => MatMul.VecDotQ8_0Vnni_4Rows(
        W(0), W(1), W(2), W(3), (byte*)_inputQ8, _blockCount, _output);

    [Benchmark(Description = "AVX2 maddubs")]
    public void Avx2_4Rows() => MatMul.VecDotQ8_0Avx2_4Rows(
        W(0), W(1), W(2), W(3), (byte*)_inputQ8, _blockCount, _output);

    private byte* W(int row) => (byte*)_weights + (long)row * _rowBytes;

    public void Dispose()
    {
        NativeMemory.AlignedFree((void*)_weights);
        NativeMemory.AlignedFree((void*)_inputQ8);
        NativeMemory.AlignedFree(_output);
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// The same A/B at prefill scale — a full weight matrix against a batch of tokens, which is what
/// <c>GemmR4TiledQ8_0</c> actually issues. Catches effects the single-group microbenchmark cannot:
/// weight streaming from L2/L3 and any change in how well the tile fits.
/// </summary>
/// <remarks>
/// Shapes are Llama-3.2-1B's: M=2048 x K=2048 is an attention projection, and N is the prompt
/// length being prefilled. Single-threaded, so this is per-core kernel throughput rather than a
/// wall-clock prefill number.
/// </remarks>
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class R4VnniPrefillBenchmarks : IDisposable
{
    private const int Q8_0GroupSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int M = 2048;
    private const int K = 2048;

    private nint _originalWeights;
    private WeightRepacking.RepackedWeight _repacked;
    private nint _inputQ8;
    private float* _output;
    private int _blockCount;
    private int _groupBytes;

    /// <summary>Token count. 1 is decode-shaped; 32 and 256 are prefill-shaped.</summary>
    [Params(1, 32, 256)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        if (!Avx512BW.IsSupported || !AvxVnni.IsSupported)
            throw new PlatformNotSupportedException("Requires AVX-512BW + AVX-VNNI.");

        var rng = new Random(42);
        _blockCount = K / Q8_0GroupSize;
        int rowBytes = _blockCount * Q8_0BlockBytes;
        _groupBytes = 4 * rowBytes;

        _originalWeights = (nint)NativeMemory.AlignedAlloc((nuint)((long)M * rowBytes), 64);
        R4VnniMicroBenchmarks.FillRandomQ8((byte*)_originalWeights, rng, M, _blockCount);

        _repacked = WeightRepacking.RepackR4(_originalWeights, QuantizationType.Q8_0, M, K);

        _inputQ8 = (nint)NativeMemory.AlignedAlloc((nuint)((long)N * rowBytes), 64);
        R4VnniMicroBenchmarks.FillRandomQ8((byte*)_inputQ8, rng, N, _blockCount);

        _output = (float*)NativeMemory.AlignedAlloc((nuint)((long)N * M * sizeof(float)), 64);
    }

    [Benchmark(Baseline = true, Description = "AVX2 (vpmaddubsw + vpmaddwd)")]
    public void Avx2_Prefill() => RunAllGroups(vnni: false);

    [Benchmark(Description = "AVX-VNNI (vpdpbusd)")]
    public void Vnni_Prefill() => RunAllGroups(vnni: true);

    /// <summary>
    /// Mirrors the loop nest inside <see cref="MatMul.GemmR4TiledQ8_0"/> — groups outermost,
    /// tokens inside — but calls one kernel explicitly instead of going through the ISA dispatch,
    /// so both variants can be measured in the same process.
    /// </summary>
    private void RunAllGroups(bool vnni)
    {
        int fullGroups = _repacked.FullGroupCount;
        int rowBytes = _blockCount * Q8_0BlockBytes;
        byte* weights = (byte*)_repacked.Ptr;

        for (int g = 0; g < fullGroups; g++)
        {
            byte* groupBase = weights + (long)g * _groupBytes;
            for (int t = 0; t < N; t++)
            {
                byte* x = (byte*)_inputQ8 + (long)t * rowBytes;
                float* c = _output + (long)t * M + g * 4;
                if (vnni)
                    MatMul.VecDotQ8_0Vnni_4RowsR4(groupBase, x, _blockCount, c);
                else
                    MatMul.VecDotQ8_0Avx2_4RowsR4(groupBase, x, _blockCount, c);
            }
        }
    }

    public void Dispose()
    {
        _repacked.Dispose();
        NativeMemory.AlignedFree((void*)_originalWeights);
        NativeMemory.AlignedFree((void*)_inputQ8);
        NativeMemory.AlignedFree(_output);
        GC.SuppressFinalize(this);
    }
}
