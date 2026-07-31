using BenchmarkDotNet.Attributes;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Benchmarks;

/// <summary>
/// Microbenchmarks for the residual add at the end of each transformer sub-block.
/// Compares two equivalent code paths:
/// <list type="bullet">
///   <item><b>Baseline</b>: saved-residual copy + out-of-place add
///         (<c>hidden = residualCopy + normOut</c>) — the previous behaviour.</item>
///   <item><b>InPlace</b>: in-place add (<c>hidden += normOut</c>) — the optimized
///         behaviour, replacing the [seqLen × hiddenSize] copy with a single
///         <c>TensorPrimitives.Add</c> with aliased output.</item>
/// </list>
/// Each transformer layer performs this add twice (post-attention, post-FFN), so the
/// per-forward saving is 2 × NumLayers × delta. At hiddenSize=4096, NumLayers=32,
/// SeqLen=512 the issue cites ~512 MB of copy traffic eliminated per prefill.
/// <para>
/// Allocated bytes should read 0 — buffers are pre-allocated under <see cref="GlobalSetup"/>.
/// </para>
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class ResidualInPlaceAddBenchmarks
{
    private float[] _hidden = null!;
    private float[] _normOut = null!;
    private float[] _residual = null!;
    private float[] _hiddenInit = null!;

    /// <summary>Hidden size — 4096 is typical for Llama-7B-class models.</summary>
    [Params(2048, 4096)]
    public int HiddenSize { get; set; }

    /// <summary>Sequence length — 1 = decode, 512 = prefill regime where the copy dominates.</summary>
    [Params(1, 128, 512)]
    public int SeqLen { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        int n = HiddenSize * SeqLen;
        var rng = new Random(42);
        _hiddenInit = new float[n];
        _normOut = new float[n];
        _hidden = new float[n];
        _residual = new float[n];
        for (int i = 0; i < n; i++)
        {
            _hiddenInit[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            _normOut[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }
    }

    /// <summary>
    /// Baseline: copy `hidden` into a saved residual buffer, then add `residual + normOut`
    /// into `hidden`. Mirrors the previous TransformerModel.Forward step-a + step-g.
    /// </summary>
    [Benchmark(Baseline = true)]
    public void Baseline_CopyThenOutOfPlaceAdd()
    {
        // Reset hidden (cost included in both variants for fairness).
        _hiddenInit.AsSpan().CopyTo(_hidden);

        // Saved-residual copy (the work this opt eliminates).
        _hidden.AsSpan().CopyTo(_residual);

        // Out-of-place add: hidden = residual + normOut.
        for (int t = 0; t < SeqLen; t++)
        {
            Add.Execute(
                new ReadOnlySpan<float>(_residual, t * HiddenSize, HiddenSize),
                new ReadOnlySpan<float>(_normOut, t * HiddenSize, HiddenSize),
                new Span<float>(_hidden, t * HiddenSize, HiddenSize));
        }
    }

    /// <summary>
    /// Optimization, step 1: skip the saved-residual copy, accumulate `normOut` into
    /// `hidden` in place, still dispatched once per token. Isolates the saving that
    /// comes purely from dropping the copy.
    /// </summary>
    [Benchmark]
    public void InPlace_AccumulateAdd_PerToken()
    {
        _hiddenInit.AsSpan().CopyTo(_hidden);

        for (int t = 0; t < SeqLen; t++)
        {
            var row = new Span<float>(_hidden, t * HiddenSize, HiddenSize);
            Add.Execute(row, new ReadOnlySpan<float>(_normOut, t * HiddenSize, HiddenSize), row);
        }
    }

    /// <summary>
    /// Optimization, step 2 (the shipped path): both buffers are contiguous
    /// [SeqLen × HiddenSize], so the per-token loop collapses into a single
    /// <c>Add.Execute</c> over the whole range. Mirrors TransformerModel.Forward
    /// steps g and k, and isolates the additional saving from removing the
    /// per-token dispatch.
    /// </summary>
    [Benchmark]
    public void InPlace_AccumulateAdd_SingleCall()
    {
        _hiddenInit.AsSpan().CopyTo(_hidden);

        var all = _hidden.AsSpan();
        Add.Execute(all, _normOut, all);
    }
}
