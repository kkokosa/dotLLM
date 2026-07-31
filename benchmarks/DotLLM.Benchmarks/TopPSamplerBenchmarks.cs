using System.Buffers;
using System.Numerics.Tensors;
using BenchmarkDotNet.Attributes;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;

namespace DotLLM.Benchmarks;

/// <summary>
/// Benchmarks for <see cref="TopPSampler.Apply"/>. The sampler softmaxes the full vocab,
/// then keeps the smallest descending-probability prefix whose cumulative mass meets
/// <see cref="TopP"/>, masking the rest to -infinity.
/// <para>
/// The pre-cutoff optimisation (Karpathy llama2.c) drops every token below
/// <c>(1 - topP) / (n - 1)</c> before sorting — typically reducing the sorted set
/// from full vocab (32K–128K) to a few hundred entries. This benchmark covers vocab
/// sizes and topP values typical of production decoding (Llama 128K, SmolLM 49K,
/// TinyLlama 32K).
/// </para>
/// <para>
/// Each invocation copies the source logits into a scratch buffer before running the
/// sampler (Apply is mutating). The copy cost is identical across configurations, so
/// relative comparisons remain valid. <see cref="MemoryDiagnoserAttribute"/> tracks
/// per-call allocations — should be 0 B in steady state (pooled scratch buffers).
/// </para>
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class TopPSamplerBenchmarks
{
    private float[] _srcLogits = null!;
    private float[] _scratch = null!;
    private TopPSampler _sampler = null!;
    private SamplerContext _context;

    /// <summary>Vocabulary size — matches typical Llama (128K) / SmolLM (49K) / TinyLlama (32K).</summary>
    [Params(32_000, 128_000)]
    public int VocabSize { get; set; }

    /// <summary>Top-P cutoff — 0.9 / 0.95 are the common production values.</summary>
    [Params(0.9f, 0.95f)]
    public float TopP { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _srcLogits = new float[VocabSize];
        _scratch = new float[VocabSize];
        for (int i = 0; i < VocabSize; i++)
            _srcLogits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);

        _sampler = new TopPSampler();
        _context = new SamplerContext(
            Temperature: 1.0f,
            TopK: 0,
            TopP: TopP,
            MinP: 0f,
            Seed: null);
    }

    /// <summary>
    /// Baseline: full-vocab sort (the pre-optimization behaviour of <see cref="TopPSampler"/>).
    /// Inlined here so the benchmark can compare against the optimized implementation in a
    /// single run. Mirrors the algorithm shape exactly; only the candidate-set pruning is
    /// absent.
    /// </summary>
    [Benchmark(Baseline = true)]
    public void Baseline_FullSort()
    {
        _srcLogits.AsSpan().CopyTo(_scratch);
        ApplyTopPFullSort(_scratch, TopP);
    }

    /// <summary>
    /// Optimized: pre-filter via Karpathy cutoff `(1 - topP) / (n - 1)` before sorting.
    /// Calls into the production sampler so this measures the actual shipped code.
    /// </summary>
    [Benchmark]
    public void Optimized_PreFilter()
    {
        _srcLogits.AsSpan().CopyTo(_scratch);
        _sampler.Apply(_scratch, _context);
    }

    /// <summary>Inlined replica of the pre-optimization TopPSampler.Apply algorithm.</summary>
    private static void ApplyTopPFullSort(Span<float> logits, float topP)
    {
        if (topP >= 1.0f) return;

        int vocabSize = logits.Length;
        float[] rentedProbs = ArrayPool<float>.Shared.Rent(vocabSize);
        int[] rentedIndices = ArrayPool<int>.Shared.Rent(vocabSize);
        bool[] rentedKeep = ArrayPool<bool>.Shared.Rent(vocabSize);
        try
        {
            var probs = rentedProbs.AsSpan(0, vocabSize);
            TensorPrimitives.SoftMax(logits, probs);

            for (int i = 0; i < vocabSize; i++)
                rentedIndices[i] = i;

            Array.Sort(rentedProbs, rentedIndices, 0, vocabSize);

            float cumulative = 0f;
            int cutoffCount = vocabSize;
            for (int i = vocabSize - 1; i >= 0; i--)
            {
                cumulative += rentedProbs[i];
                if (cumulative >= topP)
                {
                    cutoffCount = vocabSize - i;
                    break;
                }
            }

            var keep = rentedKeep.AsSpan(0, vocabSize);
            keep.Clear();
            int keepStart = vocabSize - cutoffCount;
            for (int i = keepStart; i < vocabSize; i++)
                keep[rentedIndices[i]] = true;

            for (int i = 0; i < vocabSize; i++)
                if (!keep[i])
                    logits[i] = float.NegativeInfinity;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rentedProbs);
            ArrayPool<int>.Shared.Return(rentedIndices);
            ArrayPool<bool>.Shared.Return(rentedKeep);
        }
    }
}
