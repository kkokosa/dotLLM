using BenchmarkDotNet.Attributes;
using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers;

namespace DotLLM.Benchmarks;

/// <summary>
/// Benchmarks for <see cref="TopKSampler.Apply"/>. The sampler finds the K-th largest
/// logit among <see cref="VocabSize"/> values and masks the rest to -infinity.
/// <para>
/// Each invocation copies the source logits into a scratch buffer before running the
/// sampler (Apply is mutating). The copy cost is identical across all configurations,
/// so relative comparisons remain valid.
/// </para>
/// <para>
/// The <see cref="MemoryDiagnoserAttribute"/> column <c>Allocated</c> should read
/// <c>0 B</c> in steady state — the min-heap uses <c>stackalloc</c> for small K.
/// </para>
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class TopKSamplerBenchmarks
{
    private float[] _srcLogits = null!;
    private float[] _scratch = null!;
    private TopKSampler _sampler = null!;
    private SamplerContext _context;

    /// <summary>Vocabulary size — matches typical Llama (128K) / SmolLM (49K) / TinyLlama (32K).</summary>
    [Params(32_000, 128_000)]
    public int VocabSize { get; set; }

    /// <summary>Top-K cutoff — 40 is the common default, 100 is a permissive upper bound.</summary>
    [Params(40, 100)]
    public int TopK { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _srcLogits = new float[VocabSize];
        _scratch = new float[VocabSize];
        for (int i = 0; i < VocabSize; i++)
            _srcLogits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);

        _sampler = new TopKSampler();
        _context = new SamplerContext(
            Temperature: 1.0f,
            TopK: TopK,
            TopP: 1.0f,
            MinP: 0f,
            Seed: null);
    }

    [Benchmark]
    public void TopK_Apply()
    {
        _srcLogits.AsSpan().CopyTo(_scratch);
        _sampler.Apply(_scratch, _context);
    }
}
