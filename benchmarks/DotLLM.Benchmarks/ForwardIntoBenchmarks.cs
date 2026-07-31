using BenchmarkDotNet.Attributes;
using DotLLM.Core.Tensors;

namespace DotLLM.Benchmarks;

/// <summary>
/// Microbenchmarks for the LM-head buffer routing change. Three return paths are measured,
/// all with the (identical, unmeasured) LM-head matmul excluded so only the per-call
/// buffer overhead shows up:
/// <list type="number">
///   <item><b><c>PreRefactor_AllocCopyFree</c></b> — the shape of the return path <i>before</i>
///     this PR: allocate a fresh <see cref="UnmanagedTensor"/>, memcpy
///     <c>vocabSize × sizeof(float)</c> bytes from <c>_state.Logits</c> into it, free on
///     <see cref="IDisposable.Dispose"/>. Kept for historical context only — no current
///     code path does this.</item>
///   <item><b><c>Forward_AllocOnly</c></b> — what <c>TransformerModel.Forward</c> does
///     <i>today</i>: the matmul writes straight into the freshly-allocated tensor, so the
///     copy is gone but the per-call allocate/free remains. This is the baseline
///     <c>ForwardInto</c> is actually compared against.</item>
///   <item><b><c>ForwardInto_DirectWrite</c></b> — the matmul writes straight into the
///     caller's pinned span: no allocation, no copy, no free.</item>
/// </list>
/// <para>
/// At per-token decode rates (TextGenerator hot path) this overhead runs once per generated
/// token. <see cref="MemoryDiagnoserAttribute"/> tracks the native-alloc elimination too.
/// </para>
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public unsafe class ForwardIntoBenchmarks
{
    private float[] _stateLogits = null!;
    private float[] _callerBuffer = null!;

    /// <summary>Vocab size — matches typical Llama (128K) / SmolLM (49K) / TinyLlama (32K).</summary>
    [Params(32_000, 49_152, 128_000)]
    public int VocabSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _stateLogits = new float[VocabSize];
        _callerBuffer = new float[VocabSize];
        for (int i = 0; i < VocabSize; i++)
            _stateLogits[i] = (float)(rng.NextDouble() * 20.0 - 10.0);
    }

    /// <summary>
    /// Historical (pre-PR) return-path overhead: NativeMemory.AlignedAlloc + memcpy from
    /// _state.Logits + Dispose (which calls NativeMemory.AlignedFree). No current code
    /// path does this; it is here to show what the copy alone cost.
    /// Touches the destination to prevent dead-code elimination.
    /// </summary>
    [Benchmark]
    public void PreRefactor_AllocCopyFree()
    {
        var shape = new TensorShape(1, VocabSize);
        using var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        _stateLogits.AsSpan().CopyTo(new Span<float>((void*)result.DataPointer, VocabSize));

        // Touch the result to keep the copy live.
        ((float*)result.DataPointer)[0] += 0f;
    }

    /// <summary>
    /// Current <c>Forward</c> return-path overhead: NativeMemory.AlignedAlloc + Dispose
    /// (NativeMemory.AlignedFree), with no copy — the LM-head matmul (not part of this
    /// microbench) writes directly into the allocated tensor. This is the meaningful
    /// baseline for <see cref="ForwardInto_DirectWrite"/>.
    /// </summary>
    [Benchmark(Baseline = true)]
    public void Forward_AllocOnly()
    {
        var shape = new TensorShape(1, VocabSize);
        using var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);

        // Touch the destination to keep the allocation live.
        ((float*)result.DataPointer)[0] += 0f;
    }

    /// <summary>
    /// ForwardInto return path: no allocation, no copy — the LM-head matmul writes
    /// directly into the caller's buffer. The only operation in this benchmark body
    /// is the equivalent "touch" so the comparison stays fair.
    /// </summary>
    [Benchmark]
    public void ForwardInto_DirectWrite()
    {
        fixed (float* dst = _callerBuffer)
        {
            // No alloc, no copy. The matmul (not part of this microbench) writes
            // directly into `dst` in the real ForwardInto path.
            dst[0] += 0f;
        }
    }
}
