using System.Buffers.Binary;
using System.Runtime.InteropServices;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Benchmarks.Lora;

/// <summary>
/// Builds a deterministic, in-memory synthetic LoRA adapter that covers every
/// canonical attention + FFN projection on every layer of a base model.
/// Used by the macro-benchmark (Phase 4d.3) to exercise the full LoRA
/// dispatch path without shipping a real adapter checkpoint.
/// </summary>
/// <remarks>
/// <para>
/// We populate <c>q_proj</c>, <c>k_proj</c>, <c>v_proj</c>, <c>o_proj</c>,
/// <c>gate_proj</c>, <c>up_proj</c>, <c>down_proj</c> for every layer
/// <c>[0, baseConfig.NumLayers)</c> — the same projection set the standard
/// <see cref="DotLLM.Models.Architectures.TransformerModel"/> dispatch sites
/// look up via <see cref="ILoraAdapter.GetLayerWeights"/>. Each per-projection
/// <c>(A, B)</c> pair has shape:
/// </para>
/// <list type="bullet">
/// <item><c>B</c>: row-major <c>[rank, inputDim]</c></item>
/// <item><c>A</c>: row-major <c>[outputDim, rank]</c></item>
/// </list>
/// <para>
/// Values are drawn from a fixed-seed RNG so consecutive runs of the
/// benchmark see the same bytes — the macro-bench is a perf measurement,
/// not a correctness test, and determinism keeps the JIT warm-up curve
/// reproducible across iterations.
/// </para>
/// <para>
/// All buffers are 64-byte-aligned via <see cref="NativeMemory.AlignedAlloc(nuint, nuint)"/>
/// so the resulting <see cref="LoraAdapter"/> disposes them cleanly via
/// <see cref="NativeMemory.AlignedFree"/> on the standard
/// <see cref="LoraAdapter.Dispose"/> path.
/// </para>
/// </remarks>
internal static unsafe class SyntheticLoraAdapter
{
    /// <summary>Standard projection names exercised by the macro-bench.</summary>
    public static readonly string[] AllTargetProjections =
    [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ];

    /// <summary>
    /// Creates a fully-populated synthetic adapter for the given
    /// <paramref name="baseConfig"/>. Caller owns the returned
    /// <see cref="LoraAdapter"/> and must <see cref="LoraAdapter.Dispose"/> it.
    /// </summary>
    /// <param name="name">Adapter name (informational only).</param>
    /// <param name="baseConfig">The base model's config — drives per-projection shapes.</param>
    /// <param name="rank">LoRA rank (typical PEFT default is 16).</param>
    /// <param name="alpha">LoRA alpha (typical PEFT default is 32, i.e. 2*rank).</param>
    /// <param name="dtype">Storage dtype for both A and B buffers.</param>
    /// <param name="seed">Deterministic RNG seed for buffer fill.</param>
    public static LoraAdapter Create(
        string name,
        ModelConfig baseConfig,
        int rank,
        float alpha,
        LoraWeightDType dtype,
        int seed)
    {
        ArgumentNullException.ThrowIfNull(baseConfig);
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank));

        int hidden = baseConfig.HiddenSize;
        int qOut = baseConfig.NumAttentionHeads * baseConfig.HeadDim;
        int kvOut = baseConfig.NumKvHeads * baseConfig.HeadDim;
        int ffn = baseConfig.IntermediateSize;

        var adapter = new LoraAdapter(name, rank, alpha, AllTargetProjections);

        try
        {
            var rng = new Random(seed);

            for (int layer = 0; layer < baseConfig.NumLayers; layer++)
            {
                AddProjection(adapter, layer, "q_proj", hidden, qOut, rank, dtype, rng);
                AddProjection(adapter, layer, "k_proj", hidden, kvOut, rank, dtype, rng);
                AddProjection(adapter, layer, "v_proj", hidden, kvOut, rank, dtype, rng);
                AddProjection(adapter, layer, "o_proj", qOut, hidden, rank, dtype, rng);
                AddProjection(adapter, layer, "gate_proj", hidden, ffn, rank, dtype, rng);
                AddProjection(adapter, layer, "up_proj", hidden, ffn, rank, dtype, rng);
                AddProjection(adapter, layer, "down_proj", ffn, hidden, rank, dtype, rng);
            }
        }
        catch
        {
            adapter.Dispose();
            throw;
        }

        return adapter;
    }

    /// <summary>
    /// Phase 4d.4 Q8_0-B variant. Builds an adapter where every B
    /// (down-projection) buffer is Q8_0-quantised and every A
    /// (up-projection) buffer is F16. The same RNG seed produces the same
    /// underlying F32 weights as <see cref="Create"/> with
    /// <see cref="LoraWeightDType.F16"/> — only B differs by the Q8_0
    /// round-trip error.
    /// </summary>
    public static LoraAdapter CreateQ8_0B(
        string name,
        ModelConfig baseConfig,
        int rank,
        float alpha,
        int seed)
    {
        ArgumentNullException.ThrowIfNull(baseConfig);
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank));

        int hidden = baseConfig.HiddenSize;
        int qOut = baseConfig.NumAttentionHeads * baseConfig.HeadDim;
        int kvOut = baseConfig.NumKvHeads * baseConfig.HeadDim;
        int ffn = baseConfig.IntermediateSize;

        var adapter = new LoraAdapter(name, rank, alpha, AllTargetProjections);

        try
        {
            var rng = new Random(seed);

            for (int layer = 0; layer < baseConfig.NumLayers; layer++)
            {
                AddProjectionQ8_0B(adapter, layer, "q_proj", hidden, qOut, rank, rng);
                AddProjectionQ8_0B(adapter, layer, "k_proj", hidden, kvOut, rank, rng);
                AddProjectionQ8_0B(adapter, layer, "v_proj", hidden, kvOut, rank, rng);
                AddProjectionQ8_0B(adapter, layer, "o_proj", qOut, hidden, rank, rng);
                AddProjectionQ8_0B(adapter, layer, "gate_proj", hidden, ffn, rank, rng);
                AddProjectionQ8_0B(adapter, layer, "up_proj", hidden, ffn, rank, rng);
                AddProjectionQ8_0B(adapter, layer, "down_proj", ffn, hidden, rank, rng);
            }
        }
        catch
        {
            adapter.Dispose();
            throw;
        }

        return adapter;
    }

    private static void AddProjectionQ8_0B(
        LoraAdapter adapter,
        int layer,
        string projName,
        int inputDim,
        int outputDim,
        int rank,
        Random rng)
    {
        long bElems = (long)rank * inputDim;       // B: [rank, inputDim]
        long aElems = (long)outputDim * rank;       // A: [outputDim, rank]

        // B: generate F32 noise, quantise to Q8_0 in place via a transient
        // F32 staging buffer (adapter-load is one-shot — no perf concern).
        long bBytes = LoraAdapter.Q8_0ByteSize(bElems);
        nint bHandle = LoraAdapter.AllocAlignedBytes(bBytes);

        // Use unmanaged staging so a >2GB adapter doesn't pin the GC heap.
        nint stagingHandle = LoraAdapter.AllocAligned(bElems);
        try
        {
            FillRandomF32((float*)stagingHandle, bElems, rng);
            LoraDelta.Quantize_F32_To_Q8_0(
                (float*)stagingHandle, (byte*)bHandle, rows: rank, elementsPerRow: inputDim);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)stagingHandle);
        }

        // A: F16 (2 bytes per element).
        long aBytes = aElems * 2;
        nint aHandle = LoraAdapter.AllocAlignedBytes(aBytes);
        FillRandomHalfWidth((byte*)aHandle, aElems, rng, LoraWeightDType.F16);

        adapter.AddLayerWeights(layer, projName, new LoraLayerWeights(
            AHandle: aHandle,
            BHandle: bHandle,
            InputDim: inputDim,
            OutputDim: outputDim,
            WeightDType: LoraWeightDType.Q8_0,
            AWeightDType: LoraWeightDType.F16));
    }

    private static void AddProjection(
        LoraAdapter adapter,
        int layer,
        string projName,
        int inputDim,
        int outputDim,
        int rank,
        LoraWeightDType dtype,
        Random rng)
    {
        long bElems = (long)rank * inputDim;       // B: [rank, inputDim]
        long aElems = (long)outputDim * rank;       // A: [outputDim, rank]

        nint bHandle;
        nint aHandle;
        if (dtype == LoraWeightDType.F32)
        {
            bHandle = LoraAdapter.AllocAligned(bElems);
            aHandle = LoraAdapter.AllocAligned(aElems);
            FillRandomF32((float*)bHandle, bElems, rng);
            FillRandomF32((float*)aHandle, aElems, rng);
        }
        else
        {
            // Both F16 and BF16 are 2 bytes per element.
            long bBytes = bElems * 2;
            long aBytes = aElems * 2;
            bHandle = (nint)NativeMemory.AlignedAlloc((nuint)bBytes, 64);
            aHandle = (nint)NativeMemory.AlignedAlloc((nuint)aBytes, 64);
            FillRandomHalfWidth((byte*)bHandle, bElems, rng, dtype);
            FillRandomHalfWidth((byte*)aHandle, aElems, rng, dtype);
        }

        adapter.AddLayerWeights(layer, projName, new LoraLayerWeights(
            AHandle: aHandle,
            BHandle: bHandle,
            InputDim: inputDim,
            OutputDim: outputDim,
            WeightDType: dtype));
    }

    /// <summary>
    /// Fills <paramref name="dst"/> with N(0, sigma) noise scaled small enough
    /// to avoid swamping the base activations. PEFT-trained adapters are
    /// initialised so <c>A · B</c> starts at zero; we use a small (~0.02)
    /// std-dev to mimic post-training adapters where the delta is non-zero
    /// but small relative to the base.
    /// </summary>
    private static void FillRandomF32(float* dst, long count, Random rng)
    {
        const float scale = 0.02f;
        for (long i = 0; i < count; i++)
            dst[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
    }

    /// <summary>
    /// Fills a 2-byte-element buffer (F16 or BF16) with deterministic noise.
    /// We generate an F32 sample, then encode it into the dtype's wire layout
    /// matching what <see cref="DotLLM.Cpu.Kernels.LoraDelta"/> reads back.
    /// </summary>
    private static void FillRandomHalfWidth(byte* dst, long count, Random rng, LoraWeightDType dtype)
    {
        const float scale = 0.02f;
        for (long i = 0; i < count; i++)
        {
            float v = ((float)rng.NextDouble() * 2f - 1f) * scale;

            if (dtype == LoraWeightDType.F16)
            {
                ushort raw = BitConverter.HalfToUInt16Bits((Half)v);
                BinaryPrimitives.WriteUInt16LittleEndian(
                    new Span<byte>(dst + i * 2, 2), raw);
            }
            else
            {
                // BF16: top 16 bits of the F32 representation.
                uint bits = BitConverter.SingleToUInt32Bits(v);
                ushort raw = (ushort)(bits >> 16);
                BinaryPrimitives.WriteUInt16LittleEndian(
                    new Span<byte>(dst + i * 2, 2), raw);
            }
        }
    }
}
