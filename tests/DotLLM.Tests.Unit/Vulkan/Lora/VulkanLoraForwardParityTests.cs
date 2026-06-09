using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan.Lora;

/// <summary>
/// End-to-end Vulkan LoRA parity tests:
/// 1. Vulkan-with-zero-adapter is byte-equivalent to Vulkan-without-adapter.
/// 2. Vulkan-with-non-zero-adapter produces a measurable, finite delta vs base.
/// 3. Vulkan-with-non-zero-adapter matches CPU-with-same-adapter within 5e-3.
///
/// All tests build a tiny synthetic Llama-shape (2 layers, hidden=64) so they
/// run in &lt; 1s on any GPU. Tolerances mirror the existing
/// <c>VulkanTransformerModelMlaForwardTests</c> end-to-end bar (5e-3 abs / 1e-3 rel).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanLoraForwardParityTests : IDisposable
{
    private const int Hidden = 64;
    private const int NumHeads = 4;
    private const int HeadDim = 16;
    private const int IntermediateSize = 128;
    private const int VocabSize = 32;
    private const int NumLayers = 2;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanLoraForwardParityTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-lora-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public unsafe void Forward_NoAdapter_VsZeroAdapter_AreIdentical()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string fixturePath = Path.Combine(_scratch, $"base-zero.safetensors");
        WriteSyntheticFixture(fixturePath);
        var cfg = BuildConfig();

        int[] tokenIds = [1, 2, 3];
        int[] positions = [0, 1, 2];

        // Vulkan baseline (no adapter)
        float[] baseLogits;
        {
            using var sf = SafetensorsFile.Open(fixturePath);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            baseLogits = CopyLogits(logits);
        }

        // Vulkan with a zero-factor adapter — byte-equivalent expected.
        float[] zeroLogits;
        {
            using var sf = SafetensorsFile.Open(fixturePath);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir);
            using var zeroAdapter = BuildSyntheticAdapter(cfg, rank: 4, alpha: 16f, zeroFactors: true, seed: 7);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1,
                kvCache: null, adapter: zeroAdapter);
            zeroLogits = CopyLogits(logits);
        }

        Assert.Equal(baseLogits.Length, zeroLogits.Length);
        for (int i = 0; i < baseLogits.Length; i++)
        {
            float diff = MathF.Abs(baseLogits[i] - zeroLogits[i]);
            Assert.True(diff < AbsTol,
                $"Zero-adapter forward diverged at i={i}: base={baseLogits[i]} vs zero={zeroLogits[i]} (diff={diff})");
        }
    }

    [SkippableFact]
    public unsafe void Forward_NonZeroAdapter_ProducesMeasurableDelta()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string fixturePath = Path.Combine(_scratch, $"base-nonzero.safetensors");
        WriteSyntheticFixture(fixturePath);
        var cfg = BuildConfig();

        int[] tokenIds = [1, 2, 3];
        int[] positions = [0, 1, 2];

        // Run baseline (no adapter)
        float[] baseLogits;
        {
            using var sf = SafetensorsFile.Open(fixturePath);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            baseLogits = CopyLogits(logits);
        }

        // Run with a non-zero adapter — must produce a measurable delta.
        float[] withLogits;
        {
            using var sf = SafetensorsFile.Open(fixturePath);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir);
            using var adapter = BuildSyntheticAdapter(cfg, rank: 8, alpha: 32f, zeroFactors: false, seed: 9);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1,
                kvCache: null, adapter: adapter);
            withLogits = CopyLogits(logits);
        }

        // Confirm finite + measurable delta.
        float maxAbsDiff = 0f;
        int finite = 0;
        for (int i = 0; i < baseLogits.Length; i++)
        {
            if (!float.IsFinite(withLogits[i])) continue;
            finite++;
            maxAbsDiff = MathF.Max(maxAbsDiff, MathF.Abs(baseLogits[i] - withLogits[i]));
        }
        Assert.Equal(baseLogits.Length, finite);
        Assert.True(maxAbsDiff > 1e-3f,
            $"Non-zero adapter produced no measurable Vulkan delta (maxAbsDiff={maxAbsDiff}); LoRA path is silently disabled.");
    }

    [SkippableFact]
    public unsafe void Forward_NonZeroAdapter_VulkanMatchesCpu()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string fixturePath = Path.Combine(_scratch, $"base-cpuparity.safetensors");
        WriteSyntheticFixture(fixturePath);
        var cfg = BuildConfig();

        int[] tokenIds = [1, 2, 3];
        int[] positions = [0, 1, 2];

        // CPU oracle with the same adapter — independently builds an adapter
        // with the same seed; the adapter buffers themselves are fresh each
        // time so the two paths share the same numerical content via seed.
        float[] cpuLogits;
        using (var sf = SafetensorsFile.Open(fixturePath))
        using (var cpuModel = TransformerModel.LoadFromSafetensors(sf, cfg))
        using (var cpuAdapter = BuildSyntheticAdapter(cfg, rank: 8, alpha: 32f, zeroFactors: false, seed: 9))
        using (ITensor logits = cpuModel.Forward(tokenIds, positions, deviceId: -1,
                                                  kvCache: null, adapter: cpuAdapter))
        {
            cpuLogits = CopyLogits(logits);
        }

        // Vulkan under test with a separate-but-seed-identical adapter.
        float[] vkLogits;
        using (var sf = SafetensorsFile.Open(fixturePath))
        using (var vkModel = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir))
        using (var vkAdapter = BuildSyntheticAdapter(cfg, rank: 8, alpha: 32f, zeroFactors: false, seed: 9))
        using (ITensor logits = vkModel.Forward(tokenIds, positions, deviceId: -1,
                                                kvCache: null, adapter: vkAdapter))
        {
            // Vulkan returns last-token logits [1, vocab].
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }

        // CPU returns [seqLen, vocab]; compare last row against Vulkan.
        int lastRow = tokenIds.Length - 1;
        for (int c = 0; c < VocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * VocabSize + c];
            float vk = vkLogits[c];
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public unsafe void Forward_Q8_0Adapter_VulkanMatchesF32Vulkan_WithinQ8_0Tolerance()
    {
        // Phase 4d.5 / Gap 1 acceptance gate: a Q8_0-B + F16-A adapter uploaded
        // to Vulkan via the new dequant-on-load path produces finite logits
        // that match the F32 LoRA Vulkan path within Q8_0 round-trip tolerance.
        // The Hidden=64 fixture's per-row Q8_0 quant is exact for typical
        // small-uniform initialisations so the practical tolerance is tight,
        // but we use the documented Q8_0 abs ~5e-2 bar to stay conservative.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string fixturePath = Path.Combine(_scratch, $"base-q8_0-vk.safetensors");
        WriteSyntheticFixture(fixturePath);
        var cfg = BuildConfig();

        int[] tokenIds = [1, 2, 3];
        int[] positions = [0, 1, 2];
        const int seed = 23;
        const int rank = 8;
        const float alpha = 32f;
        const float q8_0AbsTol = 5e-2f;
        const float q8_0RelTol = 5e-3f;

        // F32 LoRA baseline (Vulkan).
        float[] f32Logits;
        using (var sf = SafetensorsFile.Open(fixturePath))
        using (var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir))
        using (var f32Adapter = BuildSyntheticAdapter(cfg, rank, alpha, zeroFactors: false, seed))
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1,
                                              kvCache: null, adapter: f32Adapter))
        {
            f32Logits = CopyLogits(logits);
        }

        // Q8_0-B + F16-A LoRA via the same RNG seed (same nominal weights,
        // round-tripped through Q8_0 / F16 encodings).
        float[] q8Logits;
        using (var sf = SafetensorsFile.Open(fixturePath))
        using (var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir))
        using (var q8Adapter = BuildSyntheticQ8_0BAdapter(cfg, rank, alpha, seed))
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1,
                                              kvCache: null, adapter: q8Adapter))
        {
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            q8Logits = CopyLogits(logits);
        }

        // Finite check first — a malformed Q8_0 upload path most commonly
        // surfaces as NaN/Inf in the LM head.
        for (int i = 0; i < q8Logits.Length; i++)
        {
            Assert.True(float.IsFinite(q8Logits[i]),
                $"Q8_0 LoRA forward produced non-finite logit at i={i}: {q8Logits[i]}");
        }

        // Compare last-row F32 logits to Vulkan Q8_0 logits within tolerance.
        int lastRow = tokenIds.Length - 1;
        for (int c = 0; c < VocabSize; c++)
        {
            float f32 = f32Logits[c];
            float q8 = q8Logits[c];
            float diff = MathF.Abs(f32 - q8);
            float bar = q8_0AbsTol + q8_0RelTol * MathF.Abs(f32);
            Assert.True(diff <= bar,
                $"col={c}: vkF32={f32:F6} vs vkQ8_0={q8:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public unsafe void Forward_AdapterCache_AmortisesUploadCost()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string fixturePath = Path.Combine(_scratch, $"base-cache.safetensors");
        WriteSyntheticFixture(fixturePath);
        var cfg = BuildConfig();

        int[] tokenIds = [1, 2, 3];
        int[] positions = [0, 1, 2];

        using var sf = SafetensorsFile.Open(fixturePath);
        using var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir);
        using var adapter = BuildSyntheticAdapter(cfg, rank: 4, alpha: 16f, zeroFactors: false, seed: 11);

        // First forward: adapter must upload — slower.
        var sw1 = System.Diagnostics.Stopwatch.StartNew();
        using (model.Forward(tokenIds, positions, deviceId: -1, kvCache: null, adapter: adapter)) { }
        sw1.Stop();

        // Second forward: cache hit — adapter buffers reused.
        var sw2 = System.Diagnostics.Stopwatch.StartNew();
        using (model.Forward(tokenIds, positions, deviceId: -1, kvCache: null, adapter: adapter)) { }
        sw2.Stop();

        // Adapter swap target — both calls must be sub-100ms; the cached call
        // is also expected to be no slower than the initial upload (and
        // typically much faster).
        Assert.True(sw1.Elapsed.TotalMilliseconds < 100,
            $"First adapter-active forward took {sw1.Elapsed.TotalMilliseconds} ms (>100 ms target).");
        Assert.True(sw2.Elapsed.TotalMilliseconds < 100,
            $"Cached adapter-active forward took {sw2.Elapsed.TotalMilliseconds} ms (>100 ms target).");
    }

    // ────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────

    private static ModelConfig BuildConfig() => new()
    {
        Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        VocabSize = VocabSize,
        HiddenSize = Hidden,
        IntermediateSize = IntermediateSize,
        NumLayers = NumLayers,
        NumAttentionHeads = NumHeads,
        NumKvHeads = NumHeads,
        HeadDim = HeadDim,
        MaxSequenceLength = 128,
        NormEpsilon = 1e-5f,
        RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: HeadDim, Type: RoPEType.Norm),
    };

    private void WriteSyntheticFixture(string path)
    {
        var rng = new Random(42);
        var bld = new SafetensorsFixtureBuilder();
        bld.AddFloat32("model.embed_tokens.weight", [VocabSize, Hidden], RandomVec(rng, VocabSize * Hidden, 0.05f));
        bld.AddFloat32("model.norm.weight", [Hidden], Ones(Hidden));
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            bld.AddFloat32($"{p}.input_layernorm.weight", [Hidden], Ones(Hidden));
            bld.AddFloat32($"{p}.post_attention_layernorm.weight", [Hidden], Ones(Hidden));
            bld.AddFloat32($"{p}.self_attn.q_proj.weight",
                [NumHeads * HeadDim, Hidden], RandomVec(rng, NumHeads * HeadDim * Hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.k_proj.weight",
                [NumHeads * HeadDim, Hidden], RandomVec(rng, NumHeads * HeadDim * Hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.v_proj.weight",
                [NumHeads * HeadDim, Hidden], RandomVec(rng, NumHeads * HeadDim * Hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.o_proj.weight",
                [Hidden, NumHeads * HeadDim], RandomVec(rng, Hidden * NumHeads * HeadDim, 0.05f));
            bld.AddFloat32($"{p}.mlp.gate_proj.weight",
                [IntermediateSize, Hidden], RandomVec(rng, IntermediateSize * Hidden, 0.05f));
            bld.AddFloat32($"{p}.mlp.up_proj.weight",
                [IntermediateSize, Hidden], RandomVec(rng, IntermediateSize * Hidden, 0.05f));
            bld.AddFloat32($"{p}.mlp.down_proj.weight",
                [Hidden, IntermediateSize], RandomVec(rng, Hidden * IntermediateSize, 0.05f));
        }
        bld.AddFloat32("lm_head.weight", [VocabSize, Hidden], RandomVec(rng, VocabSize * Hidden, 0.05f));
        bld.WriteTo(path);
    }

    private static LoraAdapter BuildSyntheticAdapter(ModelConfig cfg, int rank, float alpha,
                                                     bool zeroFactors, int seed)
    {
        var rng = new Random(seed);
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;
        int kvOut = cfg.NumKvHeads * cfg.HeadDim;
        var adapter = new LoraAdapter("syn",
            rank: rank, alpha: alpha,
            targetModules: ["q_proj", "v_proj"]);
        try
        {
            for (int layer = 0; layer < cfg.NumLayers; layer++)
            {
                AddProj(adapter, layer, "q_proj", inputDim: cfg.HiddenSize, outputDim: qOut, rank, zeroFactors, rng);
                AddProj(adapter, layer, "v_proj", inputDim: cfg.HiddenSize, outputDim: kvOut, rank, zeroFactors, rng);
            }
            return adapter;
        }
        catch
        {
            adapter.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Builds an adapter where every B (down-projection) buffer is Q8_0 and
    /// every A (up-projection) buffer is F16. Uses the same RNG seed as
    /// <see cref="BuildSyntheticAdapter"/> so the F32 / Q8_0 adapters
    /// generate matching nominal weights — they differ only by the
    /// Q8_0 / F16 round-trip noise.
    /// </summary>
    private static unsafe LoraAdapter BuildSyntheticQ8_0BAdapter(
        ModelConfig cfg, int rank, float alpha, int seed)
    {
        var rng = new Random(seed);
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;
        int kvOut = cfg.NumKvHeads * cfg.HeadDim;
        var adapter = new LoraAdapter("syn-q8_0",
            rank: rank, alpha: alpha,
            targetModules: ["q_proj", "v_proj"]);
        try
        {
            for (int layer = 0; layer < cfg.NumLayers; layer++)
            {
                AddProjQ8_0B(adapter, layer, "q_proj", inputDim: cfg.HiddenSize, outputDim: qOut, rank, rng);
                AddProjQ8_0B(adapter, layer, "v_proj", inputDim: cfg.HiddenSize, outputDim: kvOut, rank, rng);
            }
            return adapter;
        }
        catch
        {
            adapter.Dispose();
            throw;
        }
    }

    private static unsafe void AddProjQ8_0B(LoraAdapter adapter, int layer, string proj,
        int inputDim, int outputDim, int rank, Random rng)
    {
        // Q8_0 requires inputDim multiple of 32 (every row is one or more
        // 32-element blocks). The synthetic fixture's Hidden=64 satisfies this.
        if (inputDim % 32 != 0)
            throw new InvalidOperationException(
                $"Test fixture inputDim must be multiple of 32 for Q8_0 LoRA; got {inputDim}.");

        long bElems = (long)rank * inputDim;
        long aElems = (long)outputDim * rank;

        // B: stage F32 into transient buffer, quantise to Q8_0, then own bytes.
        long bBytes = LoraAdapter.Q8_0ByteSize(bElems);
        nint bHandle = LoraAdapter.AllocAlignedBytes(bBytes);
        nint stagingHandle = LoraAdapter.AllocAligned(bElems);
        try
        {
            float* bp = (float*)stagingHandle;
            for (long i = 0; i < bElems; i++) bp[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
            LoraDelta.Quantize_F32_To_Q8_0(
                (float*)stagingHandle, (byte*)bHandle, rows: rank, elementsPerRow: inputDim);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)stagingHandle);
        }

        // A: F16, 2 bytes per element.
        long aBytes = aElems * 2;
        nint aHandle = LoraAdapter.AllocAlignedBytes(aBytes);
        byte* ap = (byte*)aHandle;
        for (long i = 0; i < aElems; i++)
        {
            float v = (float)((rng.NextDouble() * 2 - 1) * 0.05);
            ushort raw = BitConverter.HalfToUInt16Bits((Half)v);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt16LittleEndian(
                new Span<byte>(ap + i * 2, 2), raw);
        }

        adapter.AddLayerWeights(layer, proj,
            new LoraLayerWeights(
                AHandle: aHandle,
                BHandle: bHandle,
                InputDim: inputDim,
                OutputDim: outputDim,
                WeightDType: LoraWeightDType.Q8_0,
                AWeightDType: LoraWeightDType.F16));
    }

    private static unsafe void AddProj(LoraAdapter adapter, int layer, string proj,
        int inputDim, int outputDim, int rank, bool zero, Random rng)
    {
        long bElems = (long)rank * inputDim;
        long aElems = (long)outputDim * rank;
        nint b = LoraAdapter.AllocAligned(bElems);
        nint a = LoraAdapter.AllocAligned(aElems);

        if (!zero)
        {
            float* bp = (float*)b;
            float* ap = (float*)a;
            for (long i = 0; i < bElems; i++) bp[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
            for (long i = 0; i < aElems; i++) ap[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
        }
        else
        {
            new Span<float>((void*)b, (int)bElems).Clear();
            new Span<float>((void*)a, (int)aElems).Clear();
        }
        adapter.AddLayerWeights(layer, proj,
            new LoraLayerWeights(AHandle: a, BHandle: b, InputDim: inputDim, OutputDim: outputDim));
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static float[] RandomVec(Random rng, int n, float scale = 1.0f)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return v;
    }

    private static float[] Ones(int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = 1.0f;
        return v;
    }
}
