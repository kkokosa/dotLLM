using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// End-to-end Gemma 4 forward-pass coverage. Mirrors
/// <see cref="TransformerModelGemma3ForwardTests"/> but uses the
/// <see cref="Architecture.Gemma4"/> enum variant so we exercise the
/// loader + forward path against the Gemma 4 dispatch row, not just
/// Gemma 3's. The forward path is byte-identical to Gemma 3 — Gemma 4
/// reuses every wired ModelConfig field — so this test mostly guards
/// against future divergence in the architecture dispatch (e.g. if a
/// later PR forks <c>TransformerModel.LoadFromSafetensors</c> per
/// Gemma variant).
/// </summary>
public sealed class TransformerModelGemma4ForwardTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 6;
    private const int NumHeads = 2;
    private const int VocabSize = 8;
    private const int HeadDim = HiddenSize / NumHeads; // 8
    private const int IntermediateSize = 24;
    private const int SlidingWindow = 2;

    private readonly string _scratch;

    public TransformerModelGemma4ForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma4-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// Canonical Gemma 4 forward pass: 5 sliding + 1 full layer, attention
    /// soft-cap, final soft-cap, and query-pre-attn-scalar all active —
    /// matching the real google/gemma-4-12B shape but on the synthetic tiny
    /// fixture. Asserts finiteness and that the final soft-cap clamps the
    /// logit magnitude band.
    /// </summary>
    [Fact]
    public void Forward_Gemma4_AllMechanisms_FiniteLogits()
    {
        string path = Path.Combine(_scratch, "gemma4-all.safetensors");
        WriteFixture(path, seed: 71);

        ModelConfig config = BuildConfig(
            withAttnSoftcap: 50.0f,
            withFinalSoftcap: 30.0f,
            withQueryPreAttnScalar: HeadDim);

        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, config);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);

        var stats = ComputeStats(logits);
        Assert.Equal(stats.TotalCount, stats.FiniteCount);
        Assert.True(stats.StdDev > 0.0f,
            $"Logits degenerate: std={stats.StdDev}");

        // Final-logit soft-cap saturates to (-cap, +cap).
        Assert.True(stats.Min > -30.0f && stats.Max < 30.0f,
            $"Final-logit soft-cap did not clamp: min={stats.Min}, max={stats.Max}");
    }

    /// <summary>
    /// Gemma 4 with only the public 12B/31B defaults (no attn-softcap, no
    /// QPAS override — the public Gemma 4 12B SKU leaves both unset) still
    /// produces finite, non-degenerate logits. The final-logit soft-cap
    /// stays on because the public 12B carries
    /// <c>final_logit_softcapping=30.0</c>.
    /// </summary>
    [Fact]
    public void Forward_Gemma4_PublicDefaults_FiniteLogits()
    {
        string path = Path.Combine(_scratch, "gemma4-public.safetensors");
        WriteFixture(path, seed: 419);

        ModelConfig config = BuildConfig(
            withAttnSoftcap: null,
            withFinalSoftcap: 30.0f,
            withQueryPreAttnScalar: null);

        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, config);

        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

        var stats = ComputeStats(logits);
        Assert.Equal(stats.TotalCount, stats.FiniteCount);
        Assert.True(stats.StdDev > 0.0f,
            $"Logits degenerate: std={stats.StdDev}");
    }

    // ───────────────────────── helpers ─────────────────────────

    private static ModelConfig BuildConfig(
        float? withAttnSoftcap, float? withFinalSoftcap, float? withQueryPreAttnScalar)
    {
        var rope = new RoPEConfig(
            Theta: 1_000_000.0f,
            DimensionCount: HeadDim,
            Type: RoPEType.NeoX);

        // Matches HF Gemma 4 12B's layer_types canonical pattern: 5
        // sliding + 1 full, looped. With NumLayers=6 we get one cycle.
        var perLayer = new int?[NumLayers]
        {
            SlidingWindow, SlidingWindow, SlidingWindow,
            SlidingWindow, SlidingWindow,
            null,
        };

        return new ModelConfig
        {
            Architecture = Architecture.Gemma4,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.GELUTanh,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            SlidingWindowSize = SlidingWindow,
            PerLayerSlidingWindow = perLayer,
            AttnLogitSoftcap = withAttnSoftcap,
            FinalLogitSoftcap = withFinalSoftcap,
            QueryPreAttnScalar = withQueryPreAttnScalar,
            MlaConfig = null,
            Moe = null,
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Synthetic safetensors fixture matching the dense Llama-style loader
    /// expectations. Identical structure to the Gemma 3 fixture (same
    /// tensor naming, F32 weights) — Gemma 4 reuses the dense forward path
    /// so the safetensors layout is shared.
    /// </summary>
    private static void WriteFixture(string path, int seed)
    {
        var b = new SafetensorsFixtureBuilder();
        int qStride = NumHeads * HeadDim;
        int kvStride = NumHeads * HeadDim;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.1f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 0.05f, seed + 1, center: 1.0f, jitter: 0.05f);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.1f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 0, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 1, center: 1.0f, jitter: 0.05f);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qStride, HiddenSize], 0.1f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight", [kvStride, HiddenSize], 0.1f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight", [kvStride, HiddenSize], 0.1f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, qStride], 0.1f, s + 5);

            AddRand(b, $"{prefix}.mlp.gate_proj.weight", [IntermediateSize, HiddenSize], 0.05f, s + 6);
            AddRand(b, $"{prefix}.mlp.up_proj.weight", [IntermediateSize, HiddenSize], 0.05f, s + 7);
            AddRand(b, $"{prefix}.mlp.down_proj.weight", [HiddenSize, IntermediateSize], 0.05f, s + 8);
        }

        b.WriteTo(path);
    }

    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape,
                                float amplitude, int seed,
                                float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            float cos = MathF.Cos(phi);
            values[i] = jitter > 0f ? center + jitter * cos : amplitude * cos;
        }
        b.AddFloat32(name, shape, values);
    }

    private static unsafe LogitStats ComputeStats(ITensor logits)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);

        int finite = 0;
        double sum = 0, sumSq = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (float v in span)
        {
            if (float.IsFinite(v))
            {
                finite++;
                sum += v;
                sumSq += (double)v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        double mean = finite > 0 ? sum / finite : 0.0;
        double variance = finite > 0 ? (sumSq / finite) - (mean * mean) : 0.0;
        double stddev = Math.Sqrt(Math.Max(0.0, variance));
        return new LogitStats(total, finite, (float)mean, (float)stddev, min, max);
    }

    private readonly record struct LogitStats(
        int TotalCount, int FiniteCount, float Mean, float StdDev, float Min, float Max);
}
