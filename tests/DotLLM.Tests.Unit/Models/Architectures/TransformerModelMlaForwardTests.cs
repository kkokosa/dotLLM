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
/// Stage-in tests for the MLA (DeepSeek-V2/V3) branch of
/// <see cref="TransformerModel"/>. Writes a synthetic tiny safetensors
/// checkpoint with the exact HF DeepSeek-V2 tensor naming and shapes, loads
/// it through <see cref="TransformerModel.LoadFromSafetensors"/>, and runs a
/// prefill forward to verify shape + finiteness + non-degenerate variance.
/// Covers both the LoRA-factored Q path (V2 full / V3) and the monolithic Q
/// path (V2-Lite, <c>q_lora_rank=0</c>).
/// </summary>
public sealed class TransformerModelMlaForwardTests : IDisposable
{
    // Tiny MLA-friendly shape. Keep qkRope even (RoPE requires pairs) and
    // kvLoraRank > 0 (MLA always factors the KV side). Two layers (both
    // dense, first_k_dense_replace=NumLayers ⇒ no MoE) keeps the fixture
    // compact while exercising per-layer pointer reuse.
    private const int HiddenSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 2;
    private const int VocabSize = 8;
    private const int QkNope = 4;
    private const int QkRope = 4;
    private const int VHead = 4;
    private const int KvLoraRank = 8;
    private const int IntermediateSize = 24;

    private readonly string _scratch;

    public TransformerModelMlaForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-mla-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Forward_LoRAQ_Prefill_FiniteLogits()
    {
        RunAndAssertFinite(qLoraRank: 8, seqLen: 3, seed: 42);
    }

    [Fact]
    public void Forward_MonolithicQ_Prefill_FiniteLogits()
    {
        // DeepSeek-V2-Lite skips Q factorisation (q_lora_rank = 0).
        RunAndAssertFinite(qLoraRank: 0, seqLen: 3, seed: 7);
    }

    [Fact]
    public void Forward_LoRAQ_SingleToken_FiniteLogits()
    {
        RunAndAssertFinite(qLoraRank: 8, seqLen: 1, seed: 123);
    }

    // ───────────────────────── core runner ─────────────────────────

    private void RunAndAssertFinite(int qLoraRank, int seqLen, int seed)
    {
        string path = Path.Combine(_scratch, $"mla-q{qLoraRank}-s{seqLen}.safetensors");
        WriteFixture(path, qLoraRank, seed);

        ModelConfig config = BuildConfig(qLoraRank);
        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, config);

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokenIds[i] = i % VocabSize;
            positions[i] = i;
        }

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(seqLen, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);

        var stats = ComputeStats(logits);
        Assert.Equal(stats.TotalCount, stats.FiniteCount);
        Assert.True(stats.StdDev > 0.0f,
            $"Logits degenerate: std={stats.StdDev} for qLoraRank={qLoraRank}, seqLen={seqLen}");
    }

    private static ModelConfig BuildConfig(int qLoraRank)
    {
        var mla = new MlaConfig
        {
            KvLoraRank = KvLoraRank,
            QLoraRank = qLoraRank,
            QkNopeHeadDim = QkNope,
            QkRopeHeadDim = QkRope,
            VHeadDim = VHead,
            RopeTheta = 10000.0f,
        };
        var rope = new RoPEConfig(Theta: 10000.0f, DimensionCount: QkNope + QkRope, Type: RoPEType.Norm);

        return new ModelConfig
        {
            Architecture = Architecture.DeepSeekV2,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,   // MLA is head-parallel on the expanded side
            HeadDim = QkNope + QkRope,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.MLA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            MlaConfig = mla,
            ChatTemplate = null,
        };
    }

    private static void WriteFixture(string path, int qLoraRank, int seed)
    {
        var b = new SafetensorsFixtureBuilder();
        int qkHead = QkNope + QkRope;
        int qTotal = NumHeads * qkHead;
        int kvADim = KvLoraRank + QkRope;
        int kvBOut = NumHeads * (QkNope + VHead);
        int oInput = NumHeads * VHead;

        // Globals
        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.05f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 1.0f, seed + 1, center: 1.0f, jitter: 0.05f);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.05f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 0, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 1, center: 1.0f, jitter: 0.05f);

            // MLA attention tensors
            if (qLoraRank > 0)
            {
                AddRand(b, $"{prefix}.self_attn.q_a_proj.weight", [qLoraRank, HiddenSize], 0.1f, s + 2);
                AddRand(b, $"{prefix}.self_attn.q_a_layernorm.weight", [qLoraRank],
                        amplitude: 0.05f, seed: s + 3, center: 1.0f, jitter: 0.05f);
                AddRand(b, $"{prefix}.self_attn.q_b_proj.weight", [qTotal, qLoraRank], 0.1f, s + 4);
            }
            else
            {
                AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qTotal, HiddenSize], 0.1f, s + 2);
            }
            AddRand(b, $"{prefix}.self_attn.kv_a_proj_with_mqa.weight", [kvADim, HiddenSize], 0.1f, s + 5);
            AddRand(b, $"{prefix}.self_attn.kv_a_layernorm.weight", [KvLoraRank],
                    amplitude: 0.05f, seed: s + 6, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.self_attn.kv_b_proj.weight", [kvBOut, KvLoraRank], 0.1f, s + 7);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, oInput], 0.1f, s + 8);

            // Dense FFN (first_k_dense_replace = NumLayers ⇒ every layer is dense).
            AddRand(b, $"{prefix}.mlp.gate_proj.weight", [IntermediateSize, HiddenSize], 0.05f, s + 9);
            AddRand(b, $"{prefix}.mlp.up_proj.weight", [IntermediateSize, HiddenSize], 0.05f, s + 10);
            AddRand(b, $"{prefix}.mlp.down_proj.weight", [HiddenSize, IntermediateSize], 0.05f, s + 11);
        }

        b.WriteTo(path);
    }

    /// <summary>
    /// Deterministic small-magnitude cos-based fill (shares style with
    /// <see cref="Mamba3TransformerModelTests"/>). Optional <paramref name="center"/>
    /// lets us emit near-unity norm weights (1 ± <paramref name="jitter"/>).
    /// </summary>
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
            if (jitter > 0f)
                values[i] = center + jitter * cos;
            else
                values[i] = amplitude * cos;
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
