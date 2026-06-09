using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Synthetic-fixture tests for SmolLM3 end-to-end load + forward via
/// <see cref="TransformerModel.LoadFromSafetensors(SafetensorsFile, ModelConfig)"/>.
/// Tensor layout is identical to Llama (SmolLM3 differs only in NoPE
/// layer gating + optional YaRN), so the fixture builder is shared with
/// <see cref="TransformerSafetensorsLoadTests"/>.
/// </summary>
public sealed class SmolLM3SafetensorsLoadTests : IDisposable
{
    private const int Hidden = 64;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 16;
    private const int Intermediate = 128;
    private const int Vocab = 32;
    private const int NumLayers = 4;

    private readonly string _scratch;

    public SmolLM3SafetensorsLoadTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-smollm3-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// Builds a SmolLM3-shaped fixture (Llama-name tensors, GQA-2, small
    /// dims). NoPE layers are surfaced via the returned config, not the
    /// on-disk tensors — SmolLM3's tensor layout is identical to Llama.
    /// </summary>
    private string BuildSmolLm3Fixture(int seed)
    {
        var rng = new Random(seed);
        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [Vocab, Hidden], RandomVec(rng, Vocab * Hidden, 0.05f));
        b.AddFloat32("model.norm.weight", [Hidden], Ones(Hidden));

        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [Hidden], Ones(Hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [Hidden], Ones(Hidden));
            b.AddFloat32($"{p}.self_attn.q_proj.weight",
                [NumHeads * HeadDim, Hidden], RandomVec(rng, NumHeads * HeadDim * Hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.k_proj.weight",
                [NumKvHeads * HeadDim, Hidden], RandomVec(rng, NumKvHeads * HeadDim * Hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.v_proj.weight",
                [NumKvHeads * HeadDim, Hidden], RandomVec(rng, NumKvHeads * HeadDim * Hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.o_proj.weight",
                [Hidden, NumHeads * HeadDim], RandomVec(rng, Hidden * NumHeads * HeadDim, 0.05f));
            b.AddFloat32($"{p}.mlp.gate_proj.weight",
                [Intermediate, Hidden], RandomVec(rng, Intermediate * Hidden, 0.05f));
            b.AddFloat32($"{p}.mlp.up_proj.weight",
                [Intermediate, Hidden], RandomVec(rng, Intermediate * Hidden, 0.05f));
            b.AddFloat32($"{p}.mlp.down_proj.weight",
                [Hidden, Intermediate], RandomVec(rng, Hidden * Intermediate, 0.05f));
        }
        // SmolLM3-3B ships tie_word_embeddings=true; omit lm_head.weight.

        string path = Path.Combine(_scratch, $"smollm3-{seed:X8}.safetensors");
        b.WriteTo(path);
        return path;
    }

    private static ModelConfig BuildConfig(IReadOnlyList<int>? noRopeLayers, RoPEScalingType scalingType, float scalingFactor, int origMaxSeqLen)
        => new ModelConfig
        {
            Architecture = Architecture.SmolLM3,
            VocabSize = Vocab,
            HiddenSize = Hidden,
            IntermediateSize = Intermediate,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            HeadDim = HeadDim,
            // Use a generous max-seq so YaRN test can probe positions > orig.
            MaxSequenceLength = 64,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = true,
            RoPEConfig = new RoPEConfig(
                Theta: 10000.0f,
                DimensionCount: HeadDim,
                Type: RoPEType.Norm,
                ScalingType: scalingType,
                ScalingFactor: scalingFactor,
                OrigMaxSeqLen: origMaxSeqLen,
                AttnFactor: 1.0f,
                BetaFast: 32.0f,
                BetaSlow: 1.0f),
            NoRopeLayers = noRopeLayers,
        };

    [Fact]
    public void Forward_ProducesFiniteVocabLogits_WithNonzeroStddev()
    {
        // Smoke test — 4-layer SmolLM3 fixture with NoPE on every other layer
        // produces finite, varied logits across all 3 prompt positions.
        string path = BuildSmolLm3Fixture(seed: 42);
        using var file = SafetensorsFile.Open(path);
        var config = BuildConfig(noRopeLayers: new[] { 1, 3 }, RoPEScalingType.None, 1.0f, 0);

        using var model = TransformerModel.LoadFromSafetensors(file, config);
        using var logits = model.Forward(
            tokenIds: [0, 1, 2],
            positions: [0, 1, 2],
            deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(3, logits.Shape[0]);
        Assert.Equal(Vocab, logits.Shape[1]);

        var sample = SampleLogits(logits);
        Assert.All(sample, v => Assert.True(float.IsFinite(v), $"non-finite logit: {v}"));
        Assert.True(Stddev(sample) > 1e-4f, "logits stddev collapsed to ~0 — model degenerate");
    }

    /// <summary>
    /// NoPE invariant: when EVERY layer is marked NoPE, the forward pass
    /// must be bit-identical regardless of the supplied positions, because
    /// no RoPE rotation is applied to Q/K anywhere. This is the cleanest
    /// hook-free verification that the per-layer NoPE gating actually
    /// short-circuits the RoPE call.
    /// </summary>
    [Fact]
    public void AllNoPe_LogitsBitIdentical_AcrossPositions()
    {
        string path = BuildSmolLm3Fixture(seed: 7);
        using var file = SafetensorsFile.Open(path);

        var allNoPeLayers = Enumerable.Range(0, NumLayers).ToArray();
        var config = BuildConfig(noRopeLayers: allNoPeLayers, RoPEScalingType.None, 1.0f, 0);

        using var model = TransformerModel.LoadFromSafetensors(file, config);
        using var logitsA = model.Forward([0, 1, 2], [0, 1, 2], deviceId: -1);
        using var logitsB = model.Forward([0, 1, 2], [10, 20, 30], deviceId: -1);

        var a = SampleLogits(logitsA);
        var b = SampleLogits(logitsB);
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i]); // bit-equal — no RoPE applied anywhere
    }

    /// <summary>
    /// Cross-check on the gating: when NoPE is DISABLED everywhere, varying
    /// the positions array MUST change the output (otherwise the RoPE call
    /// is silently broken — separate concern from NoPE but the inverse of
    /// the test above and a useful regression bound).
    /// </summary>
    [Fact]
    public void NoNoPe_DifferentPositions_LogitsDiverge()
    {
        string path = BuildSmolLm3Fixture(seed: 7);
        using var file = SafetensorsFile.Open(path);

        var config = BuildConfig(noRopeLayers: null, RoPEScalingType.None, 1.0f, 0);

        using var model = TransformerModel.LoadFromSafetensors(file, config);
        using var logitsA = model.Forward([0, 1, 2], [0, 1, 2], deviceId: -1);
        using var logitsB = model.Forward([0, 1, 2], [10, 20, 30], deviceId: -1);

        var a = SampleLogits(logitsA);
        var b = SampleLogits(logitsB);
        Assert.Equal(a.Length, b.Length);
        bool anyDifferent = false;
        for (int i = 0; i < a.Length && !anyDifferent; i++)
            if (a[i] != b[i]) anyDifferent = true;
        Assert.True(anyDifferent, "positions had no effect on logits — RoPE call may be unreachable");
    }

    // YarnLongContext_PositionBeyondOrigMax_YarnAffectsLogits deferred —
    // requires the dense-path YaRN dispatch in TransformerModel +
    // RoPE.PrecomputeFrequencyTableYarn, which ship via the MLA chain
    // (#187 MLA-3, not yet on upstream main / this PR's ancestry). The
    // test will re-land alongside the dense-YaRN dispatch as a follow-up.

    /// <summary>
    /// Gated real-weight load test — points at
    /// <c>~/.dotllm/test-cache/HuggingFaceTB/SmolLM3-3B/</c>. Skipped when
    /// the checkpoint is absent (the default for CI). When present, runs a
    /// 3-token prefill and asserts finite vocab-sized logits. The user
    /// drops the safetensors / config.json into the cache; this test does
    /// not initiate the download.
    /// </summary>
    [Fact]
    public void RealWeights_Loads_And_ForwardsFiniteLogits_WhenAvailable()
    {
        string userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string checkpointDir = Path.Combine(
            userProfile, ".dotllm", "test-cache", "HuggingFaceTB", "SmolLM3-3B");
        string configPath = Path.Combine(checkpointDir, "config.json");
        if (!File.Exists(configPath))
            return; // gated — not part of CI

        var (model, source, config) = DotLLM.Models.ModelLoader.LoadFromSafetensors(checkpointDir);
        try
        {
            Assert.Equal(Architecture.SmolLM3, config.Architecture);
            using var logits = model.Forward(
                tokenIds: [0, 1, 2],
                positions: [0, 1, 2],
                deviceId: -1);
            Assert.Equal(config.VocabSize, logits.Shape[1]);
            var sample = SampleLogits(logits);
            Assert.All(sample, v => Assert.True(float.IsFinite(v)));
            Assert.True(Stddev(sample) > 1e-4f);
        }
        finally
        {
            (model as IDisposable)?.Dispose();
            source.Dispose();
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────

    private static float[] RandomVec(Random rng, int n, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
            v[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return v;
    }

    private static float[] Ones(int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = 1.0f;
        return v;
    }

    private static unsafe float[] SampleLogits(ITensor logits)
    {
        int seq = (int)logits.Shape[0];
        int vocab = (int)logits.Shape[1];
        var sample = new float[seq * vocab];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seq * vocab);
        span.CopyTo(sample);
        return sample;
    }

    private static float Stddev(float[] xs)
    {
        if (xs.Length == 0) return 0;
        double sum = 0;
        for (int i = 0; i < xs.Length; i++) sum += xs[i];
        double mean = sum / xs.Length;
        double ss = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            double d = xs[i] - mean;
            ss += d * d;
        }
        return (float)Math.Sqrt(ss / xs.Length);
    }

    private static bool FloatsEqual(float a, float b) => a.Equals(b);
}
