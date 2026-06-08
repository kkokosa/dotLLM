using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>
/// Synthetic-fixture tests for
/// <see cref="TransformerModel.LoadFromSafetensors(SafetensorsFile, ModelConfig)"/>.
/// Uses <see cref="SafetensorsFixtureBuilder"/> to write a byte-accurate
/// mini Llama-shaped file, then verifies the loader wires tensors correctly
/// and the forward pass produces finite vocab-sized logits.
/// </summary>
public sealed class TransformerSafetensorsLoadTests : IDisposable
{
    private readonly string _scratch;

    public TransformerSafetensorsLoadTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-tsl-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// Builds a minimal 2-layer Llama-shaped safetensors fixture with all
    /// required HF tensor names, F32 dtype, and small random-normal-ish
    /// values (±0.05) so forward-pass activations stay in a finite range.
    /// The builder's ramp default <c>startValue + i</c> grows to ~8000 for
    /// a 128×64 gate_proj and would blow up activations; here we supply
    /// deterministic PRNG-derived values explicitly.
    /// </summary>
    private string BuildLlamaFixture(bool tieEmbeddings, int numLayers = 2)
    {
        const int hidden = 64;
        const int numHeads = 4;
        const int headDim = 16;
        const int intermediate = 128;
        const int vocab = 32;

        // Deterministic seed per test so fixtures round-trip stably.
        var rng = new Random(42);

        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, scale: 0.05f));
        b.AddFloat32("model.norm.weight", [hidden], Ones(hidden));

        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.self_attn.q_proj.weight",
                [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.k_proj.weight",
                [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.v_proj.weight",
                [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.o_proj.weight",
                [hidden, numHeads * headDim], RandomVec(rng, hidden * numHeads * headDim, 0.05f));
            b.AddFloat32($"{p}.mlp.gate_proj.weight",
                [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
            b.AddFloat32($"{p}.mlp.up_proj.weight",
                [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
            b.AddFloat32($"{p}.mlp.down_proj.weight",
                [hidden, intermediate], RandomVec(rng, hidden * intermediate, 0.05f));
        }
        if (!tieEmbeddings)
            b.AddFloat32("lm_head.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));

        string path = Path.Combine(_scratch, tieEmbeddings ? "tied.safetensors" : "untied.safetensors");
        b.WriteTo(path);
        return path;
    }

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

    private static ModelConfig BuildLlamaConfig(bool tieEmbeddings)
        => new ModelConfig
        {
            Architecture = Architecture.Llama,
            VocabSize = 32,
            HiddenSize = 64,
            IntermediateSize = 128,
            NumLayers = 2,
            NumAttentionHeads = 4,
            NumKvHeads = 4,
            HeadDim = 16,
            MaxSequenceLength = 128,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = tieEmbeddings,
            RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: 16, Type: RoPEType.Norm),
        };

    [Fact]
    public void UntiedEmbeddings_ForwardProducesFiniteVocabLogits()
    {
        string path = BuildLlamaFixture(tieEmbeddings: false);
        using var file = SafetensorsFile.Open(path);
        var config = BuildLlamaConfig(tieEmbeddings: false);

        using var model = TransformerModel.LoadFromSafetensors(file, config);
        using var logits = model.Forward(
            tokenIds: [0, 1, 2],
            positions: [0, 1, 2],
            deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(3, logits.Shape[0]);
        Assert.Equal(config.VocabSize, logits.Shape[1]);
        AssertAllFinite(logits);
    }

    [Fact]
    public void TiedEmbeddings_LoadsWithoutLmHeadTensor()
    {
        string path = BuildLlamaFixture(tieEmbeddings: true);
        using var file = SafetensorsFile.Open(path);
        // Sanity check on the fixture itself
        Assert.False(file.TensorsByName.ContainsKey("lm_head.weight"),
            "Tied fixture must not contain lm_head.weight");

        var config = BuildLlamaConfig(tieEmbeddings: true);
        using var model = TransformerModel.LoadFromSafetensors(file, config);

        // Forward pass succeeds using the aliased embedding matrix as the LM head.
        using var logits = model.Forward(
            tokenIds: [0, 1],
            positions: [0, 1],
            deviceId: -1);
        Assert.Equal(config.VocabSize, logits.Shape[1]);
        AssertAllFinite(logits);
    }

    [Fact]
    public void MissingProjection_ThrowsWithTensorName()
    {
        // Build a fixture that's missing q_proj on layer 0.
        const int hidden = 64, numHeads = 4, headDim = 16, intermediate = 128, vocab = 32;
        var rng = new Random(1);
        var b = new SafetensorsFixtureBuilder()
            .AddFloat32("model.embed_tokens.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f))
            .AddFloat32("model.norm.weight", [hidden], Ones(hidden))
            .AddFloat32("lm_head.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f))
            .AddFloat32("model.layers.0.input_layernorm.weight", [hidden], Ones(hidden))
            .AddFloat32("model.layers.0.post_attention_layernorm.weight", [hidden], Ones(hidden))
            // missing: self_attn.q_proj.weight
            .AddFloat32("model.layers.0.self_attn.k_proj.weight", [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f))
            .AddFloat32("model.layers.0.self_attn.v_proj.weight", [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f))
            .AddFloat32("model.layers.0.self_attn.o_proj.weight", [hidden, numHeads * headDim], RandomVec(rng, hidden * numHeads * headDim, 0.05f))
            .AddFloat32("model.layers.0.mlp.gate_proj.weight", [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f))
            .AddFloat32("model.layers.0.mlp.up_proj.weight", [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f))
            .AddFloat32("model.layers.0.mlp.down_proj.weight", [hidden, intermediate], RandomVec(rng, hidden * intermediate, 0.05f));

        string path = Path.Combine(_scratch, "missing.safetensors");
        b.WriteTo(path);

        using var file = SafetensorsFile.Open(path);
        var config = BuildLlamaConfig(tieEmbeddings: false) with { NumLayers = 1 };

        var ex = Assert.Throws<InvalidDataException>(() =>
        {
            var m = TransformerModel.LoadFromSafetensors(file, config);
            m.Dispose();
        });
        Assert.Contains("self_attn.q_proj.weight", ex.Message);
    }

    [Fact]
    public void Bf16Dtype_UpcastsAndLoads()
    {
        // Build a fixture where gate_proj is bf16 and everything else is F32.
        const int hidden = 64, numHeads = 4, headDim = 16, intermediate = 128, vocab = 32;
        int numLayers = 1;
        var rng = new Random(2);
        var b = new SafetensorsFixtureBuilder()
            .AddFloat32("model.embed_tokens.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f))
            .AddFloat32("model.norm.weight", [hidden], Ones(hidden))
            .AddFloat32("lm_head.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.self_attn.q_proj.weight", [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.k_proj.weight", [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.v_proj.weight", [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.o_proj.weight", [hidden, numHeads * headDim], RandomVec(rng, hidden * numHeads * headDim, 0.05f));
            // BF16 gate: 2 bytes per element; value = 0.03125f has bf16 bit pattern 0x3D00.
            // Keep values small so bf16 → f32 upcast lands in a plausible weight range.
            int gateElements = intermediate * hidden;
            var bf16Bytes = new byte[gateElements * 2];
            ushort bf16Value = 0x3D00; // bf16 representation of 0.03125
            for (int j = 0; j < gateElements; j++)
            {
                bf16Bytes[j * 2] = (byte)(bf16Value & 0xFF);
                bf16Bytes[j * 2 + 1] = (byte)(bf16Value >> 8);
            }
            b.AddRaw($"{p}.mlp.gate_proj.weight", "BF16", [intermediate, hidden], bf16Bytes);
            b.AddFloat32($"{p}.mlp.up_proj.weight", [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
            b.AddFloat32($"{p}.mlp.down_proj.weight", [hidden, intermediate], RandomVec(rng, hidden * intermediate, 0.05f));
        }

        string path = Path.Combine(_scratch, "bf16.safetensors");
        b.WriteTo(path);

        using var file = SafetensorsFile.Open(path);
        var config = BuildLlamaConfig(tieEmbeddings: false) with { NumLayers = numLayers };
        using var model = TransformerModel.LoadFromSafetensors(file, config);

        using var logits = model.Forward([0], [0], deviceId: -1);
        AssertAllFinite(logits);
    }

    private static unsafe void AssertAllFinite(ITensor logits)
    {
        int n = 1;
        for (int i = 0; i < logits.Shape.Rank; i++)
            n *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        for (int i = 0; i < span.Length; i++)
        {
            float v = span[i];
            Assert.True(float.IsFinite(v), $"Logit index {i} is non-finite ({v}).");
        }
    }
}
