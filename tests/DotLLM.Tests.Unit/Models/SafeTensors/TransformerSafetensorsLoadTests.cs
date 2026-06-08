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

    /// <summary>
    /// Synthetic Mixtral-convention fixture: 2 layers, 4 experts, top-2 gating,
    /// GQA (2 KV heads), F32. Exercises the Mixtral tensor-name resolution path
    /// in <see cref="TransformerWeightsSafetensorsLoader"/> and confirms the
    /// forward pass dispatches through <see cref="DotLLM.Cpu.Kernels.MoeSwiGluMlp"/>.
    /// </summary>
    [Fact]
    public void MixtralMoe_SyntheticFixture_ForwardProducesFiniteVocabLogits()
    {
        const int hidden = 16;
        const int numHeads = 4;
        const int numKvHeads = 2;
        const int headDim = 4;
        const int intermediate = 32;
        const int vocab = 32;
        const int numLayers = 2;
        const int numExperts = 4;
        const int topK = 2;

        var rng = new Random(1337);

        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));
        b.AddFloat32("model.norm.weight", [hidden], Ones(hidden));
        b.AddFloat32("lm_head.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));

        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.self_attn.q_proj.weight",
                [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.k_proj.weight",
                [numKvHeads * headDim, hidden], RandomVec(rng, numKvHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.v_proj.weight",
                [numKvHeads * headDim, hidden], RandomVec(rng, numKvHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.o_proj.weight",
                [hidden, numHeads * headDim], RandomVec(rng, hidden * numHeads * headDim, 0.05f));

            // Mixtral MoE FFN: router gate + (w1, w2, w3) per expert.
            b.AddFloat32($"{p}.block_sparse_moe.gate.weight",
                [numExperts, hidden], RandomVec(rng, numExperts * hidden, 0.05f));
            for (int e = 0; e < numExperts; e++)
            {
                b.AddFloat32($"{p}.block_sparse_moe.experts.{e}.w1.weight",
                    [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
                b.AddFloat32($"{p}.block_sparse_moe.experts.{e}.w2.weight",
                    [hidden, intermediate], RandomVec(rng, hidden * intermediate, 0.05f));
                b.AddFloat32($"{p}.block_sparse_moe.experts.{e}.w3.weight",
                    [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
            }
        }

        string path = Path.Combine(_scratch, "mixtral.safetensors");
        b.WriteTo(path);

        using var file = SafetensorsFile.Open(path);
        var config = new ModelConfig
        {
            Architecture = Architecture.Mixtral,
            VocabSize = vocab,
            HiddenSize = hidden,
            IntermediateSize = intermediate,
            NumLayers = numLayers,
            NumAttentionHeads = numHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = 128,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            RoPEConfig = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: headDim, Type: RoPEType.Norm),
            Moe = new MoeConfig
            {
                NumExperts = numExperts,
                NumExpertsPerTok = topK,
                MoeIntermediateSize = intermediate,
            },
        };

        using var model = TransformerModel.LoadFromSafetensors(file, config);
        using var logits = model.Forward(
            tokenIds: [0, 1, 2],
            positions: [0, 1, 2],
            deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(3, logits.Shape[0]);
        Assert.Equal(vocab, logits.Shape[1]);
        AssertAllFinite(logits);
    }

    /// <summary>
    /// Synthetic Qwen-MoE fixture (Qwen3-MoE convention, no shared expert,
    /// no interleaved dense layers) — 2 layers of routed MoE with the HF
    /// Llama-style expert tensor names (<c>mlp.experts.{e}.{gate,up,down}_proj</c>)
    /// and a router gate at <c>mlp.gate</c>. Proves the Qwen-MoE tensor-name
    /// loader path goes through <see cref="DotLLM.Cpu.Kernels.MoeSwiGluMlp"/>
    /// and yields finite logits.
    /// </summary>
    [Fact]
    public void QwenMoe_SyntheticFixture_ForwardProducesFiniteVocabLogits()
    {
        const int hidden = 16;
        const int numHeads = 4;
        const int numKvHeads = 2;
        const int headDim = 4;
        const int intermediate = 32;
        const int vocab = 32;
        const int numLayers = 2;
        const int numExperts = 4;
        const int topK = 2;

        var rng = new Random(2026);

        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));
        b.AddFloat32("model.norm.weight", [hidden], Ones(hidden));
        b.AddFloat32("lm_head.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));

        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.self_attn.q_proj.weight",
                [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.k_proj.weight",
                [numKvHeads * headDim, hidden], RandomVec(rng, numKvHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.v_proj.weight",
                [numKvHeads * headDim, hidden], RandomVec(rng, numKvHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.o_proj.weight",
                [hidden, numHeads * headDim], RandomVec(rng, hidden * numHeads * headDim, 0.05f));

            // Qwen-MoE MoE FFN: mlp.gate + mlp.experts.{e}.{gate,up,down}_proj.
            b.AddFloat32($"{p}.mlp.gate.weight",
                [numExperts, hidden], RandomVec(rng, numExperts * hidden, 0.05f));
            for (int e = 0; e < numExperts; e++)
            {
                b.AddFloat32($"{p}.mlp.experts.{e}.gate_proj.weight",
                    [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
                b.AddFloat32($"{p}.mlp.experts.{e}.down_proj.weight",
                    [hidden, intermediate], RandomVec(rng, hidden * intermediate, 0.05f));
                b.AddFloat32($"{p}.mlp.experts.{e}.up_proj.weight",
                    [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
            }
        }

        string path = Path.Combine(_scratch, "qwen-moe.safetensors");
        b.WriteTo(path);

        using var file = SafetensorsFile.Open(path);
        var config = new ModelConfig
        {
            Architecture = Architecture.QwenMoe,
            VocabSize = vocab,
            HiddenSize = hidden,
            IntermediateSize = intermediate,
            NumLayers = numLayers,
            NumAttentionHeads = numHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = 128,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            RoPEConfig = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: headDim, Type: RoPEType.NeoX),
            Moe = new MoeConfig
            {
                NumExperts = numExperts,
                NumExpertsPerTok = topK,
                MoeIntermediateSize = intermediate,
                NormTopKProb = true,
                DecoderSparseStep = 1,
            },
        };

        using var model = TransformerModel.LoadFromSafetensors(file, config);
        using var logits = model.Forward(
            tokenIds: [0, 1, 2],
            positions: [0, 1, 2],
            deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(3, logits.Shape[0]);
        Assert.Equal(vocab, logits.Shape[1]);
        AssertAllFinite(logits);
    }

    /// <summary>
    /// Qwen1.5-MoE-A2.7B fixture: 1 MoE layer, 4 routed experts top-2, a
    /// shared expert (<c>mlp.shared_expert.*</c>) with a sigmoid gate
    /// (<c>mlp.shared_expert_gate.weight</c>), and <c>norm_topk_prob=false</c>.
    /// Proves the shared-expert + no-renorm path wires up end-to-end.
    /// </summary>
    [Fact]
    public void QwenMoe_SharedExpertFixture_ForwardProducesFiniteVocabLogits()
    {
        const int hidden = 16;
        const int numHeads = 4;
        const int numKvHeads = 2;
        const int headDim = 4;
        const int intermediate = 32;
        const int sharedIntermediate = 24;   // deliberately != intermediate
        const int vocab = 32;
        const int numLayers = 1;
        const int numExperts = 4;
        const int topK = 2;

        var rng = new Random(4711);

        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));
        b.AddFloat32("model.norm.weight", [hidden], Ones(hidden));
        b.AddFloat32("lm_head.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));

        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.self_attn.q_proj.weight",
                [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.k_proj.weight",
                [numKvHeads * headDim, hidden], RandomVec(rng, numKvHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.v_proj.weight",
                [numKvHeads * headDim, hidden], RandomVec(rng, numKvHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.o_proj.weight",
                [hidden, numHeads * headDim], RandomVec(rng, hidden * numHeads * headDim, 0.05f));

            b.AddFloat32($"{p}.mlp.gate.weight",
                [numExperts, hidden], RandomVec(rng, numExperts * hidden, 0.05f));
            for (int e = 0; e < numExperts; e++)
            {
                b.AddFloat32($"{p}.mlp.experts.{e}.gate_proj.weight",
                    [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
                b.AddFloat32($"{p}.mlp.experts.{e}.down_proj.weight",
                    [hidden, intermediate], RandomVec(rng, hidden * intermediate, 0.05f));
                b.AddFloat32($"{p}.mlp.experts.{e}.up_proj.weight",
                    [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
            }
            // Shared expert (dense SwiGLU) + sigmoid gate.
            b.AddFloat32($"{p}.mlp.shared_expert.gate_proj.weight",
                [sharedIntermediate, hidden], RandomVec(rng, sharedIntermediate * hidden, 0.05f));
            b.AddFloat32($"{p}.mlp.shared_expert.up_proj.weight",
                [sharedIntermediate, hidden], RandomVec(rng, sharedIntermediate * hidden, 0.05f));
            b.AddFloat32($"{p}.mlp.shared_expert.down_proj.weight",
                [hidden, sharedIntermediate], RandomVec(rng, hidden * sharedIntermediate, 0.05f));
            b.AddFloat32($"{p}.mlp.shared_expert_gate.weight",
                [1, hidden], RandomVec(rng, hidden, 0.1f));
        }

        string path = Path.Combine(_scratch, "qwen-moe-shared.safetensors");
        b.WriteTo(path);

        using var file = SafetensorsFile.Open(path);
        var config = new ModelConfig
        {
            Architecture = Architecture.QwenMoe,
            VocabSize = vocab,
            HiddenSize = hidden,
            IntermediateSize = intermediate,
            NumLayers = numLayers,
            NumAttentionHeads = numHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = 128,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            RoPEConfig = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: headDim, Type: RoPEType.NeoX),
            Moe = new MoeConfig
            {
                NumExperts = numExperts,
                NumExpertsPerTok = topK,
                MoeIntermediateSize = intermediate,
                NormTopKProb = false, // Qwen1.5-MoE convention
                SharedExpertIntermediateSize = sharedIntermediate,
                HasSharedExpertGate = true,
                DecoderSparseStep = 1,
            },
        };

        using var model = TransformerModel.LoadFromSafetensors(file, config);
        using var logits = model.Forward(
            tokenIds: [0, 1, 2],
            positions: [0, 1, 2],
            deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(3, logits.Shape[0]);
        Assert.Equal(vocab, logits.Shape[1]);
        AssertAllFinite(logits);
    }

    /// <summary>
    /// DeepSeek-V2/V3 convention fixture: Qwen-MoE tensor naming for routed
    /// experts (<c>mlp.experts.{e}.*</c>) plus the PLURAL
    /// <c>mlp.shared_experts.{k}.*</c> shared-expert naming with
    /// <c>n_shared_experts = 2</c> and <b>no</b> sigmoid gate. Exercises the
    /// multi-shared-expert loader path; the forward pass is driven through
    /// <see cref="Architecture.QwenMoe"/> as a stand-in (DeepSeek-V2/V3 uses
    /// MLA attention which is not yet wired into TransformerModel — tracked
    /// separately). This proves the MoE weight loader correctly resolves the
    /// plural tensor names into the <c>MoeLayerWeights</c> arrays.
    /// </summary>
    [Fact]
    public void DeepSeekStyleMoE_PluralSharedExperts_LoadsAndProducesFiniteLogits()
    {
        const int hidden = 16;
        const int numHeads = 4;
        const int numKvHeads = 2;
        const int headDim = 4;
        const int intermediate = 32;
        const int sharedIntermediate = 20; // == moe_intermediate_size per shared
        const int vocab = 32;
        const int numLayers = 1;
        const int numExperts = 4;
        const int topK = 2;
        const int numSharedExperts = 2;

        var rng = new Random(20260419);

        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));
        b.AddFloat32("model.norm.weight", [hidden], Ones(hidden));
        b.AddFloat32("lm_head.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));

        for (int i = 0; i < numLayers; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [hidden], Ones(hidden));
            b.AddFloat32($"{p}.self_attn.q_proj.weight",
                [numHeads * headDim, hidden], RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.k_proj.weight",
                [numKvHeads * headDim, hidden], RandomVec(rng, numKvHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.v_proj.weight",
                [numKvHeads * headDim, hidden], RandomVec(rng, numKvHeads * headDim * hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.o_proj.weight",
                [hidden, numHeads * headDim], RandomVec(rng, hidden * numHeads * headDim, 0.05f));

            b.AddFloat32($"{p}.mlp.gate.weight",
                [numExperts, hidden], RandomVec(rng, numExperts * hidden, 0.05f));
            for (int e = 0; e < numExperts; e++)
            {
                b.AddFloat32($"{p}.mlp.experts.{e}.gate_proj.weight",
                    [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
                b.AddFloat32($"{p}.mlp.experts.{e}.down_proj.weight",
                    [hidden, intermediate], RandomVec(rng, hidden * intermediate, 0.05f));
                b.AddFloat32($"{p}.mlp.experts.{e}.up_proj.weight",
                    [intermediate, hidden], RandomVec(rng, intermediate * hidden, 0.05f));
            }
            // Plural shared experts (DeepSeek naming): mlp.shared_experts.{k}.*
            for (int k = 0; k < numSharedExperts; k++)
            {
                b.AddFloat32($"{p}.mlp.shared_experts.{k}.gate_proj.weight",
                    [sharedIntermediate, hidden], RandomVec(rng, sharedIntermediate * hidden, 0.05f));
                b.AddFloat32($"{p}.mlp.shared_experts.{k}.up_proj.weight",
                    [sharedIntermediate, hidden], RandomVec(rng, sharedIntermediate * hidden, 0.05f));
                b.AddFloat32($"{p}.mlp.shared_experts.{k}.down_proj.weight",
                    [hidden, sharedIntermediate], RandomVec(rng, hidden * sharedIntermediate, 0.05f));
            }
        }

        string path = Path.Combine(_scratch, "deepseek-style-plural.safetensors");
        b.WriteTo(path);

        using var file = SafetensorsFile.Open(path);
        // Drive through QwenMoe arch so the existing TransformerModel forward
        // path handles the MoE plumbing end-to-end (DeepSeek's MLA attention
        // is out of scope for this test — we're verifying the multi-shared
        // LOADER contract, not the DeepSeek attention kernel).
        var config = new ModelConfig
        {
            Architecture = Architecture.QwenMoe,
            VocabSize = vocab,
            HiddenSize = hidden,
            IntermediateSize = intermediate,
            NumLayers = numLayers,
            NumAttentionHeads = numHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = 128,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            RoPEConfig = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: headDim, Type: RoPEType.NeoX),
            Moe = new MoeConfig
            {
                NumExperts = numExperts,
                NumExpertsPerTok = topK,
                MoeIntermediateSize = intermediate,
                NormTopKProb = false,
                SharedExpertIntermediateSize = sharedIntermediate,
                NumSharedExperts = numSharedExperts,
                HasSharedExpertGate = false, // DeepSeek: no gate
                DecoderSparseStep = 1,
            },
        };

        using var model = TransformerModel.LoadFromSafetensors(file, config);
        using var logits = model.Forward(
            tokenIds: [0, 1, 2],
            positions: [0, 1, 2],
            deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(3, logits.Shape[0]);
        Assert.Equal(vocab, logits.Shape[1]);
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
