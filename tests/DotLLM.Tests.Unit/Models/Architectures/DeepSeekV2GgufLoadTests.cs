using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// End-to-end CPU GGUF DeepSeek-V2 load integration test. Builds a synthetic
/// minimal-shaped DeepSeek-V2-Lite-style GGUF in memory (single dense layer,
/// all tensors F32 to keep the fixture small and avoid Q4_K block-alignment
/// constraints on hidden=16), loads it via
/// <see cref="TransformerWeights.LoadFromGguf"/>, and asserts the MLA layer
/// projections were dequantised into the loader's owned-allocations list.
/// </summary>
/// <remarks>
/// MoE-layer loading is intentionally deferred to the parallel
/// DeepSeek-GGUF A-2 PR (3D-stacked ffn_*_exps tensor loader). The fixture
/// here only exercises the dense path of <c>LoadMlaLayer</c>.
/// </remarks>
public sealed class DeepSeekV2GgufLoadTests
{
    private const int HiddenSize = 16;
    private const int NumLayers = 1;       // dense-only — MoE loader ships in A-2
    private const int NumHeads = 2;
    private const int VocabSize = 8;
    private const int QkNope = 4;
    private const int QkRope = 4;          // RoPE pairs → must be even
    private const int VHead = 4;
    private const int KvLoraRank = 8;
    private const int IntermediateSize = 24;

    [Fact]
    public void LoadFromGguf_DeepSeekV2Lite_MonolithicQ_Loads()
    {
        string ggufPath = WriteFixture(qLoraRank: 0, seed: 42);
        try
        {
            using var gguf = GgufFile.Open(ggufPath);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

            // Sanity-check the extracted config
            Assert.Equal(Architecture.DeepSeekV2, config.Architecture);
            Assert.Equal(AttentionType.MLA, config.AttentionType);
            Assert.NotNull(config.MlaConfig);
            Assert.Equal(0, config.MlaConfig.QLoraRank);

            // Load the weights — dense FFN path on the (only) leading-dense layer.
            using var weights = TransformerWeights.LoadFromGguf(gguf, config);
            Assert.Single(weights.Layers);

            ref readonly var layer0 = ref weights.Layers[0];
            Assert.NotNull(layer0.Mla);
            Assert.Equal(0, layer0.Mla.QLoraRank);
            Assert.NotEqual((nint)0, layer0.Mla.QProj);                // monolithic Q populated
            Assert.Equal((nint)0, layer0.Mla.QAProj);                  // LoRA-Q absent
            Assert.NotEqual((nint)0, layer0.GateWeight);               // dense FFN
        }
        finally
        {
            File.Delete(ggufPath);
        }
    }

    [Fact]
    public void LoadFromGguf_DeepSeekV2_LoraQ_Loads()
    {
        string ggufPath = WriteFixture(qLoraRank: 8, seed: 7);
        try
        {
            using var gguf = GgufFile.Open(ggufPath);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

            Assert.Equal(8, config.MlaConfig!.QLoraRank);

            using var weights = TransformerWeights.LoadFromGguf(gguf, config);
            ref readonly var layer0 = ref weights.Layers[0];
            Assert.NotNull(layer0.Mla);
            Assert.Equal(8, layer0.Mla.QLoraRank);
            Assert.NotEqual((nint)0, layer0.Mla.QAProj);
            Assert.Equal((nint)0, layer0.Mla.QProj);
        }
        finally
        {
            File.Delete(ggufPath);
        }
    }

    /// <summary>
    /// Real-checkpoint config-extraction smoke test against a cached
    /// <c>DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf</c> (~10.4 GB,
    /// 16B-class with MLA + 64 routed + 2 shared experts). Skipped when
    /// the file isn't downloaded. This validates that the metadata
    /// extractor handles the real key naming + type encoding without
    /// blowing the host RAM (only metadata is read; tensor data isn't
    /// touched). MoE config assertions ship with the parallel
    /// DeepSeek-GGUF A-2 PR.
    /// </summary>
    [SkippableFact]
    public void RealGguf_ConfigExtractor_ParsesDeepSeekV2Lite()
    {
        string path = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf");
        Skip.If(!File.Exists(path), $"Real DeepSeek-V2-Lite GGUF not cached at {path}");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.DeepSeekV2, config.Architecture);
        Assert.Equal(AttentionType.MLA, config.AttentionType);

        // V2-Lite expected hyperparameters (per HF config.json on
        // deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct):
        //   hidden_size = 2048, num_hidden_layers = 27,
        //   num_attention_heads = 16, num_key_value_heads = 16,
        //   intermediate_size = 10944, vocab_size ≈ 102400,
        //   q_lora_rank = 0 (monolithic Q on V2-Lite),
        //   kv_lora_rank = 512, qk_nope_head_dim = 128,
        //   qk_rope_head_dim = 64, v_head_dim = 128.
        Assert.Equal(2048, config.HiddenSize);
        Assert.Equal(27, config.NumLayers);
        Assert.Equal(16, config.NumAttentionHeads);

        Assert.NotNull(config.MlaConfig);
        Assert.Equal(0, config.MlaConfig.QLoraRank);   // V2-Lite is monolithic-Q
        Assert.Equal(512, config.MlaConfig.KvLoraRank);
        Assert.Equal(128, config.MlaConfig.QkNopeHeadDim);
        Assert.Equal(64, config.MlaConfig.QkRopeHeadDim);
        Assert.Equal(128, config.MlaConfig.VHeadDim);
        Assert.Equal(192, config.HeadDim);             // patched to qk_nope + qk_rope
    }

    /// <summary>
    /// Writes a synthetic minimal DeepSeek-V2-Lite-shaped GGUF in F32 (no
    /// quantization) to a temp file. F32 keeps the byte layout trivial:
    /// <c>elementCount * 4</c> bytes per tensor, no per-block alignment
    /// concerns. Only the dense leading layer is materialised — the MoE
    /// 3D-stacked expert tensor loader (and the matching MoeConfig
    /// extraction) ships with the parallel DeepSeek-GGUF A-2 PR.
    /// </summary>
    private static string WriteFixture(int qLoraRank, int seed)
    {
        var b = new GgufTestData(version: 3);

        int qkHead = QkNope + QkRope;
        int qTotal = NumHeads * qkHead;
        int kvAOut = KvLoraRank + QkRope;
        int kvBOut = NumHeads * (QkNope + VHead);
        int oInput = NumHeads * VHead;

        // ── Metadata ──────────────────────────────────────────────────
        b.AddString("general.architecture", "deepseek2");
        b.AddUInt32("deepseek2.embedding_length", (uint)HiddenSize);
        b.AddUInt32("deepseek2.block_count", (uint)NumLayers);
        b.AddUInt32("deepseek2.feed_forward_length", (uint)IntermediateSize);
        b.AddUInt32("deepseek2.attention.head_count", (uint)NumHeads);
        b.AddUInt32("deepseek2.attention.head_count_kv", (uint)NumHeads);
        b.AddUInt32("deepseek2.context_length", 16);
        b.AddFloat32("deepseek2.attention.layer_norm_rms_epsilon", 1e-6f);
        b.AddUInt32("deepseek2.vocab_size", (uint)VocabSize);
        b.AddFloat32("deepseek2.rope.freq_base", 10000.0f);
        b.AddUInt32("deepseek2.rope.dimension_count", (uint)QkRope);

        // MLA
        b.AddUInt32("deepseek2.attention.q_lora_rank", (uint)qLoraRank);
        b.AddUInt32("deepseek2.attention.kv_lora_rank", (uint)KvLoraRank);
        // attention.key_length = TOTAL per-head qk dim (qk_nope + qk_rope), per
        // llama.cpp's gguf_writer convention for MLA models. The extractor
        // derives qk_nope = key_length - rope.dimension_count.
        b.AddUInt32("deepseek2.attention.key_length", (uint)(QkNope + QkRope));
        b.AddUInt32("deepseek2.attention.value_length", (uint)VHead);

        // ── Tensors (F32) ─────────────────────────────────────────────
        // Globals
        AddF32Tensor(b, "token_embd.weight", [HiddenSize, VocabSize], seed + 100);  // GGUF: [K, M]
        AddF32Tensor(b, "output_norm.weight", [HiddenSize], seed + 101, center: 1.0f, jitter: 0.05f);
        AddF32Tensor(b, "output.weight", [HiddenSize, VocabSize], seed + 102);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 1000 * (i + 1);
            string p = $"blk.{i}";

            AddF32Tensor(b, $"{p}.attn_norm.weight", [HiddenSize], s + 0, center: 1.0f, jitter: 0.05f);
            AddF32Tensor(b, $"{p}.ffn_norm.weight", [HiddenSize], s + 1, center: 1.0f, jitter: 0.05f);

            // MLA attention
            if (qLoraRank > 0)
            {
                AddF32Tensor(b, $"{p}.attn_q_a.weight", [HiddenSize, qLoraRank], s + 2);
                AddF32Tensor(b, $"{p}.attn_q_a_norm.weight", [qLoraRank], s + 3, center: 1.0f, jitter: 0.05f);
                AddF32Tensor(b, $"{p}.attn_q_b.weight", [qLoraRank, qTotal], s + 4);
            }
            else
            {
                AddF32Tensor(b, $"{p}.attn_q.weight", [HiddenSize, qTotal], s + 2);
            }
            AddF32Tensor(b, $"{p}.attn_kv_a_mqa.weight", [HiddenSize, kvAOut], s + 5);
            AddF32Tensor(b, $"{p}.attn_kv_a_norm.weight", [KvLoraRank], s + 6, center: 1.0f, jitter: 0.05f);
            AddF32Tensor(b, $"{p}.attn_kv_b.weight", [KvLoraRank, kvBOut], s + 7);
            AddF32Tensor(b, $"{p}.attn_output.weight", [oInput, HiddenSize], s + 8);

            // Dense FFN (only the leading layer; MoE 3D-stacked path lives in A-2).
            AddF32Tensor(b, $"{p}.ffn_gate.weight", [HiddenSize, IntermediateSize], s + 9);
            AddF32Tensor(b, $"{p}.ffn_up.weight", [HiddenSize, IntermediateSize], s + 10);
            AddF32Tensor(b, $"{p}.ffn_down.weight", [IntermediateSize, HiddenSize], s + 11);
        }

        return b.WriteToTempFile();
    }

    /// <summary>
    /// Adds an F32 tensor with a deterministic cos-based fill. <paramref name="center"/>
    /// + <paramref name="jitter"/> form the near-unity range used by norm weights.
    /// </summary>
    private static void AddF32Tensor(GgufTestData b, string name, int[] shape, int seed,
                                     float amplitude = 0.1f,
                                     float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        foreach (int d in shape) n *= d;
        byte[] bytes = new byte[n * sizeof(float)];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            float cos = MathF.Cos(phi);
            float v = jitter > 0f ? center + jitter * cos : amplitude * cos;
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(
                bytes.AsSpan((int)(i * sizeof(float)), sizeof(float)), v);
        }
        // GGUF tensor type IDs: F32=0, F16=1, ...
        b.AddTensor(name, shape, quantType: 0, bytes);
    }
}
