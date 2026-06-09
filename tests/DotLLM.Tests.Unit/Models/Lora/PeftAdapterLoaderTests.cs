using System.Buffers.Binary;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.Architectures;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lora;

/// <summary>
/// Synthetic-fixture tests for <see cref="PeftAdapterLoader.LoadFromDirectory"/>.
/// Builds a byte-accurate PEFT directory in <see cref="Path.GetTempPath"/>
/// (adapter_config.json + adapter_model.safetensors), invokes the loader,
/// and verifies metadata + per-(layer, proj) weights are wired correctly.
/// </summary>
public sealed class PeftAdapterLoaderTests : IDisposable
{
    private readonly string _scratch;

    public PeftAdapterLoaderTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-peft-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private static ModelConfig BuildBaseConfig() => new()
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
        RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: 16, Type: RoPEType.Norm),
    };

    /// <summary>
    /// Writes a minimal PEFT adapter directory targeting q_proj + v_proj on
    /// each of <paramref name="numLayers"/> layers.
    /// </summary>
    private string BuildPeftFixture(int rank, float alpha, int hidden, int qOut, int kvOut,
                                    int numLayers, string prefix = "base_model.model.",
                                    bool useDefaultSuffix = false,
                                    string taskType = "CAUSAL_LM")
    {
        string dir = Path.Combine(_scratch, $"adapter-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        // adapter_config.json
        var cfgObj = new
        {
            r = rank,
            lora_alpha = alpha,
            target_modules = new[] { "q_proj", "v_proj" },
            lora_dropout = 0.0,
            bias = "none",
            task_type = taskType,
            use_rslora = false,
            use_dora = false,
        };
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"),
            JsonSerializer.Serialize(cfgObj));

        // adapter_model.safetensors
        var b = new SafetensorsFixtureBuilder();
        var rng = new Random(42);
        string suffix = useDefaultSuffix ? ".default" : "";
        for (int i = 0; i < numLayers; i++)
        {
            string p = $"{prefix}model.layers.{i}.self_attn";
            b.AddFloat32($"{p}.q_proj.lora_A{suffix}.weight",
                [rank, hidden], RandomVec(rng, rank * hidden, scale: 0.02f));
            b.AddFloat32($"{p}.q_proj.lora_B{suffix}.weight",
                [qOut, rank], RandomVec(rng, qOut * rank, scale: 0.02f));
            b.AddFloat32($"{p}.v_proj.lora_A{suffix}.weight",
                [rank, hidden], RandomVec(rng, rank * hidden, scale: 0.02f));
            b.AddFloat32($"{p}.v_proj.lora_B{suffix}.weight",
                [kvOut, rank], RandomVec(rng, kvOut * rank, scale: 0.02f));
        }
        b.WriteTo(Path.Combine(dir, "adapter_model.safetensors"));
        return dir;
    }

    private static float[] RandomVec(Random rng, int n, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
            v[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return v;
    }

    [Fact]
    public void LoadFromDirectory_ParsesMetadataAndTensors()
    {
        var cfg = BuildBaseConfig();
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;
        int kvOut = cfg.NumKvHeads * cfg.HeadDim;
        string dir = BuildPeftFixture(rank: 8, alpha: 16f, hidden: cfg.HiddenSize,
            qOut: qOut, kvOut: kvOut, numLayers: cfg.NumLayers);

        using var adapter = PeftAdapterLoader.LoadFromDirectory("test", dir, cfg);

        Assert.Equal("test", adapter.Name);
        Assert.Equal(8, adapter.Rank);
        Assert.Equal(16f, adapter.Alpha);
        Assert.Equal(2, adapter.TargetModules.Count);
        Assert.Contains("q_proj", adapter.TargetModules);
        Assert.Contains("v_proj", adapter.TargetModules);

        // Per-layer/proj entries present
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            Assert.NotNull(adapter.GetLayerWeights(i, "q_proj"));
            Assert.NotNull(adapter.GetLayerWeights(i, "v_proj"));
            Assert.Null(adapter.GetLayerWeights(i, "k_proj")); // not in target_modules
        }

        // IsCompatible should hold for the model whose dims were used.
        Assert.True(adapter.IsCompatible(cfg));
    }

    [Fact]
    public void LoadFromDirectory_HandlesDefaultSuffixVariant()
    {
        var cfg = BuildBaseConfig();
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;
        int kvOut = cfg.NumKvHeads * cfg.HeadDim;
        string dir = BuildPeftFixture(rank: 8, alpha: 16f, hidden: cfg.HiddenSize,
            qOut: qOut, kvOut: kvOut, numLayers: cfg.NumLayers, useDefaultSuffix: true);

        using var adapter = PeftAdapterLoader.LoadFromDirectory("test-default", dir, cfg);

        // Same layer/proj entries should resolve via the .default.weight regex branch.
        Assert.NotNull(adapter.GetLayerWeights(0, "q_proj"));
        Assert.NotNull(adapter.GetLayerWeights(1, "v_proj"));
    }

    [Fact]
    public void LoadFromDirectory_AcceptsAlternatePrefix()
    {
        // Some PEFT exports omit "base_model." or use "base_model.model.".
        var cfg = BuildBaseConfig();
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;
        int kvOut = cfg.NumKvHeads * cfg.HeadDim;
        string dir = BuildPeftFixture(rank: 8, alpha: 16f, hidden: cfg.HiddenSize,
            qOut: qOut, kvOut: kvOut, numLayers: cfg.NumLayers, prefix: "");

        using var adapter = PeftAdapterLoader.LoadFromDirectory("test-noprefix", dir, cfg);

        Assert.NotNull(adapter.GetLayerWeights(0, "q_proj"));
    }

    [Fact]
    public void LoadFromDirectory_RejectsUseRslora()
    {
        var cfg = BuildBaseConfig();
        string dir = Path.Combine(_scratch, "rslora");
        Directory.CreateDirectory(dir);
        var cfgObj = new { r = 8, lora_alpha = 16, target_modules = new[] { "q_proj" }, use_rslora = true, task_type = "CAUSAL_LM" };
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"), JsonSerializer.Serialize(cfgObj));
        // Make a stub safetensors file so we get past the file-existence check.
        new SafetensorsFixtureBuilder()
            .AddFloat32("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                [8, cfg.HiddenSize])
            .AddFloat32("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                [cfg.NumAttentionHeads * cfg.HeadDim, 8])
            .WriteTo(Path.Combine(dir, "adapter_model.safetensors"));

        Assert.Throws<NotSupportedException>(() =>
            PeftAdapterLoader.LoadFromDirectory("rslora", dir, cfg));
    }

    [Fact]
    public void LoadFromDirectory_RejectsUseDora()
    {
        var cfg = BuildBaseConfig();
        string dir = Path.Combine(_scratch, "dora");
        Directory.CreateDirectory(dir);
        var cfgObj = new { r = 8, lora_alpha = 16, target_modules = new[] { "q_proj" }, use_dora = true, task_type = "CAUSAL_LM" };
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"), JsonSerializer.Serialize(cfgObj));
        new SafetensorsFixtureBuilder()
            .AddFloat32("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                [8, cfg.HiddenSize])
            .AddFloat32("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                [cfg.NumAttentionHeads * cfg.HeadDim, 8])
            .WriteTo(Path.Combine(dir, "adapter_model.safetensors"));

        Assert.Throws<NotSupportedException>(() =>
            PeftAdapterLoader.LoadFromDirectory("dora", dir, cfg));
    }

    [Fact]
    public void LoadFromDirectory_RejectsIncompatibleShape()
    {
        var cfg = BuildBaseConfig();
        // Build adapter sized for a totally different model.
        string dir = BuildPeftFixture(rank: 8, alpha: 16f,
            hidden: 999,         // wrong
            qOut: 999,
            kvOut: 999,
            numLayers: cfg.NumLayers);

        Assert.Throws<InvalidDataException>(() =>
            PeftAdapterLoader.LoadFromDirectory("bad", dir, cfg));
    }

    [Fact]
    public void LoadFromDirectory_MissingConfigJsonThrows()
    {
        string dir = Path.Combine(_scratch, "missing-cfg");
        Directory.CreateDirectory(dir);
        Assert.Throws<FileNotFoundException>(() =>
            PeftAdapterLoader.LoadFromDirectory("x", dir, null));
    }

    [Fact]
    public void LoadFromDirectory_RejectsUnknownTaskType()
    {
        var cfg = BuildBaseConfig();
        string dir = Path.Combine(_scratch, "wrong-task");
        Directory.CreateDirectory(dir);
        var cfgObj = new { r = 8, lora_alpha = 16, target_modules = new[] { "q_proj" }, task_type = "TOKEN_CLS" };
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"), JsonSerializer.Serialize(cfgObj));
        new SafetensorsFixtureBuilder()
            .AddFloat32("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                [8, cfg.HiddenSize])
            .AddFloat32("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                [cfg.NumAttentionHeads * cfg.HeadDim, 8])
            .WriteTo(Path.Combine(dir, "adapter_model.safetensors"));
        Assert.Throws<NotSupportedException>(() =>
            PeftAdapterLoader.LoadFromDirectory("x", dir, cfg));
    }
}
