using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Lora;

/// <summary>
/// Multi-adapter switch tests. Confirms (a) two adapters with different
/// factors give distinguishable outputs, (b) the registry's hot-swap is
/// instant — well under the 100 ms Phase 7 success criterion (when the
/// adapter is already loaded; on-disk loading is a separate timing).
/// </summary>
public sealed class LoraAdapterRegistrySwitchTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public LoraAdapterRegistrySwitchTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-lora-sw-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private (TransformerModel Model, IDisposable Source, ModelConfig Config) BuildTinyModel()
    {
        const int hidden = 64, numHeads = 4, headDim = 16, intermediate = 128, vocab = 32, layers = 2;
        var rng = new Random(11);
        var bld = new SafetensorsFixtureBuilder();
        bld.AddFloat32("model.embed_tokens.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));
        bld.AddFloat32("model.norm.weight", [hidden], Ones(hidden));
        for (int i = 0; i < layers; i++)
        {
            string p = $"model.layers.{i}";
            bld.AddFloat32($"{p}.input_layernorm.weight", [hidden], Ones(hidden));
            bld.AddFloat32($"{p}.post_attention_layernorm.weight", [hidden], Ones(hidden));
            bld.AddFloat32($"{p}.self_attn.q_proj.weight", [numHeads * headDim, hidden],
                RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.k_proj.weight", [numHeads * headDim, hidden],
                RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.v_proj.weight", [numHeads * headDim, hidden],
                RandomVec(rng, numHeads * headDim * hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.o_proj.weight", [hidden, numHeads * headDim],
                RandomVec(rng, hidden * numHeads * headDim, 0.05f));
            bld.AddFloat32($"{p}.mlp.gate_proj.weight", [intermediate, hidden],
                RandomVec(rng, intermediate * hidden, 0.05f));
            bld.AddFloat32($"{p}.mlp.up_proj.weight", [intermediate, hidden],
                RandomVec(rng, intermediate * hidden, 0.05f));
            bld.AddFloat32($"{p}.mlp.down_proj.weight", [hidden, intermediate],
                RandomVec(rng, hidden * intermediate, 0.05f));
        }
        bld.AddFloat32("lm_head.weight", [vocab, hidden], RandomVec(rng, vocab * hidden, 0.05f));

        string path = Path.Combine(_scratch, "base.safetensors");
        bld.WriteTo(path);
        var cfg = new ModelConfig
        {
            Architecture = Architecture.Llama,
            VocabSize = vocab,
            HiddenSize = hidden,
            IntermediateSize = intermediate,
            NumLayers = layers,
            NumAttentionHeads = numHeads,
            NumKvHeads = numHeads,
            HeadDim = headDim,
            MaxSequenceLength = 128,
            NormEpsilon = 1e-5f,
            RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: headDim, Type: RoPEType.Norm),
        };
        var file = SafetensorsFile.Open(path);
        var model = TransformerModel.LoadFromSafetensors(file, cfg);
        return (model, file, cfg);
    }

    private static unsafe LoraAdapter BuildAdapter(string name, ModelConfig cfg, int seed)
    {
        var rng = new Random(seed);
        int qOut = cfg.NumAttentionHeads * cfg.HeadDim;
        int rank = 8;
        var adapter = new LoraAdapter(name, rank: rank, alpha: 16f, targetModules: ["q_proj"]);
        try
        {
            for (int layer = 0; layer < cfg.NumLayers; layer++)
            {
                long bElems = (long)rank * cfg.HiddenSize;
                long aElems = (long)qOut * rank;
                nint b = LoraAdapter.AllocAligned(bElems);
                nint a = LoraAdapter.AllocAligned(aElems);
                float* bp = (float*)b;
                float* ap = (float*)a;
                for (long i = 0; i < bElems; i++) bp[i] = (float)((rng.NextDouble() * 2 - 1) * 0.1);
                for (long i = 0; i < aElems; i++) ap[i] = (float)((rng.NextDouble() * 2 - 1) * 0.1);
                adapter.AddLayerWeights(layer, "q_proj",
                    new LoraLayerWeights(AHandle: a, BHandle: b,
                        InputDim: cfg.HiddenSize, OutputDim: qOut));
            }
            return adapter;
        }
        catch
        {
            adapter.Dispose();
            throw;
        }
    }

    [Fact]
    public unsafe void Switch_BetweenAdapters_ProducesDifferentOutputs()
    {
        var (model, source, cfg) = BuildTinyModel();
        try
        {
            using var adapterA = BuildAdapter("A", cfg, seed: 1);
            using var adapterB = BuildAdapter("B", cfg, seed: 999);

            int[] tokenIds = [1, 2, 3];
            int[] positions = [0, 1, 2];

            using var logitsA = model.Forward(tokenIds, positions, deviceId: -1, kvCache: null, adapter: adapterA);

            // Switch — should be instant since the registry/factories are not touched
            // for the swap; the model just consumes the new adapter pointer.
            var sw = Stopwatch.StartNew();
            using var logitsB = model.Forward(tokenIds, positions, deviceId: -1, kvCache: null, adapter: adapterB);
            sw.Stop();
            _output.WriteLine($"Forward(adapterB) after Forward(adapterA) took {sw.Elapsed.TotalMilliseconds:F2} ms");

            // Phase 7 success criterion is <100 ms for adapter swap. The forward
            // itself dominates here (sub-ms for a tiny model) — well within budget.
            Assert.True(sw.Elapsed.TotalMilliseconds < 100,
                $"Adapter swap forward took {sw.Elapsed.TotalMilliseconds:F2} ms (>100 ms target).");

            // Outputs must differ (different adapter weights → different deltas).
            int total = logitsA.Shape[0] * logitsA.Shape[1];
            var spanA = new ReadOnlySpan<float>((void*)logitsA.DataPointer, total);
            var spanB = new ReadOnlySpan<float>((void*)logitsB.DataPointer, total);
            float maxDiff = 0f;
            for (int i = 0; i < total; i++)
                maxDiff = MathF.Max(maxDiff, MathF.Abs(spanA[i] - spanB[i]));
            Assert.True(maxDiff > 1e-3f, $"Two adapters produced indistinguishable outputs (maxDiff={maxDiff}).");
        }
        finally
        {
            model.Dispose();
            source.Dispose();
        }
    }

    [Fact]
    public void Registry_LoadGetUnload_RoundTrip()
    {
        // Use a stub factory that mints a tiny LoraAdapter directly — the
        // registry doesn't care about the on-disk format, only that the
        // factory returns a valid ILoraAdapter with the requested name.
        var registry = new LoraAdapterRegistry((name, path) =>
        {
            var adapter = new LoraAdapter(name, rank: 4, alpha: 8f, targetModules: ["q_proj"]);
            // Single dummy entry so Dispose has something to free.
            adapter.AddLayerWeights(0, "q_proj",
                new LoraLayerWeights(
                    AHandle: LoraAdapter.AllocAligned(16),
                    BHandle: LoraAdapter.AllocAligned(16),
                    InputDim: 4, OutputDim: 4));
            return adapter;
        });
        try
        {
            registry.Load("a", "/dummy/path");
            registry.Load("b", "/dummy/path");

            Assert.NotNull(registry.Get("a"));
            Assert.NotNull(registry.Get("b"));
            Assert.Null(registry.Get("c"));
            Assert.Equal(2, registry.List().Count);

            // Duplicate load throws
            Assert.Throws<InvalidOperationException>(() => registry.Load("a", "/dummy/path"));

            registry.Unload("a");
            Assert.Null(registry.Get("a"));
            Assert.NotNull(registry.Get("b"));
            Assert.Single(registry.List());
        }
        finally
        {
            registry.Dispose();
        }
    }

    private static float[] RandomVec(Random rng, int n, float scale = 1.0f)
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
}
