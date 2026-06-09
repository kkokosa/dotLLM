using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.HuggingFace;
using DotLLM.Models;
using DotLLM.Models.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// End-to-end verification that <see cref="ModelLoader.LoadFromSafetensors"/>
/// can open a real HuggingFace tiny-random Qwen-MoE checkpoint, correctly
/// detect Qwen-MoE (<c>Architecture.QwenMoe</c>) via
/// <see cref="HfConfigExtractor"/>, and run a forward pass that produces
/// finite vocab-sized logits. Mirrors <see cref="TinyMixtralSafetensorsLoadTests"/>.
/// </summary>
/// <remarks>
/// <para>
/// Uses <c>yujiepan/qwen3-moe-tiny-random</c> (~20 MB, safetensors,
/// <c>Qwen3MoeForCausalLM</c> class, 2 layers × 8 experts × top-2,
/// <c>decoder_sparse_step=2</c> so layer 0 is dense and layer 1 is MoE).
/// This exercises: Qwen3-MoE tensor-name resolution
/// (<c>mlp.gate</c>, <c>mlp.experts.{j}.{gate,up,down}_proj</c>), BF16→F32
/// upcast for expert weights, mixed dense + MoE layers in one model, and
/// <c>norm_topk_prob=true</c> (Qwen3 default).
/// </para>
/// <para>
/// Skips gracefully on <c>head_dim &lt; 2</c> — same escape hatch as the
/// Mixtral test — or when HF is unreachable. Does NOT probe Qwen1.5-MoE-A2.7B
/// (~14 GB) — the shared-expert + sigmoid-gate path is covered by the
/// synthetic unit-test fixture.
/// </para>
/// <para>
/// Cache location: <c>~/.dotllm/test-cache/&lt;repo&gt;/</c>. 50 MB cap.
/// </para>
/// </remarks>
public sealed class TinyQwenMoeSafetensorsLoadTests
{
    /// <summary>Tiny-random Qwen3-MoE is ~20 MB; cap at 50 MB.</summary>
    private const int MaxAllowedBytes = 50 * 1024 * 1024;

    /// <summary>
    /// Ordered candidate repos. First reachable wins. All three ship a
    /// <c>Qwen3MoeForCausalLM</c> safetensors checkpoint under ~30 MB.
    /// </summary>
    private static readonly (string RepoId, string[] Files)[] Candidates =
    [
        ("yujiepan/qwen3-moe-tiny-random", ["model.safetensors", "config.json"]),
        ("tiny-random/qwen3-moe", ["model.safetensors", "config.json"]),
        ("optimum-internal-testing/tiny-random-qwen3_moe", ["model.safetensors", "config.json"]),
    ];

    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache");

    private readonly ITestOutputHelper _output;

    public TinyQwenMoeSafetensorsLoadTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Proves <see cref="HfConfigExtractor"/> + <see cref="ModelLoader"/>
    /// correctly detect Qwen-MoE and populate MoE config from a real HF
    /// checkpoint's <c>config.json</c>.
    /// </summary>
    [SkippableFact]
    public void RealQwenMoeConfig_IsDetectedAsQwenMoeWithMoe()
    {
        string? modelPath = TryEnsureTinyQwenMoe(out string? skipReason);
        Skip.If(modelPath is null, skipReason ?? "tiny-random Qwen-MoE download unavailable");

        string configPath = Path.Combine(Path.GetDirectoryName(modelPath!)!, "config.json");
        Assert.True(System.IO.File.Exists(configPath), "config.json must be co-located with the model.");

        var cfg = HfConfigExtractor.Extract(System.IO.File.ReadAllText(configPath));
        _output.WriteLine(
            $"Real HF config: arch={cfg.Architecture} hidden={cfg.HiddenSize} layers={cfg.NumLayers} "
          + $"heads={cfg.NumAttentionHeads} kv_heads={cfg.NumKvHeads} head_dim={cfg.HeadDim} "
          + $"intermediate={cfg.IntermediateSize} vocab={cfg.VocabSize}");
        Assert.Equal(Core.Configuration.Architecture.QwenMoe, cfg.Architecture);
        Assert.NotNull(cfg.Moe);
        Assert.True(cfg.Moe!.NumExperts >= 2);
        Assert.True(cfg.Moe.NumExpertsPerTok >= 1);
        Assert.True(cfg.Moe.NumExpertsPerTok <= cfg.Moe.NumExperts);
        Assert.True(cfg.Moe.MoeIntermediateSize > 0);
        Assert.True(cfg.Moe.DecoderSparseStep >= 1);
        _output.WriteLine(
            $"Moe: num_experts={cfg.Moe.NumExperts} top_k={cfg.Moe.NumExpertsPerTok} "
          + $"moe_intermediate={cfg.Moe.MoeIntermediateSize} norm_topk={cfg.Moe.NormTopKProb} "
          + $"sparse_step={cfg.Moe.DecoderSparseStep} shared_expert_intermediate={cfg.Moe.SharedExpertIntermediateSize} "
          + $"has_shared_gate={cfg.Moe.HasSharedExpertGate}");
    }

    /// <summary>
    /// End-to-end load + 3-token forward pass on the real HF checkpoint.
    /// Asserts finite logits, nonzero variance, and matching vocab-size
    /// shape. Skips gracefully on degenerate geometries or HF unavailability.
    /// </summary>
    [SkippableFact]
    public void LoadAndForwardPass_ProducesFiniteVocabLogits()
    {
        string? modelPath = TryEnsureTinyQwenMoe(out string? skipReason);
        Skip.If(modelPath is null, skipReason ?? "tiny-random Qwen-MoE download unavailable");

        _output.WriteLine($"Loaded tiny-random Qwen-MoE from: {modelPath}");

        using var result = LoadedModelOrSkip.Open(modelPath!, _output, out string? loadSkip);
        Skip.If(result is null, loadSkip ?? "load skipped");

        var (model, _, config) = (result!.Model, result.File, result.Config);

        _output.WriteLine(
            $"Config: arch={config.Architecture} vocab={config.VocabSize} hidden={config.HiddenSize} "
          + $"layers={config.NumLayers} heads={config.NumAttentionHeads} kv_heads={config.NumKvHeads} "
          + $"head_dim={config.HeadDim} intermediate={config.IntermediateSize} tied={config.TiedEmbeddings}");
        Assert.Equal(Core.Configuration.Architecture.QwenMoe, config.Architecture);
        Assert.NotNull(config.Moe);
        _output.WriteLine(
            $"Moe: num_experts={config.Moe!.NumExperts} top_k={config.Moe.NumExpertsPerTok} "
          + $"moe_intermediate={config.Moe.MoeIntermediateSize} norm_topk={config.Moe.NormTopKProb} "
          + $"sparse_step={config.Moe.DecoderSparseStep}");

        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];
        var sw = Stopwatch.StartNew();
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        sw.Stop();

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(config.VocabSize, logits.Shape[1]);

        var stats = ComputeStats(logits);
        _output.WriteLine(
            $"Forward: shape=[{logits.Shape[0]}, {logits.Shape[1]}] "
          + $"finite={stats.FiniteCount}/{stats.TotalCount} "
          + $"min={stats.Min:G4} max={stats.Max:G4} mean={stats.Mean:G4} stddev={stats.StdDev:G4} "
          + $"in {sw.Elapsed.TotalMilliseconds:F1} ms");

        Assert.Equal(stats.TotalCount, stats.FiniteCount);
        Assert.True(stats.StdDev > 0, "Logits have zero variance — forward pass likely degenerate.");
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

    /// <summary>
    /// Downloads a tiny-random Qwen-MoE repo into the local cache on first
    /// run. Returns path to <c>model.safetensors</c>, or null + reason on any
    /// failure (CI offline, HF outage, rate limit, repo deleted).
    /// </summary>
    private string? TryEnsureTinyQwenMoe(out string? skipReason)
    {
        foreach (var (repoId, files) in Candidates)
        {
            string cachedDir = Path.Combine(
                CacheDir, repoId.Replace('/', Path.DirectorySeparatorChar));
            string cachedModel = Path.Combine(cachedDir, "model.safetensors");
            string cachedConfig = Path.Combine(cachedDir, "config.json");

            if (File.Exists(cachedModel) && File.Exists(cachedConfig))
            {
                long size = new FileInfo(cachedModel).Length;
                if (size > MaxAllowedBytes)
                {
                    skipReason = $"cached {repoId} model is {size} bytes, exceeds cap {MaxAllowedBytes}";
                    return null;
                }
                skipReason = null;
                return cachedModel;
            }

            try
            {
                using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
                using var downloader = new HuggingFaceDownloader(http);

                string url = $"https://huggingface.co/{repoId}/resolve/main/model.safetensors";
                using (var head = new HttpRequestMessage(HttpMethod.Head, url))
                using (var headResp = http.SendAsync(head, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult())
                {
                    if (!headResp.IsSuccessStatusCode)
                    {
                        _output.WriteLine($"{repoId}: HEAD returned {(int)headResp.StatusCode}, trying next candidate");
                        continue;
                    }
                    long? total = headResp.Content.Headers.ContentLength;
                    if (total is long t && t > MaxAllowedBytes)
                    {
                        _output.WriteLine($"{repoId}: model.safetensors is {t} bytes > cap {MaxAllowedBytes}, skipping");
                        continue;
                    }
                }

                _output.WriteLine($"{repoId}: downloading model.safetensors + config.json to {cachedDir}");
                foreach (var filename in files)
                {
                    downloader.DownloadFileAsync(
                        repoId, filename, CacheDir, progress: null)
                        .GetAwaiter().GetResult();
                }

                if (File.Exists(cachedModel) && File.Exists(cachedConfig))
                {
                    skipReason = null;
                    return cachedModel;
                }
            }
            catch (Exception ex)
            {
                _output.WriteLine($"{repoId}: download failed with {ex.GetType().Name}: {ex.Message}");
            }
        }

        skipReason = "tiny-random Qwen-MoE unavailable (offline, rate limited, or all candidates failed)";
        return null;
    }

    private sealed record LoadedModel(
        DotLLM.Core.Models.IModel Model,
        IDisposable File,
        DotLLM.Core.Models.ModelConfig Config) : IDisposable
    {
        public static LoadedModel Open(string path)
        {
            var (model, file, config) = ModelLoader.LoadFromSafetensors(path);
            return new LoadedModel(model, file, config);
        }
        public void Dispose()
        {
            Model.Dispose();
            File.Dispose();
        }
    }

    /// <summary>
    /// Attempts to open the model. Skips on degenerate upstream geometries
    /// (e.g. <c>head_dim &lt; 2</c> — same fallback as the Mixtral test).
    /// </summary>
    private static class LoadedModelOrSkip
    {
        public static LoadedModel? Open(string path, ITestOutputHelper output, out string? skipReason)
        {
            try
            {
                skipReason = null;
                return LoadedModel.Open(path);
            }
            catch (ArgumentException ex) when (ex.Message.Contains("headDim", StringComparison.OrdinalIgnoreCase))
            {
                output.WriteLine(
                    $"Skipping forward-pass: tiny-random Qwen-MoE has a degenerate head_dim ({ex.Message}). "
                  + "Unit-test fixture exercises the dispatch path end-to-end.");
                skipReason = $"tiny-random Qwen-MoE head_dim incompatible: {ex.Message}";
                return null;
            }
        }
    }
}
