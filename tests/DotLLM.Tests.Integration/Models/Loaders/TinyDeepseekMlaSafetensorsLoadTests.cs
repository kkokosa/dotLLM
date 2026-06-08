using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.HuggingFace;
using DotLLM.Models;
using DotLLM.Models.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// End-to-end verification that <see cref="HfConfigExtractor"/> correctly
/// detects real-world tiny-random DeepSeek-V2/V3 checkpoints — the first
/// integration-level coverage for the MLA attention family.
/// </summary>
/// <remarks>
/// <para>
/// Downloads one of several tiny-random DeepSeek checkpoints on first run
/// (~2–20 MB each) and asserts:
/// <list type="bullet">
///   <item><c>Architecture.DeepSeekV2</c> or <c>Architecture.DeepSeekV3</c>.</item>
///   <item><c>AttentionType.MLA</c>.</item>
///   <item><c>MlaConfig</c> is populated with positive ranks and dims.</item>
/// </list>
/// MoE config assertions land in the MoE extraction PR; this PR only covers
/// the MLA attention foundation.
/// </para>
/// <para>
/// With the MLA integration PR, <see cref="ModelLoader.LoadFromSafetensors"/>
/// now dispatches DeepSeek-V2/V3 into <see cref="DotLLM.Models.Architectures.TransformerModel.LoadFromSafetensors"/>
/// which routes attention through the MLA branch backed by
/// <see cref="DotLLM.Cpu.Kernels.MlaAttention"/>. The PoC skips the KV-cache
/// optimisation (the scalar kernel re-runs the full MLA forward per call);
/// that is tracked as a follow-up.
/// </para>
/// <para>
/// Cache location: <c>~/.dotllm/test-cache/&lt;repo&gt;/</c>. 50 MB cap.
/// Gracefully skips if all candidates are offline/rate-limited.
/// </para>
/// </remarks>
public sealed class TinyDeepseekMlaSafetensorsLoadTests
{
    /// <summary>All candidate tiny-random checkpoints are well under 50 MB.</summary>
    private const int MaxAllowedBytes = 50 * 1024 * 1024;

    /// <summary>
    /// Ordered candidate repos. First reachable wins. Each ships a
    /// Deepseek{V2,V3}ForCausalLM safetensors checkpoint with full MLA + MoE
    /// config under ~20 MB.
    /// </summary>
    private static readonly (string RepoId, Architecture ExpectedArch, string[] Files)[] Candidates =
    [
        ("yujiepan/deepseek-v2-tiny-random",   Architecture.DeepSeekV2, ["model.safetensors", "config.json"]),
        ("yujiepan/deepseek-v3-tiny-random",   Architecture.DeepSeekV3, ["model.safetensors", "config.json"]),
        ("katuni4ka/tiny-random-deepseek-v3",  Architecture.DeepSeekV3, ["model.safetensors", "config.json"]),
    ];

    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache");

    private readonly ITestOutputHelper _output;

    public TinyDeepseekMlaSafetensorsLoadTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Proves <see cref="HfConfigExtractor"/> correctly detects DeepSeek-V2/V3
    /// and populates <see cref="DotLLM.Core.Models.MlaConfig"/> from a real HF
    /// checkpoint's <c>config.json</c>. The strongest assertion we can make
    /// today — the actual weight-loader dispatch is a follow-up. MoE config
    /// population is verified by the MoE extraction PR.
    /// </summary>
    [SkippableFact]
    public void RealDeepseekConfig_IsDetectedAsMla()
    {
        var located = TryEnsureTinyDeepseek(out string? skipReason);
        Skip.If(located is null, skipReason ?? "tiny-random DeepSeek download unavailable");
        var (modelPath, expectedArch) = located.Value;

        string configPath = Path.Combine(Path.GetDirectoryName(modelPath)!, "config.json");
        Assert.True(System.IO.File.Exists(configPath), "config.json must be co-located with the model.");

        var cfg = HfConfigExtractor.Extract(System.IO.File.ReadAllText(configPath));
        _output.WriteLine(
            $"Real HF config: arch={cfg.Architecture} attn={cfg.AttentionType} "
          + $"hidden={cfg.HiddenSize} layers={cfg.NumLayers} heads={cfg.NumAttentionHeads} "
          + $"head_dim={cfg.HeadDim} intermediate={cfg.IntermediateSize} vocab={cfg.VocabSize}");

        Assert.Equal(expectedArch, cfg.Architecture);
        Assert.Equal(AttentionType.MLA, cfg.AttentionType);

        Assert.NotNull(cfg.MlaConfig);
        var mla = cfg.MlaConfig!;
        _output.WriteLine(
            $"MlaConfig: kv_lora_rank={mla.KvLoraRank} q_lora_rank={mla.QLoraRank} "
          + $"qk_nope={mla.QkNopeHeadDim} qk_rope={mla.QkRopeHeadDim} v_head={mla.VHeadDim} "
          + $"qk_head={mla.QkHeadDim} rope_theta={mla.RopeTheta}");
        Assert.True(mla.KvLoraRank > 0, "kv_lora_rank must be positive");
        Assert.True(mla.QkNopeHeadDim >= 0, "qk_nope_head_dim must be non-negative");
        Assert.True(mla.QkRopeHeadDim > 0 && mla.QkRopeHeadDim % 2 == 0, "qk_rope_head_dim must be positive and even");
        Assert.True(mla.VHeadDim > 0, "v_head_dim must be positive");
        Assert.True(mla.QLoraRank >= 0, "q_lora_rank must be non-negative (0 = no factorisation)");
        // HeadDim in ModelConfig reflects qk_head_dim for MLA.
        Assert.Equal(mla.QkHeadDim, cfg.HeadDim);
    }

    /// <summary>
    /// End-to-end: load a tiny-random DeepSeek-V2/V3 checkpoint, run a prefill
    /// forward pass, and assert the resulting logits are finite with non-zero
    /// variance. Exercises the full MLA load + dispatch path:
    /// <see cref="HfConfigExtractor"/> → <see cref="ModelLoader.LoadFromSafetensors"/>
    /// → <see cref="DotLLM.Models.Architectures.TransformerModel.LoadFromSafetensors"/>
    /// (<c>LoadDeepSeekMlaLayer</c> per layer) →
    /// <see cref="DotLLM.Cpu.Kernels.MlaAttention.Execute"/> per layer.
    /// </summary>
    [SkippableFact]
    public void DeepseekLoadFromSafetensors_Forward_ProducesFiniteLogits()
    {
        var located = TryEnsureTinyDeepseek(out string? skipReason);
        Skip.If(located is null, skipReason ?? "tiny-random DeepSeek download unavailable");
        var (modelPath, expectedArch) = located.Value;

        var (model, file, cfg) = ModelLoader.LoadFromSafetensors(modelPath);
        try
        {
            Assert.Equal(expectedArch, cfg.Architecture);
            Assert.Equal(AttentionType.MLA, cfg.AttentionType);
            Assert.NotNull(cfg.MlaConfig);

            // Clamp the prompt to the model's supported context. Tiny-random
            // checkpoints usually keep max_position_embeddings small (2k–163k
            // for V3) so 4 is safe everywhere.
            int[] tokenIds = [1, 2, 3, 4];
            int[] positions = [0, 1, 2, 3];

            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

            Assert.Equal(2, logits.Shape.Rank);
            Assert.Equal(tokenIds.Length, logits.Shape[0]);
            Assert.Equal(cfg.VocabSize, logits.Shape[1]);

            var stats = ComputeStats(logits);
            _output.WriteLine(
                $"MLA forward: shape=[{logits.Shape[0]},{logits.Shape[1]}] "
              + $"finite={stats.FiniteCount}/{stats.TotalCount} "
              + $"mean={stats.Mean:F4} std={stats.StdDev:F4} "
              + $"min={stats.Min:F4} max={stats.Max:F4} argmax={stats.ArgmaxFirstRow}");
            Assert.Equal(stats.TotalCount, stats.FiniteCount);
            Assert.True(stats.StdDev > 0.0f,
                $"Logits degenerate: std={stats.StdDev} — MLA branch wired incorrectly.");
        }
        finally
        {
            (model as IDisposable)?.Dispose();
            file.Dispose();
        }
    }

    private static unsafe LogitStats ComputeStats(ITensor logits)
    {
        int rows = logits.Shape[0];
        int cols = logits.Shape[1];
        int total = rows * cols;
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

        int argmax = 0;
        float best = float.NegativeInfinity;
        for (int i = 0; i < cols; i++)
            if (span[i] > best) { best = span[i]; argmax = i; }

        return new LogitStats(total, finite, (float)mean, (float)stddev, min, max, argmax);
    }

    private readonly record struct LogitStats(
        int TotalCount, int FiniteCount, float Mean, float StdDev, float Min, float Max, int ArgmaxFirstRow);

    /// <summary>
    /// Downloads a tiny-random DeepSeek-V2 or V3 repo into the local cache.
    /// Returns path + detected architecture, or null + reason on failure
    /// (offline, rate limit, all candidates 404).
    /// </summary>
    private (string ModelPath, Architecture Arch)? TryEnsureTinyDeepseek(out string? skipReason)
    {
        foreach (var (repoId, expectedArch, files) in Candidates)
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
                return (cachedModel, expectedArch);
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

                _output.WriteLine($"{repoId}: downloading to {cachedDir}");
                foreach (var filename in files)
                {
                    downloader.DownloadFileAsync(
                        repoId, filename, CacheDir, progress: null)
                        .GetAwaiter().GetResult();
                }

                if (File.Exists(cachedModel) && File.Exists(cachedConfig))
                {
                    skipReason = null;
                    return (cachedModel, expectedArch);
                }
            }
            catch (Exception ex)
            {
                _output.WriteLine($"{repoId}: download failed with {ex.GetType().Name}: {ex.Message}");
            }
        }

        skipReason = "tiny-random DeepSeek V2/V3 unavailable (offline, rate limited, or all candidates failed)";
        return null;
    }
}
