using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.HuggingFace;
using DotLLM.Models;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// End-to-end verification that
/// <see cref="ModelLoader.LoadFromSafetensors"/> can open a real HuggingFace
/// tiny-random Llama checkpoint and run a forward pass that produces
/// finite vocab-sized logits.
/// </summary>
/// <remarks>
/// <para>
/// Tiny-random models are published by <c>hf-internal-testing</c> specifically
/// as CI fixtures: a few MB each, architecturally correct, random weights.
/// We are proving the loading plumbing here — the safetensors header is
/// parsed, the HF tensor names resolve, the bf16/F32 ingest path matches
/// the config, and the forward pass returns <c>[seq, vocab]</c> logits
/// without NaN/Inf. We are NOT asserting any semantic output quality:
/// random weights produce random logits.
/// </para>
/// <para>
/// The test fetches <c>model.safetensors</c> + <c>config.json</c> into
/// <c>~/.dotllm/test-cache/&lt;repo&gt;/</c> on first run and caches them for
/// subsequent runs. Cap: 50 MB. If the download fails (offline CI, HF
/// outage, rate limit, repo deleted) the test skips gracefully rather than
/// failing, per the pattern established by Mamba3 reference tests.
/// </para>
/// </remarks>
public sealed class TinyLlamaSafetensorsLoadTests
{
    /// <summary>Per the HF Hub page (2026-04), 1.0 M params, F32 → ~4 MB.</summary>
    private const int MaxAllowedBytes = 50 * 1024 * 1024;

    /// <summary>
    /// Ordered candidate repos. First hit wins; each subsequent entry is a
    /// fallback if the prior is unreachable / deleted.
    /// </summary>
    private static readonly (string RepoId, string[] Files)[] Candidates =
    [
        ("hf-internal-testing/tiny-random-LlamaForCausalLM", ["model.safetensors", "config.json"]),
        ("trl-internal-testing/tiny-random-LlamaForCausalLM", ["model.safetensors", "config.json"]),
    ];

    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache");

    private readonly ITestOutputHelper _output;

    public TinyLlamaSafetensorsLoadTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void LoadAndForwardPass_ProducesFiniteVocabLogits()
    {
        string? modelPath = TryEnsureTinyLlama(out string? skipReason);
        Skip.If(modelPath is null, skipReason ?? "tiny-random Llama download unavailable");

        _output.WriteLine($"Loaded tiny-random Llama from: {modelPath}");

        using var result = LoadedModel.Open(modelPath!);
        var (model, _, config) = (result.Model, result.File, result.Config);

        _output.WriteLine(
            $"Config: arch={config.Architecture} vocab={config.VocabSize} hidden={config.HiddenSize} "
          + $"layers={config.NumLayers} heads={config.NumAttentionHeads} kv_heads={config.NumKvHeads} "
          + $"head_dim={config.HeadDim} intermediate={config.IntermediateSize} tied={config.TiedEmbeddings}");

        // Forward: [0, 1, 2] — small prompt, any valid in-vocab token ids.
        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];
        var sw = Stopwatch.StartNew();
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        sw.Stop();

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(config.VocabSize, logits.Shape[1]);

        // Finite-check + variance sanity (not a quality assertion — random weights
        // produce a distribution but must not degenerate to a constant vector).
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
    /// Downloads <c>model.safetensors</c> + <c>config.json</c> for the first
    /// reachable tiny-random repo. Returns the path to the cached safetensors
    /// on success, or null + a skip reason on any failure.
    /// </summary>
    private string? TryEnsureTinyLlama(out string? skipReason)
    {
        foreach (var (repoId, files) in Candidates)
        {
            string cachedDir = Path.Combine(
                CacheDir, repoId.Replace('/', Path.DirectorySeparatorChar));
            string cachedModel = Path.Combine(cachedDir, "model.safetensors");
            string cachedConfig = Path.Combine(cachedDir, "config.json");

            // Cache hit — skip the download.
            if (File.Exists(cachedModel) && File.Exists(cachedConfig))
            {
                long size = new FileInfo(cachedModel).Length;
                if (size > MaxAllowedBytes)
                {
                    skipReason = $"tiny-random {repoId} cached model is {size} bytes, exceeds cap {MaxAllowedBytes}";
                    return null;
                }
                skipReason = null;
                return cachedModel;
            }

            try
            {
                using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
                using var downloader = new HuggingFaceDownloader(http);

                // Probe content-length first — bail out if the model exceeds the cap.
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
                // Try the next candidate
            }
        }

        skipReason = "tiny-random Llama unavailable (offline, rate limited, or all candidates failed)";
        return null;
    }

    /// <summary>
    /// Scoped helper that disposes both <see cref="ModelLoader"/> outputs
    /// in the correct order (model first, then safetensors file).
    /// </summary>
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
}
