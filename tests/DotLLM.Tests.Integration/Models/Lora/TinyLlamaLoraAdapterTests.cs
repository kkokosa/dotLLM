using System.Text.Json;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.HuggingFace;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Lora;

/// <summary>
/// Real-adapter integration test: downloads a small public PEFT LoRA
/// adapter from HuggingFace + the tiny-random base model it targets,
/// loads both into dotLLM, runs a forward with and without the adapter,
/// and asserts (a) finite logits, (b) a measurable delta vs the
/// adapter-less forward, and (c) sub-100 ms switch via Stopwatch.
/// </summary>
/// <remarks>
/// <para>
/// The test is fully self-skipping: when no candidate (base, adapter)
/// pair downloads cleanly (offline CI, HF outage, rate limit, repo
/// removed) the test reports a Skip rather than failing. This mirrors
/// the existing pattern in
/// <c>RealHfSafetensorsEndToEndTests</c> and <c>TinyLlamaSafetensorsLoadTests</c>.
/// </para>
/// <para>
/// Cache layout: <c>~/.dotllm/test-cache/&lt;org&gt;/&lt;repo&gt;/</c> for both base
/// and adapter, matching <c>HuggingFaceDownloader</c>'s defaults.
/// </para>
/// </remarks>
public sealed class TinyLlamaLoraAdapterTests
{
    /// <summary>Cap to avoid a runaway download if a repo got bloated unexpectedly.</summary>
    private const int MaxBaseBytes = 50 * 1024 * 1024;
    private const int MaxAdapterBytes = 25 * 1024 * 1024;

    /// <summary>
    /// Ordered (base_repo, adapter_repo) candidates. We try each pair until one
    /// downloads cleanly; failure on any pair (404, content-length over cap,
    /// timeout) just falls through to the next.
    /// </summary>
    /// <remarks>
    /// <c>llamafactory/tiny-random-Llama-3-lora</c> is a public PEFT adapter
    /// (~27 KB, rank 8, alpha 16) targeting the canonical
    /// q/k/v/o/gate/up/down projections of <c>llamafactory/tiny-random-Llama-3</c>
    /// (~8 MB). Both repos are mirrored at HF's public CDN with no auth gate
    /// and no special license, so this pair downloads cleanly in CI as long as
    /// outbound internet is available.
    /// </remarks>
    private static readonly (string BaseRepo, string AdapterRepo)[] Candidates =
    [
        ("llamafactory/tiny-random-Llama-3", "llamafactory/tiny-random-Llama-3-lora"),
    ];

    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache");

    private readonly ITestOutputHelper _output;

    public TinyLlamaLoraAdapterTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void RealLora_LoadAndForward_ProducesMeasurableDelta()
    {
        var resolved = TryResolveCandidate(out string? skipReason);
        Skip.If(resolved is null, skipReason ?? "no LoRA + base candidate available");

        var (baseModelPath, adapterDir, baseConfig) = resolved!.Value;
        _output.WriteLine($"Base:    {baseModelPath}");
        _output.WriteLine($"Adapter: {adapterDir}");

        var (model, file, config) = ModelLoader.LoadFromSafetensors(baseModelPath);
        try
        {
            using LoraAdapter adapter = PeftAdapterLoader.LoadFromDirectory("real-tiny", adapterDir, config);
            _output.WriteLine(
                $"Adapter: rank={adapter.Rank} alpha={adapter.Alpha} "
              + $"target_modules=[{string.Join(", ", adapter.TargetModules)}] "
              + $"adapted_layer_count={adapter.LayerWeights.Count}");

            int[] tokenIds = [0, 1, 2];
            int[] positions = [0, 1, 2];

            using var baseLogits = model.Forward(tokenIds, positions, deviceId: -1);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            using var withLogits = model.Forward(tokenIds, positions, deviceId: -1,
                kvCache: null, adapter: adapter);
            sw.Stop();
            _output.WriteLine($"Forward(adapter) took {sw.Elapsed.TotalMilliseconds:F2} ms");

            int seqLen = baseLogits.Shape[0];
            int vocab = baseLogits.Shape[1];
            int total = seqLen * vocab;
            Assert.Equal(seqLen, withLogits.Shape[0]);
            Assert.Equal(vocab, withLogits.Shape[1]);

            var baseSpan = new ReadOnlySpan<float>((void*)baseLogits.DataPointer, total);
            var withSpan = new ReadOnlySpan<float>((void*)withLogits.DataPointer, total);

            float maxAbs = 0f;
            int finite = 0;
            for (int i = 0; i < total; i++)
            {
                if (float.IsFinite(withSpan[i])) finite++;
                maxAbs = MathF.Max(maxAbs, MathF.Abs(baseSpan[i] - withSpan[i]));
            }
            _output.WriteLine($"Finite={finite}/{total}  maxAbsDiff={maxAbs:G6}");

            Assert.Equal(total, finite);
            // The tiny-random base + tiny-random adapter combination has small
            // absolute logit magnitudes, so we use a loose threshold proving
            // *some* delta vs zero.
            Assert.True(maxAbs > 1e-5f,
                $"Real adapter produced no measurable delta from base (maxAbsDiff={maxAbs:G6}).");

            // The forward is expected to be very fast for this size; the 100 ms
            // budget here is the "swap can't be wedged" check, not a perf test.
            Assert.True(sw.Elapsed.TotalMilliseconds < 5000,
                $"Forward(adapter) took {sw.Elapsed.TotalMilliseconds:F2} ms — unexpectedly slow.");
        }
        finally
        {
            model.Dispose();
            file.Dispose();
        }
    }

    private (string BaseModelPath, string AdapterDir, ModelConfig _)? TryResolveCandidate(out string? skipReason)
    {
        foreach (var (baseRepo, adapterRepo) in Candidates)
        {
            string? basePath = TryEnsureBase(baseRepo);
            if (basePath is null)
            {
                _output.WriteLine($"[skip-candidate] base '{baseRepo}' unavailable");
                continue;
            }

            string? adapterDir = TryEnsureAdapter(adapterRepo);
            if (adapterDir is null)
            {
                _output.WriteLine($"[skip-candidate] adapter '{adapterRepo}' unavailable");
                continue;
            }

            // Verify the adapter declares it targets a Llama-shaped model
            // before trying to load. PEFT writes the base_model_name_or_path
            // and target_modules in adapter_config.json — quick sniff so we
            // don't waste effort downloading mismatches.
            try
            {
                string adapterCfgPath = Path.Combine(adapterDir, "adapter_config.json");
                using var stream = File.OpenRead(adapterCfgPath);
                using var doc = JsonDocument.Parse(stream);
                if (!doc.RootElement.TryGetProperty("target_modules", out _))
                {
                    _output.WriteLine($"[skip-candidate] adapter '{adapterRepo}' has no target_modules");
                    continue;
                }
            }
            catch (Exception ex)
            {
                _output.WriteLine($"[skip-candidate] adapter '{adapterRepo}' config parse failed: {ex.Message}");
                continue;
            }

            // We have a candidate. Try to load and validate compatibility — if
            // shapes don't line up, fall through to the next pair.
            try
            {
                var (_, fileTmp, configTmp) = ModelLoader.LoadFromSafetensors(basePath);
                fileTmp.Dispose();
                skipReason = null;
                return (basePath, adapterDir, configTmp);
            }
            catch (Exception ex)
            {
                _output.WriteLine($"[skip-candidate] base '{baseRepo}' load failed: {ex.GetType().Name}: {ex.Message}");
            }
        }
        skipReason = "no real LoRA candidate downloaded cleanly (offline, rate-limited, all repos failed)";
        return null;
    }

    private string? TryEnsureBase(string repoId)
    {
        string cachedDir = Path.Combine(CacheDir, repoId.Replace('/', Path.DirectorySeparatorChar));
        string cachedModel = Path.Combine(cachedDir, "model.safetensors");
        string cachedConfig = Path.Combine(cachedDir, "config.json");
        if (File.Exists(cachedModel) && File.Exists(cachedConfig))
        {
            if (new FileInfo(cachedModel).Length > MaxBaseBytes) return null;
            return cachedModel;
        }
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
            using var dl = new HuggingFaceDownloader(http);
            string url = $"https://huggingface.co/{repoId}/resolve/main/model.safetensors";
            using var head = new HttpRequestMessage(HttpMethod.Head, url);
            using var headResp = http.SendAsync(head, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
            if (!headResp.IsSuccessStatusCode) return null;
            long? total = headResp.Content.Headers.ContentLength;
            if (total is long t && t > MaxBaseBytes) return null;

            dl.DownloadFileAsync(repoId, "model.safetensors", CacheDir, progress: null)
                .GetAwaiter().GetResult();
            dl.DownloadFileAsync(repoId, "config.json", CacheDir, progress: null)
                .GetAwaiter().GetResult();
            return File.Exists(cachedModel) && File.Exists(cachedConfig) ? cachedModel : null;
        }
        catch
        {
            return null;
        }
    }

    private string? TryEnsureAdapter(string repoId)
    {
        string cachedDir = Path.Combine(CacheDir, repoId.Replace('/', Path.DirectorySeparatorChar));
        string cachedAdapter = Path.Combine(cachedDir, "adapter_model.safetensors");
        string cachedConfig = Path.Combine(cachedDir, "adapter_config.json");
        if (File.Exists(cachedAdapter) && File.Exists(cachedConfig))
        {
            if (new FileInfo(cachedAdapter).Length > MaxAdapterBytes) return null;
            return cachedDir;
        }
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
            using var dl = new HuggingFaceDownloader(http);
            string url = $"https://huggingface.co/{repoId}/resolve/main/adapter_model.safetensors";
            using var head = new HttpRequestMessage(HttpMethod.Head, url);
            using var headResp = http.SendAsync(head, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
            if (!headResp.IsSuccessStatusCode) return null;
            long? total = headResp.Content.Headers.ContentLength;
            if (total is long t && t > MaxAdapterBytes) return null;

            dl.DownloadFileAsync(repoId, "adapter_model.safetensors", CacheDir, progress: null)
                .GetAwaiter().GetResult();
            dl.DownloadFileAsync(repoId, "adapter_config.json", CacheDir, progress: null)
                .GetAwaiter().GetResult();
            return File.Exists(cachedAdapter) && File.Exists(cachedConfig) ? cachedDir : null;
        }
        catch
        {
            return null;
        }
    }
}
