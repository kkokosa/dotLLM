using System.Text.Json;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.HuggingFace;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan.Lora;

/// <summary>
/// Real-adapter integration test for the Vulkan LoRA path. Sister to the
/// CPU <c>TinyLlamaLoraAdapterTests</c>: downloads the same tiny-random
/// PEFT adapter + base, runs CPU-with-adapter and Vulkan-with-adapter, and
/// asserts the two paths agree within abs 5e-3 / rel 1e-3 — the standard
/// Vulkan end-to-end parity bar.
/// </summary>
/// <remarks>
/// <para>
/// Self-skipping: when no candidate (base, adapter) pair downloads cleanly
/// (offline CI, HF outage, rate limit, repo removed) the test reports a
/// Skip rather than failing. Vulkan unavailability also self-skips.
/// </para>
/// <para>
/// Cache layout: <c>~/.dotllm/test-cache/&lt;org&gt;/&lt;repo&gt;/</c> for both
/// base and adapter — same as the CPU sister test, so a previous CPU run
/// warms the cache for this test.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class VulkanTinyLlamaLoraAdapterTests
{
    private const int MaxBaseBytes = 50 * 1024 * 1024;
    private const int MaxAdapterBytes = 25 * 1024 * 1024;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private static readonly (string BaseRepo, string AdapterRepo)[] Candidates =
    [
        ("llamafactory/tiny-random-Llama-3", "llamafactory/tiny-random-Llama-3-lora"),
    ];

    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache");

    private readonly ITestOutputHelper _output;

    public VulkanTinyLlamaLoraAdapterTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void RealLora_VulkanMatchesCpu_OnSameTokens()
    {
        SkipIfVulkanUnavailable(out string spvDir);

        var resolved = TryResolveCandidate(out string? skipReason);
        Skip.If(resolved is null, skipReason ?? "no LoRA + base candidate available");

        var (baseModelPath, adapterDir, _) = resolved!.Value;
        _output.WriteLine($"Base:    {baseModelPath}");
        _output.WriteLine($"Adapter: {adapterDir}");

        // Single-token forward (decode path). The tiny-random-Llama-3 base
        // has hidden_size=16, num_kv_heads=4, head_dim=4 — at seqLen>1 the
        // KHR_cooperative_matrix F16 GEMM kernel rejects the K=16 contraction
        // (it requires K % 32 == 0). The decode path avoids that constraint
        // and is the more important regime to pin for a real LoRA forward.
        int[] tokenIds = [0];
        int[] positions = [0];

        // ── CPU oracle (with adapter) ─────────────────────────────────
        // ModelLoader returns (model, file, config); we keep the file alive
        // for the duration of the forward and dispose afterwards.
        float[] cpuLogits;
        int seqLen, vocab;
        ModelConfig config;
        {
            var (cpuModel, cpuFile, cpuConfig) = ModelLoader.LoadFromSafetensors(baseModelPath);
            try
            {
                config = cpuConfig;
                using LoraAdapter cpuAdapter = PeftAdapterLoader.LoadFromDirectory("real-cpu", adapterDir, cpuConfig);
                _output.WriteLine(
                    $"Adapter: rank={cpuAdapter.Rank} alpha={cpuAdapter.Alpha} "
                  + $"adapted_layer_count={cpuAdapter.LayerWeights.Count}");

                using ITensor logits = cpuModel.Forward(tokenIds, positions, deviceId: -1,
                    kvCache: null, adapter: cpuAdapter);
                seqLen = logits.Shape[0];
                vocab = logits.Shape[1];
                cpuLogits = CopyLogits(logits);
            }
            finally
            {
                cpuModel.Dispose();
                cpuFile.Dispose();
            }
        }

        // ── Vulkan under test (with adapter) ──────────────────────────
        float[] vkLogits;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        {
            using var sf = SafetensorsFile.Open(baseModelPath);
            using var vkModel = VulkanTransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using LoraAdapter vkAdapter = PeftAdapterLoader.LoadFromDirectory("real-vk", adapterDir, config);

            using ITensor logits = vkModel.Forward(tokenIds, positions, deviceId: -1,
                kvCache: null, adapter: vkAdapter);
            // Vulkan returns last-token logits [1, vocab].
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(vocab, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }
        sw.Stop();
        _output.WriteLine($"Vulkan Forward(adapter) took {sw.Elapsed.TotalMilliseconds:F2} ms");

        // CPU returns [seqLen, vocab]; compare last row vs Vulkan single row.
        int lastRow = seqLen - 1;
        int finite = 0;
        float maxAbs = 0f, maxRel = 0f;
        int errors = 0;
        for (int c = 0; c < vocab; c++)
        {
            float cpu = cpuLogits[lastRow * vocab + c];
            float vk = vkLogits[c];
            if (!float.IsFinite(vk)) { errors++; continue; }
            finite++;

            float diff = MathF.Abs(cpu - vk);
            float rel = diff / MathF.Max(MathF.Abs(cpu), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;

            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            if (diff > bar) errors++;
        }
        _output.WriteLine($"Finite={finite}/{vocab}  maxAbs={maxAbs:G6}  maxRel={maxRel:G6}  errors={errors}");

        Assert.Equal(0, errors);

        // The adapter-active Vulkan forward should still be sub-5s for this
        // tiny size; the CPU sister test hit ~72ms locally, Vulkan should be
        // similar at this scale (the adapter upload happens on first forward).
        Assert.True(sw.Elapsed.TotalMilliseconds < 5000,
            $"Vulkan Forward(adapter) took {sw.Elapsed.TotalMilliseconds:F2} ms — unexpectedly slow.");
    }

    private static void SkipIfVulkanUnavailable(out string spvDir)
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string? found = FindSpvDir();
        Skip.If(
            found is null,
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
        spvDir = found!;
    }

    private static string? FindSpvDir()
    {
        string[] candidates =
        [
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        ];
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        return null;
    }

    // ────────────────────────────────────────────────────────────────────
    // HF candidate resolution — mirrors the CPU sister test exactly.
    // ────────────────────────────────────────────────────────────────────

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

            dl.DownloadFileAsync(repoId, "model.safetensors", CacheDir, progress: null).GetAwaiter().GetResult();
            dl.DownloadFileAsync(repoId, "config.json", CacheDir, progress: null).GetAwaiter().GetResult();
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

            dl.DownloadFileAsync(repoId, "adapter_model.safetensors", CacheDir, progress: null).GetAwaiter().GetResult();
            dl.DownloadFileAsync(repoId, "adapter_config.json", CacheDir, progress: null).GetAwaiter().GetResult();
            return File.Exists(cachedAdapter) && File.Exists(cachedConfig) ? cachedDir : null;
        }
        catch
        {
            return null;
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }
}
