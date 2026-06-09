using System.Diagnostics;
using DotLLM.Core.Configuration;
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
/// Gated macro-benchmark for LoRA follow-up 4d.3. Measures real TinyLlama
/// forward cost with and without an active PEFT LoRA adapter so the synthetic
/// kernel-level prefill regression can be checked against full-model overhead.
/// </summary>
/// <remarks>
/// Required:
/// <list type="bullet">
///   <item><description><c>DOTLLM_TINYLLAMA_CHECKPOINT_PATH</c> or <c>C:/temp/dotllm-tinyllama</c>.</description></item>
///   <item><description><c>DOTLLM_TINYLLAMA_LORA_ADAPTER_PATH</c>, <c>C:/temp/dotllm-tinyllama-lora</c>, or <c>DOTLLM_TINYLLAMA_LORA_ADAPTER_REPO</c>.</description></item>
/// </list>
/// Optional:
/// <list type="bullet">
///   <item><description><c>DOTLLM_TINYLLAMA_LORA_BENCH_PREFILL_TOKENS</c> (default 64).</description></item>
///   <item><description><c>DOTLLM_TINYLLAMA_LORA_BENCH_SAMPLES</c> (default 3).</description></item>
/// </list>
/// The test intentionally asserts only shape/finite output. Performance values
/// are emitted to test output and should be compared on a stable local host.
/// </remarks>
public sealed class TinyLlamaLoraForwardBenchmarkTests
{
    private const string CheckpointEnvVar = "DOTLLM_TINYLLAMA_CHECKPOINT_PATH";
    private const string AdapterPathEnvVar = "DOTLLM_TINYLLAMA_LORA_ADAPTER_PATH";
    private const string AdapterRepoEnvVar = "DOTLLM_TINYLLAMA_LORA_ADAPTER_REPO";
    private const string PrefillTokensEnvVar = "DOTLLM_TINYLLAMA_LORA_BENCH_PREFILL_TOKENS";
    private const string SamplesEnvVar = "DOTLLM_TINYLLAMA_LORA_BENCH_SAMPLES";

    private const string ConventionalCheckpointPath = "C:/temp/dotllm-tinyllama";
    private const string ConventionalAdapterPath = "C:/temp/dotllm-tinyllama-lora";
    private const long MaxAdapterBytes = 512L * 1024 * 1024;

    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache");

    private readonly ITestOutputHelper _output;

    public TinyLlamaLoraForwardBenchmarkTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public void TinyLlama_RealCheckpoint_BaseVsLoraActive_ForwardTiming()
    {
        string? checkpointRoot = ResolveExistingPath(CheckpointEnvVar, ConventionalCheckpointPath);
        Skip.If(
            checkpointRoot is null,
            $"TinyLlama checkpoint not found. Set {CheckpointEnvVar} or place it at {ConventionalCheckpointPath}.");

        string? adapterDir = ResolveAdapterDirectory();
        Skip.If(
            adapterDir is null,
            $"TinyLlama LoRA adapter not found. Set {AdapterPathEnvVar}, {AdapterRepoEnvVar}, "
            + $"or place a PEFT adapter at {ConventionalAdapterPath}.");

        int prefillTokens = ReadPositiveInt(PrefillTokensEnvVar, 64);
        int samples = ReadPositiveInt(SamplesEnvVar, 3);

        var (model, source, config) = ModelLoader.LoadFromSafetensors(checkpointRoot!);
        try
        {
            Skip.If(
                config.Architecture != Architecture.Llama || config.HiddenSize != 2048,
                $"Expected real TinyLlama/Llama hidden=2048 checkpoint, got arch={config.Architecture} hidden={config.HiddenSize}.");

            using LoraAdapter adapter = PeftAdapterLoader.LoadFromDirectory("tinyllama-bench", adapterDir!, config);
            _output.WriteLine($"Checkpoint: {checkpointRoot}");
            _output.WriteLine($"Adapter:    {adapterDir}");
            _output.WriteLine(
                $"Model: arch={config.Architecture} vocab={config.VocabSize} hidden={config.HiddenSize} "
                + $"layers={config.NumLayers} heads={config.NumAttentionHeads} kv_heads={config.NumKvHeads}");
            _output.WriteLine(
                $"Adapter: rank={adapter.Rank} alpha={adapter.Alpha} adapted_layer_count={adapter.LayerWeights.Count}");
            _output.WriteLine($"Prefill tokens: {prefillTokens}; samples: {samples}");

            var prefillTokenIds = CreateTokenIds(prefillTokens, config.VocabSize);
            var prefillPositions = CreatePositions(prefillTokens);
            int[] decodeTokenIds = [prefillTokenIds[0]];
            int[] decodePositions = [0];

            // Warm both paths once so setup/JIT noise does not dominate the small
            // sample count. KV-cache is omitted to isolate model forward cost.
            using (model.Forward(prefillTokenIds, prefillPositions, deviceId: -1)) { }
            using (model.Forward(prefillTokenIds, prefillPositions, deviceId: -1, kvCache: null, adapter: adapter)) { }

            var basePrefillMs = new double[samples];
            var loraPrefillMs = new double[samples];
            var baseDecodeMs = new double[samples];
            var loraDecodeMs = new double[samples];

            for (int i = 0; i < samples; i++)
            {
                basePrefillMs[i] = MeasureForwardMs(model, prefillTokenIds, prefillPositions, adapter: null, config.VocabSize);
                loraPrefillMs[i] = MeasureForwardMs(model, prefillTokenIds, prefillPositions, adapter, config.VocabSize);
                baseDecodeMs[i] = MeasureForwardMs(model, decodeTokenIds, decodePositions, adapter: null, config.VocabSize);
                loraDecodeMs[i] = MeasureForwardMs(model, decodeTokenIds, decodePositions, adapter, config.VocabSize);

                _output.WriteLine(
                    $"sample={i + 1} "
                    + $"prefill_base_ms={basePrefillMs[i]:F2} prefill_lora_ms={loraPrefillMs[i]:F2} "
                    + $"decode_base_ms={baseDecodeMs[i]:F2} decode_lora_ms={loraDecodeMs[i]:F2}");
            }

            double basePrefillMedian = Median(basePrefillMs);
            double loraPrefillMedian = Median(loraPrefillMs);
            double baseDecodeMedian = Median(baseDecodeMs);
            double loraDecodeMedian = Median(loraDecodeMs);

            _output.WriteLine(
                $"median_prefill_base_ms={basePrefillMedian:F2} "
                + $"median_prefill_lora_ms={loraPrefillMedian:F2} "
                + $"prefill_overhead_pct={PercentOver(basePrefillMedian, loraPrefillMedian):F2}");
            _output.WriteLine(
                $"median_decode_base_ms={baseDecodeMedian:F2} "
                + $"median_decode_lora_ms={loraDecodeMedian:F2} "
                + $"decode_overhead_pct={PercentOver(baseDecodeMedian, loraDecodeMedian):F2}");
        }
        finally
        {
            model.Dispose();
            (source as IDisposable)?.Dispose();
        }
    }

    private static string? ResolveExistingPath(string envVar, string conventionalPath)
    {
        string? envPath = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(envPath) && PathExists(envPath))
            return envPath;

        return PathExists(conventionalPath) ? conventionalPath : null;
    }

    private string? ResolveAdapterDirectory()
    {
        string? adapterDir = ResolveExistingPath(AdapterPathEnvVar, ConventionalAdapterPath);
        if (adapterDir is not null && HasPeftAdapterFiles(adapterDir))
            return adapterDir;

        string? repoId = Environment.GetEnvironmentVariable(AdapterRepoEnvVar);
        if (string.IsNullOrWhiteSpace(repoId))
            return null;

        return TryDownloadAdapter(repoId);
    }

    private string? TryDownloadAdapter(string repoId)
    {
        string cachedDir = Path.Combine(CacheDir, repoId.Replace('/', Path.DirectorySeparatorChar));
        if (HasPeftAdapterFiles(cachedDir))
            return cachedDir;

        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
            using var dl = new HuggingFaceDownloader(http);
            string url = $"https://huggingface.co/{repoId}/resolve/main/adapter_model.safetensors";
            using var head = new HttpRequestMessage(HttpMethod.Head, url);
            using var headResp = http.SendAsync(head, HttpCompletionOption.ResponseHeadersRead)
                .GetAwaiter().GetResult();
            if (!headResp.IsSuccessStatusCode)
            {
                _output.WriteLine($"[skip-adapter] {repoId}: HEAD returned {(int)headResp.StatusCode}");
                return null;
            }

            long? total = headResp.Content.Headers.ContentLength;
            if (total is long bytes && bytes > MaxAdapterBytes)
            {
                _output.WriteLine($"[skip-adapter] {repoId}: adapter_model.safetensors is {bytes} bytes");
                return null;
            }

            dl.DownloadFileAsync(repoId, "adapter_model.safetensors", CacheDir, progress: null)
                .GetAwaiter().GetResult();
            dl.DownloadFileAsync(repoId, "adapter_config.json", CacheDir, progress: null)
                .GetAwaiter().GetResult();

            return HasPeftAdapterFiles(cachedDir) ? cachedDir : null;
        }
        catch (Exception ex)
        {
            _output.WriteLine($"[skip-adapter] {repoId}: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static bool HasPeftAdapterFiles(string dir)
        => Directory.Exists(dir)
           && File.Exists(Path.Combine(dir, "adapter_model.safetensors"))
           && File.Exists(Path.Combine(dir, "adapter_config.json"));

    private static bool PathExists(string path) => Directory.Exists(path) || File.Exists(path);

    private static int ReadPositiveInt(string envVar, int fallback)
    {
        string? value = Environment.GetEnvironmentVariable(envVar);
        return int.TryParse(value, out int parsed) && parsed > 0 ? parsed : fallback;
    }

    private static int[] CreateTokenIds(int count, int vocabSize)
    {
        var tokenIds = new int[count];
        int max = Math.Max(2, Math.Min(vocabSize, 32000));
        for (int i = 0; i < tokenIds.Length; i++)
            tokenIds[i] = 1 + (i % (max - 1));
        return tokenIds;
    }

    private static int[] CreatePositions(int count)
    {
        var positions = new int[count];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;
        return positions;
    }

    private static double MeasureForwardMs(
        IModel model,
        int[] tokenIds,
        int[] positions,
        LoraAdapter? adapter,
        int expectedVocab)
    {
        long start = Stopwatch.GetTimestamp();
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kvCache: null, adapter: adapter);
        long stop = Stopwatch.GetTimestamp();

        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(expectedVocab, logits.Shape[1]);
        AssertFinite(logits);

        return (stop - start) * 1000.0 / Stopwatch.Frequency;
    }

    private static unsafe void AssertFinite(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        var values = new ReadOnlySpan<float>((void*)logits.DataPointer, total);
        for (int i = 0; i < values.Length; i++)
            Assert.True(float.IsFinite(values[i]), $"Non-finite logit at index {i}: {values[i]}");
    }

    private static double Median(double[] values)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        int n = sorted.Length;
        return n % 2 == 1
            ? sorted[n / 2]
            : (sorted[(n / 2) - 1] + sorted[n / 2]) / 2.0;
    }

    private static double PercentOver(double baseline, double candidate)
        => baseline <= 0 ? 0 : ((candidate / baseline) - 1.0) * 100.0;
}
