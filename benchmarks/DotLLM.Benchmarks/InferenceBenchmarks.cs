using BenchmarkDotNet.Attributes;
using DotLLM.Benchmarks.Columns;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Engine;
using DotLLM.HuggingFace;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks;

/// <summary>
/// Supported models for inference benchmarks.
/// Default param is SmolLM_135M (~145 MB); add larger models via <c>--filter</c>.
/// </summary>
public enum BenchmarkModel
{
    SmolLM_135M,
    Llama32_1B,
    Llama32_3B
}

/// <summary>
/// End-to-end inference benchmarks measuring prefill + decode throughput.
/// Custom BDN columns display tok/s via a file-based metrics bridge.
/// </summary>
[SimpleJob(warmupCount: 2, iterationCount: 5)]
public class InferenceBenchmarks
{
    private static readonly Dictionary<BenchmarkModel, (string RepoId, string Filename, int ApproxSizeMB)> s_models = new()
    {
        [BenchmarkModel.SmolLM_135M] = ("QuantFactory/SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf", 145),
        [BenchmarkModel.Llama32_1B] = ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf", 1300),
        [BenchmarkModel.Llama32_3B] = ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q8_0.gguf", 3400),
    };

    private const string DefaultPrompt = "The capital of France is";
    private const int DefaultMaxTokens = 20;

    /// <summary>Model to benchmark. Use <c>--filter *SmolLM*</c> etc. to select one.</summary>
    [ParamsAllValues]
    public BenchmarkModel Model { get; set; }

    private GgufFile _gguf = null!;
    private IModel _model = null!;
    private BpeTokenizer _tokenizer = null!;
    private TextGenerator _generator = null!;
    private string _modelPath = null!;
    private string _prompt = DefaultPrompt;
    private int _maxTokens = DefaultMaxTokens;

    /// <summary>
    /// Metrics key used for the file bridge. Normally the <see cref="BenchmarkModel"/> enum name;
    /// when <c>DOTLLM_BENCH_MODEL_PATH</c> is set, the filename stem of the override path.
    /// </summary>
    private string _metricsKey = null!;

    // Accumulate timings across BDN iterations for median computation.
    private readonly List<InferenceTimings> _timings = new();

    [GlobalSetup]
    public void Setup()
    {
        var envModelPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MODEL_PATH");
        if (!string.IsNullOrEmpty(envModelPath) && File.Exists(envModelPath))
        {
            _modelPath = envModelPath;
            _metricsKey = Path.GetFileNameWithoutExtension(envModelPath);
            Console.WriteLine($"*** Model override active: {_metricsKey} ***");
            Console.WriteLine($"    Path: {envModelPath}");
            Console.WriteLine($"    (BDN 'Model' column shows enum name — ignore it)");
        }
        else
        {
            var (repoId, filename, approxMB) = s_models[Model];
            _modelPath = DownloadModel(repoId, filename, approxMB);
            _metricsKey = Model.ToString();
        }

        // Read optional prompt and max-tokens overrides from env vars
        // (set by bench_compare.py to ensure both engines use the same inputs)
        var envPrompt = Environment.GetEnvironmentVariable("DOTLLM_BENCH_PROMPT");
        if (!string.IsNullOrEmpty(envPrompt))
            _prompt = envPrompt;

        var envMaxTokens = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MAX_TOKENS");
        if (!string.IsNullOrEmpty(envMaxTokens) && int.TryParse(envMaxTokens, out var parsedTokens))
            _maxTokens = parsedTokens;

        var promptPreview = _prompt.Length > 60 ? _prompt[..60] + "..." : _prompt;
        Console.WriteLine($"Prompt: \"{promptPreview}\", MaxTokens: {_maxTokens}");

        _gguf = GgufFile.Open(_modelPath);
        var config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
        _tokenizer = GgufBpeTokenizerFactory.Load(_gguf.Metadata);

        var envDevice = Environment.GetEnvironmentVariable("DOTLLM_BENCH_DEVICE") ?? "cpu";
        Func<ModelConfig, int, IKvCache>? kvFactory = null;

        if (envDevice.StartsWith("gpu", StringComparison.OrdinalIgnoreCase))
        {
            int gpuId = 0;
            if (envDevice.Contains(':'))
                int.TryParse(envDevice.Split(':')[1], out gpuId);

            var cudaModel = CudaTransformerModel.LoadFromGguf(_gguf, config, gpuId);
            _model = cudaModel;
            kvFactory = (cfg, size) => cudaModel.CreateKvCache(size);

            var device = CudaDevice.GetDevice(gpuId);
            Console.WriteLine($"Device: GPU ({device})");
        }
        else
        {
            _model = TransformerModel.LoadFromGguf(_gguf, config, ThreadingConfig.Auto);
            Console.WriteLine($"Device: CPU ({ThreadingConfig.Auto.EffectiveThreadCount} threads)");
        }

        _generator = new TextGenerator(_model, _tokenizer, kvFactory);
    }

    [Benchmark(Description = "E2E inference (prefill + decode)")]
    public InferenceResponse Inference()
    {
        var options = new InferenceOptions
        {
            Temperature = 0f, // greedy
            MaxTokens = _maxTokens
        };

        var response = _generator.Generate(_prompt, options);
        _timings.Add(response.Timings);
        return response;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        // Write metrics for the custom IColumn bridge
        if (_timings.Count > 0)
        {
            var prefillTokPerSecAll = _timings.Select(t => t.PrefillTokensPerSec).ToArray();
            var decodeTokPerSecAll = _timings.Select(t => t.DecodeTokensPerSec).ToArray();
            var prefillMsAll = _timings.Select(t => t.PrefillTimeMs).ToArray();
            var decodeMsAll = _timings.Select(t => t.DecodeTimeMs).ToArray();

            var prefillTokPerSecSorted = prefillTokPerSecAll.OrderBy(v => v).ToList();
            var decodeTokPerSecSorted = decodeTokPerSecAll.OrderBy(v => v).ToList();
            var prefillMsSorted = prefillMsAll.OrderBy(v => v).ToList();
            var decodeMsSorted = decodeMsAll.OrderBy(v => v).ToList();

            var metrics = new InferenceMetricsFile(
                MedianPrefillTokPerSec: Median(prefillTokPerSecSorted),
                MedianDecodeTokPerSec: Median(decodeTokPerSecSorted),
                MedianPrefillMs: Median(prefillMsSorted),
                MedianDecodeMs: Median(decodeMsSorted),
                PrefillTokenCount: _timings[0].PrefillTokenCount,
                DecodeTokenCount: _timings[0].DecodeTokenCount,
                Iterations: _timings.Count,
                BestPrefillTokPerSec: prefillTokPerSecAll.Max(),
                BestDecodeTokPerSec: decodeTokPerSecAll.Max(),
                BestPrefillMs: prefillMsAll.Min(),
                BestDecodeMs: decodeMsAll.Min(),
                DecodeCv: Cv(decodeTokPerSecAll),
                PrefillCv: Cv(prefillTokPerSecAll),
                AllDecodeTokPerSec: decodeTokPerSecAll,
                AllPrefillTokPerSec: prefillTokPerSecAll,
                AllDecodeMs: decodeMsAll,
                AllPrefillMs: prefillMsAll);

            InferenceMetricsFile.Write(_metricsKey, metrics);
        }

        _model?.Dispose();
        _gguf?.Dispose();
    }

    private static double Median(List<double> sorted)
    {
        int n = sorted.Count;
        if (n == 0) return 0;
        if (n % 2 == 1) return sorted[n / 2];
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    private static double StdDev(double[] values)
    {
        if (values.Length < 2) return 0;
        double mean = values.Average();
        double sumSq = values.Sum(v => (v - mean) * (v - mean));
        return Math.Sqrt(sumSq / (values.Length - 1));
    }

    private static double Cv(double[] values)
    {
        if (values.Length < 2) return 0;
        double mean = values.Average();
        if (mean == 0) return 0;
        return StdDev(values) / mean;
    }

    private static string DownloadModel(string repoId, string filename, int approxMB)
    {
        string cacheDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache");

        string cachedPath = Path.Combine(cacheDir, repoId.Replace('/', Path.DirectorySeparatorChar), filename);

        if (File.Exists(cachedPath))
            return cachedPath;

        Console.WriteLine($"Downloading {repoId}/{filename} (~{approxMB} MB)...");
        using var downloader = new HuggingFaceDownloader();
        return downloader.DownloadFileAsync(repoId, filename, cacheDir).GetAwaiter().GetResult();
    }
}
