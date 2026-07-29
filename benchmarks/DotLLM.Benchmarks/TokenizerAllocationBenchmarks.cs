using System.Text;
using BenchmarkDotNet.Attributes;
using DotLLM.HuggingFace;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks;

/// <summary>
/// Compares the engine's BPE encoder against <see cref="NaiveBpeTokenizer"/>, an
/// implementation of the same algorithm written with the constructs a C# developer
/// reaches for first: class-based linked-list nodes, string-keyed dictionaries, string
/// concatenation for lookup keys, and <c>new</c> instead of <c>ArrayPool</c>.
/// </summary>
/// <remarks>
/// <para>
/// The engine implementation is the baseline, so the naive row reads directly as
/// "N times slower" and "N times the allocation".
/// </para>
/// <para>
/// Both tokenizers are built from the real SmolLM-135M vocabulary (49k entries,
/// <c>tokenizer.ggml.model = "gpt2"</c>) — the tokenizer type used by every model in the
/// standard benchmark set. The model is auto-downloaded on first run and cached.
/// </para>
/// <para>
/// <see cref="Setup"/> asserts the two implementations produce identical token ID
/// sequences, so a divergence fails the run rather than skewing the comparison.
/// Tokenizers are constructed with no token types, which puts <see cref="BpeTokenizer"/>
/// on its no-special-tokens fast path: both sides then do exactly the BPE work and
/// nothing else. Special-token pre-splitting is covered separately by
/// <see cref="SpecialTokenEncodeBenchmarks"/>.
/// </para>
/// </remarks>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class TokenizerAllocationBenchmarks
{
    private const string RepoId = "QuantFactory/SmolLM-135M-GGUF";
    private const string Filename = "SmolLM-135M.Q8_0.gguf";

    private BpeTokenizer _engine = null!;
    private NaiveBpeTokenizer _naive = null!;
    private string _text = null!;

    /// <summary>Number of characters of prose to encode.</summary>
    [Params(1024, 8192, 32768)]
    public int TextLength { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        string modelPath = ResolveModelPath();

        string[] tokens;
        string[] merges;
        string? preType;
        using (var gguf = GgufFile.Open(modelPath))
        {
            tokens = gguf.Metadata.GetStringArray("tokenizer.ggml.tokens");
            merges = gguf.Metadata.ContainsKey("tokenizer.ggml.merges")
                ? gguf.Metadata.GetStringArray("tokenizer.ggml.merges")
                : [];
            string pre = gguf.Metadata.GetStringOrDefault("tokenizer.ggml.pre");
            preType = pre.Length > 0 ? pre : null;
        }

        // tokenTypes: null → no special tokens → BpeTokenizer.Encode takes its fast path
        // straight into the BPE encoding, which is what this benchmark is about.
        _engine = BpeTokenizer.CreateTiktoken(tokens, merges, tokenTypes: null,
            bosId: 0, eosId: 0, preTokenizerType: preType);
        _naive = new NaiveBpeTokenizer(tokens, merges, NaiveBpeTokenizer.GetPreRegex(preType));

        _text = BuildText(TextLength);

        int[] expected = _engine.Encode(_text);
        int[] actual = _naive.Encode(_text);
        if (!expected.AsSpan().SequenceEqual(actual))
        {
            throw new InvalidOperationException(
                $"Naive baseline diverges from the engine: expected {expected.Length} tokens, " +
                $"got {actual.Length}. The comparison is only meaningful when both produce " +
                "identical output.");
        }
    }

    [Benchmark(Baseline = true, Description = "dotLLM (pooled struct symbols)")]
    public int Engine() => _engine.Encode(_text).Length;

    [Benchmark(Description = "Naive (class nodes, string keys, no pooling)")]
    public int Naive() => _naive.Encode(_text).Length;

    /// <summary>
    /// Locates the SmolLM-135M GGUF, checking the test cache and the CLI model cache
    /// before downloading. Matches <see cref="GgufRealModelBenchmarks"/>.
    /// </summary>
    private static string ResolveModelPath()
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string relative = Path.Combine(RepoId.Replace('/', Path.DirectorySeparatorChar), Filename);

        string testCacheDir = Environment.GetEnvironmentVariable("DOTLLM_TEST_CACHE_DIR")
            ?? Path.Combine(home, ".dotllm", "test-cache");

        foreach (string root in new[] { testCacheDir, Path.Combine(home, ".dotllm", "models") })
        {
            string candidate = Path.Combine(root, relative);
            if (File.Exists(candidate)) return candidate;
        }

        Console.WriteLine($"Downloading {RepoId}/{Filename} (~145 MB)...");
        using var downloader = new HuggingFaceDownloader();
        return downloader.DownloadFileAsync(RepoId, Filename, testCacheDir).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Builds <paramref name="length"/> characters of real prose from
    /// <c>prompt-large.txt</c>, repeating it as needed. Repetition is safe: BPE has no
    /// cross-segment state, so each repeat costs the same as the first.
    /// </summary>
    private static string BuildText(int length)
    {
        string source = LoadPromptText();
        var sb = new StringBuilder(length);
        while (sb.Length < length)
            sb.Append(source, 0, Math.Min(source.Length, length - sb.Length));
        return sb.ToString();
    }

    /// <summary>
    /// Reads <c>prompt-large.txt</c> by walking up from the assembly location to the
    /// repository root. Falls back to inline prose when the repository is not on disk
    /// (for example a published or packaged run).
    /// </summary>
    private static string LoadPromptText()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "prompt-large.txt");
            if (File.Exists(candidate)) return File.ReadAllText(candidate);
        }

        Console.WriteLine("prompt-large.txt not found; using inline fallback prose.");
        return "Large language models have become a cornerstone of modern artificial "
             + "intelligence, demonstrating capabilities in text generation, reasoning, "
             + "and code synthesis. The journey began with early neural language models "
             + "that used simple recurrent architectures. ";
    }
}
