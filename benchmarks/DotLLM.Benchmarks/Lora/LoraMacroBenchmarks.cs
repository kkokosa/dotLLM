using BenchmarkDotNet.Attributes;
using DotLLM.Benchmarks.Columns;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Benchmarks.Lora;

/// <summary>LoRA adapter dtype variant exercised by the macro-bench.</summary>
public enum LoraVariant
{
    /// <summary>Baseline — no adapter active; identical to the pre-LoRA forward path.</summary>
    NoLora,
    /// <summary>F32 adapter — Phase 4a kernel path (no dequant).</summary>
    LoraF32,
    /// <summary>F16 adapter — Phase 4d.1 dequant-on-read path.</summary>
    LoraF16,
    /// <summary>BF16 adapter — Phase 4d.1 dequant-on-read path.</summary>
    LoraBF16,
    /// <summary>
    /// Q8_0 B + F16 A — Phase 4d.4 path. Closes the prefill regression
    /// that F32-on-Q8_0-base introduces (Agent 8 measured −36% on Strix
    /// Halo for LoraF32). Stage 1 uses GemmQ8_0; stage 2 dequants A.
    /// </summary>
    LoraQ8_0,
}

/// <summary>Macro-bench scenario — fixes (prompt-len, decode-len) workload pair.</summary>
public enum LoraScenario
{
    /// <summary>~512-token prefill chunk, single decode step. Measures bandwidth-amortised LoRA cost.</summary>
    Prefill512,
    /// <summary>~32-token prompt, 128 decode steps. Measures per-token LoRA cost at decode batch=1.</summary>
    Decode128,
}

/// <summary>
/// Phase 4d.3 macro-benchmark — answers: does the +9% kernel-level prefill
/// LoRA overhead translate to a measurable system-level slowdown, or is it
/// amortised by everything else the forward pass does (attention, FFN, KV-cache,
/// memory bandwidth)?
/// </summary>
/// <remarks>
/// <para>
/// Strategy: load a real GGUF checkpoint (TinyLlama 1.1B preferred; falls back
/// to Llama-3.2-1B then SmolLM-135M from the local test-cache), build a
/// deterministic synthetic LoRA adapter via <see cref="SyntheticLoraAdapter"/>,
/// then run two workloads — a ~512-token prefill and a 128-token decode loop —
/// with and without the adapter active. Per-iteration prefill / decode tok/s
/// are captured via <see cref="InferenceTimings"/> (the same source the existing
/// Step-13 benchmarks use) and written to the file-based metrics bridge for the
/// custom <see cref="PrefillTokPerSecColumn"/> / <see cref="DecodeTokPerSecColumn"/>
/// columns to display.
/// </para>
/// <para>
/// Acceptance gate: if no real checkpoint exists locally, every benchmark is
/// a fast no-op so the BDN suite stays green. We do NOT trigger downloads
/// here — the bench is measurement-only, not a fixture provisioner.
/// See <c>.continue-here-lora-macro-bench.md</c> for required fixture paths.
/// </para>
/// </remarks>
[SimpleJob(warmupCount: 1, iterationCount: 3)]
public class LoraMacroBenchmarks
{
    /// <summary>LoRA adapter dtype to apply during the forward pass.</summary>
    [ParamsAllValues]
    public LoraVariant Variant { get; set; }

    /// <summary>Workload shape — Prefill512 or Decode128.</summary>
    [ParamsAllValues]
    public LoraScenario Scenario { get; set; }

    // Fixed adapter geometry — typical PEFT settings.
    private const int LoraRank = 16;
    private const float LoraAlpha = 32f;
    private const int AdapterSeed = 0xD071A;

    // Fixed scenario shapes — keep stable across runs so deltas are comparable.
    // Prefill: ~512 tokens of greedy decode budget so prompt token count dominates.
    private const int PrefillPromptTokens = 512;
    private const int PrefillDecodeTokens = 1;
    // Decode: small prompt so most of the wall-clock is per-step decode forward.
    private const int DecodePromptTokens = 32;
    private const int DecodeStepTokens = 128;

    private IModel _model = null!;
    private ITokenizer _tokenizer = null!;
    private GgufFile _gguf = null!;
    private TextGenerator _generator = null!;
    private DotLLM.Core.Lora.LoraAdapter? _adapterF32;
    private DotLLM.Core.Lora.LoraAdapter? _adapterF16;
    private DotLLM.Core.Lora.LoraAdapter? _adapterBF16;
    private DotLLM.Core.Lora.LoraAdapter? _adapterQ8_0;
    private string _prompt = string.Empty;
    private string _modelLabel = string.Empty;
    private bool _skipped;
    private string? _skipReason;

    private readonly List<InferenceTimings> _timings = new();

    [GlobalSetup]
    public void Setup()
    {
        string? modelPath = LoraMacroBenchFixture.ResolveCheckpoint(out string label, out _skipReason);
        if (modelPath is null)
        {
            _skipped = true;
            Console.WriteLine($"[LoraMacroBenchmarks] SKIP: {_skipReason}");
            return;
        }

        _modelLabel = label;
        Console.WriteLine($"[LoraMacroBenchmarks] model: {label}  path: {modelPath}");

        _gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
        _tokenizer = GgufBpeTokenizerFactory.Load(_gguf.Metadata);
        _model = TransformerModel.LoadFromGguf(_gguf, config, ThreadingConfig.Auto);
        _generator = new TextGenerator(_model, _tokenizer);

        // Build a prompt long enough that BPE-encoding produces >= 512 tokens for
        // the prefill scenario. The body is meaningless — we are measuring
        // forward-pass throughput, not output quality.
        _prompt = BuildLongPrompt(_tokenizer, targetTokens: PrefillPromptTokens + 16);

        // Build all adapter variants up-front so the per-benchmark hot path
        // contains only the Forward calls.
        _adapterF32 = SyntheticLoraAdapter.Create("synth-f32", config, LoraRank, LoraAlpha, LoraWeightDType.F32, AdapterSeed);
        _adapterF16 = SyntheticLoraAdapter.Create("synth-f16", config, LoraRank, LoraAlpha, LoraWeightDType.F16, AdapterSeed);
        _adapterBF16 = SyntheticLoraAdapter.Create("synth-bf16", config, LoraRank, LoraAlpha, LoraWeightDType.BF16, AdapterSeed);
        // Phase 4d.4: Q8_0 B + F16 A — the new path. Same RNG seed so the
        // adapter values are identical to the F16 case modulo Q8_0 quantisation.
        _adapterQ8_0 = SyntheticLoraAdapter.CreateQ8_0B("synth-q8_0", config, LoraRank, LoraAlpha, AdapterSeed);

        Console.WriteLine(
            $"[LoraMacroBenchmarks] config: hidden={config.HiddenSize} layers={config.NumLayers} "
            + $"heads={config.NumAttentionHeads} kv_heads={config.NumKvHeads} ffn={config.IntermediateSize} "
            + $"adapter_layer_count={_adapterF32.LayerWeights.Count}");
    }

    [Benchmark]
    public InferenceResponse Run()
    {
        if (_skipped)
            return CreateSkipResponse();

        var adapter = SelectAdapter(Variant);
        var (promptTok, decodeTok) = SelectShape(Scenario);

        // The prompt is encoded inside TextGenerator; we trim our too-long
        // synthetic prompt down to roughly the desired token budget so the
        // BDN per-iteration cost reflects the targeted shape.
        string scopedPrompt = TruncateToTokens(_prompt, _tokenizer, promptTok);

        var options = new InferenceOptions
        {
            Temperature = 0f, // greedy — no sampling variance across iterations
            MaxTokens = decodeTok,
        };

        var response = _generator.Generate(scopedPrompt, options, adapter: adapter);
        _timings.Add(response.Timings);
        return response;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (!_skipped && _timings.Count > 0)
        {
            WriteMetrics();
        }

        _adapterF32?.Dispose();
        _adapterF16?.Dispose();
        _adapterBF16?.Dispose();
        _adapterQ8_0?.Dispose();
        _model?.Dispose();
        _gguf?.Dispose();
    }

    private ILoraAdapter? SelectAdapter(LoraVariant v) => v switch
    {
        LoraVariant.NoLora => null,
        LoraVariant.LoraF32 => _adapterF32,
        LoraVariant.LoraF16 => _adapterF16,
        LoraVariant.LoraBF16 => _adapterBF16,
        LoraVariant.LoraQ8_0 => _adapterQ8_0,
        _ => null,
    };

    private static (int PromptTok, int DecodeTok) SelectShape(LoraScenario s) => s switch
    {
        LoraScenario.Prefill512 => (PrefillPromptTokens, PrefillDecodeTokens),
        LoraScenario.Decode128 => (DecodePromptTokens, DecodeStepTokens),
        _ => (PrefillPromptTokens, PrefillDecodeTokens),
    };

    private void WriteMetrics()
    {
        var prefillTokPerSecAll = _timings.Select(t => t.PrefillTokensPerSec).ToArray();
        var decodeTokPerSecAll = _timings.Select(t => t.DecodeTokensPerSec).ToArray();
        var prefillMsAll = _timings.Select(t => t.PrefillTimeMs).ToArray();
        var decodeMsAll = _timings.Select(t => t.DecodeTimeMs).ToArray();

        // Best-of-N (max for throughput, min for latency) — same convention as
        // InferenceBenchmarks. The median is retained for back-compat with the
        // older metrics consumers; bench_compare.py and the BDN columns prefer
        // best-of-N.
        var metrics = new InferenceMetricsFile(
            MedianPrefillTokPerSec: Median(prefillTokPerSecAll),
            MedianDecodeTokPerSec: Median(decodeTokPerSecAll),
            MedianPrefillMs: Median(prefillMsAll),
            MedianDecodeMs: Median(decodeMsAll),
            PrefillTokenCount: _timings[0].PrefillTokenCount,
            DecodeTokenCount: _timings[0].DecodeTokenCount,
            Iterations: _timings.Count,
            BestPrefillTokPerSec: prefillTokPerSecAll.Length > 0 ? prefillTokPerSecAll.Max() : 0,
            BestDecodeTokPerSec: decodeTokPerSecAll.Length > 0 ? decodeTokPerSecAll.Max() : 0,
            BestPrefillMs: prefillMsAll.Length > 0 ? prefillMsAll.Min() : 0,
            BestDecodeMs: decodeMsAll.Length > 0 ? decodeMsAll.Min() : 0,
            AllPrefillTokPerSec: prefillTokPerSecAll,
            AllDecodeTokPerSec: decodeTokPerSecAll,
            AllPrefillMs: prefillMsAll,
            AllDecodeMs: decodeMsAll);

        string key = MetricsKey(_modelLabel, Variant, Scenario);
        InferenceMetricsFile.Write(key, metrics);
    }

    /// <summary>
    /// Composite metrics key — model-label, variant, scenario. The custom
    /// BDN columns recover the matching file from the on-disk metrics dir
    /// via <c>ColumnHelpers.TryGetMetricsKey</c>, which probes for the
    /// expected (variant, scenario) suffix.
    /// </summary>
    internal static string MetricsKey(string modelLabel, LoraVariant v, LoraScenario s)
        => $"Lora_{modelLabel}_{v}_{s}";

    private static double Median(double[] xs)
    {
        if (xs.Length == 0) return 0;
        var sorted = xs.OrderBy(v => v).ToArray();
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    /// <summary>
    /// Generates a long, deterministic prompt. We assemble a paragraph that is
    /// long in characters, then rely on the caller's <see cref="TruncateToTokens"/>
    /// to slice it down per scenario. The body is intentionally meaningless —
    /// macro-bench measures throughput, not output quality.
    /// </summary>
    private static string BuildLongPrompt(ITokenizer tok, int targetTokens)
    {
        // A neutral filler paragraph; repeat until BPE-encoded length >= target.
        const string filler =
            "The history of computing is a story of progressive abstraction. " +
            "From punched cards to transistors, integrated circuits to multicore, " +
            "the machines have grown faster while the programs have grown larger. " +
            "Each generation of hardware enabled a new class of software. ";

        var sb = new System.Text.StringBuilder(filler.Length * 8);
        while (true)
        {
            sb.Append(filler);
            int[] enc = tok.Encode(sb.ToString());
            if (enc.Length >= targetTokens) break;
            if (sb.Length > 1 << 20) break; // safety cap (1 MB of prompt)
        }
        return sb.ToString();
    }

    /// <summary>
    /// Re-encodes <paramref name="prompt"/> and trims to roughly
    /// <paramref name="targetTokens"/> tokens. We decode after slicing to keep
    /// the TextGenerator path identical to a "user-supplied prompt" — the
    /// alternative would be to extend the IModel surface with a bypass, which
    /// is out of scope for a measurement-only benchmark.
    /// </summary>
    private static string TruncateToTokens(string prompt, ITokenizer tok, int targetTokens)
    {
        int[] enc = tok.Encode(prompt);
        if (enc.Length <= targetTokens) return prompt;
        return tok.Decode(enc.AsSpan(0, targetTokens).ToArray());
    }

    /// <summary>
    /// When the macro-bench cannot find a real checkpoint, every benchmark case
    /// returns this empty response immediately. The BDN suite stays green and
    /// the columns surface "N/A" rather than nonsense numbers.
    /// </summary>
    private static InferenceResponse CreateSkipResponse() => new()
    {
        GeneratedTokenIds = [],
        Text = string.Empty,
        FinishReason = FinishReason.Length,
        PromptTokenCount = 0,
        GeneratedTokenCount = 0,
    };
}
