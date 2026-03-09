using System.ComponentModel;
using System.Diagnostics;
using System.Text.RegularExpressions;
using DotLLM.Cli.Helpers;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;
using Spectre.Console.Rendering;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Runs text generation on a GGUF model: load → encode prompt → stream tokens via TextGenerator.
/// Supports greedy (default) and sampled decoding via composable sampling pipeline.
/// </summary>
internal sealed class RunCommand : AsyncCommand<RunCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF).")]
        public string Model { get; set; } = string.Empty;

        [CommandOption("--prompt|-p")]
        [Description("Input prompt for generation (required).")]
        public string? Prompt { get; set; }

        [CommandOption("--max-tokens|-n")]
        [Description("Maximum number of tokens to generate.")]
        [DefaultValue(128)]
        public int MaxTokens { get; set; } = 128;

        [CommandOption("--temp|-t")]
        [Description("Sampling temperature. 0 = greedy (default).")]
        [DefaultValue(0f)]
        public float Temperature { get; set; }

        [CommandOption("--top-k")]
        [Description("Top-K sampling. 0 = disabled.")]
        [DefaultValue(0)]
        public int TopK { get; set; }

        [CommandOption("--top-p")]
        [Description("Top-P (nucleus) sampling threshold.")]
        [DefaultValue(1.0f)]
        public float TopP { get; set; } = 1.0f;

        [CommandOption("--min-p")]
        [Description("Min-P sampling threshold. 0 = disabled.")]
        [DefaultValue(0f)]
        public float MinP { get; set; }

        [CommandOption("--repeat-penalty")]
        [Description("Repetition penalty factor. 1.0 = disabled.")]
        [DefaultValue(1.0f)]
        public float RepeatPenalty { get; set; } = 1.0f;

        [CommandOption("--repeat-last-n")]
        [Description("Number of recent tokens for repetition penalty lookback. 0 = full history.")]
        [DefaultValue(0)]
        public int RepeatLastN { get; set; }

        [CommandOption("--seed|-s")]
        [Description("Random seed for reproducible sampling. Omit for non-deterministic.")]
        public int? Seed { get; set; }

        [CommandOption("--threads")]
        [Description("Number of CPU threads for inference. 0 = auto/all cores (default), 1 = single-threaded.")]
        [DefaultValue(0)]
        public int Threads { get; set; }

        [CommandOption("--quant|-q")]
        [Description("Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0).")]
        public string? Quant { get; set; }
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        if (string.IsNullOrEmpty(settings.Prompt))
        {
            AnsiConsole.MarkupLine("[red]--prompt|-p is required.[/]");
            return 1;
        }

        var resolvedPath = GgufFileResolver.Resolve(settings.Model, settings.Quant);
        if (resolvedPath is null)
            return 1;

        GgufFile gguf = null!;
        ModelConfig config = null!;
        Tokenizers.Bpe.BpeTokenizer tokenizer = null!;
        LlamaModel model = null!;

        var loadSw = Stopwatch.StartNew();
        AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start("Loading model...", ctx =>
            {
                ctx.Status("Opening GGUF file...");
                gguf = GgufFile.Open(resolvedPath);

                ctx.Status("Extracting model config...");
                config = GgufModelConfigExtractor.Extract(gguf.Metadata);

                ctx.Status("Loading tokenizer...");
                tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

                var threading = new ThreadingConfig(settings.Threads);
                ctx.Status($"Loading {config.Architecture} model ({config.NumLayers} layers, {config.HiddenSize} hidden, {threading.EffectiveThreadCount} threads)...");
                model = LlamaModel.LoadFromGguf(gguf, config, threading);
            });
        loadSw.Stop();

        var threadingInfo = new ThreadingConfig(settings.Threads);

        // Build inference options from CLI flags
        var inferenceOptions = new InferenceOptions
        {
            Temperature = settings.Temperature,
            TopK = settings.TopK,
            TopP = settings.TopP,
            MinP = settings.MinP,
            RepetitionPenalty = settings.RepeatPenalty,
            RepetitionPenaltyWindow = settings.RepeatLastN,
            MaxTokens = settings.MaxTokens,
            Seed = settings.Seed,
            Threading = new ThreadingConfig(settings.Threads)
        };

        // Build compact pre-gen header rule
        var quantLabel = InferQuantLabel(resolvedPath, settings.Quant);
        var samplingLabel = BuildSamplingLabel(settings);
        var segments = $"{config.Architecture} {config.NumLayers}L/{config.HiddenSize}H | {quantLabel} | {threadingInfo.EffectiveThreadCount} threads | {samplingLabel}";
        AnsiConsole.Write(new Rule($"[grey]dotllm | {Markup.Escape(segments)}[/]").LeftJustified());
        AnsiConsole.WriteLine();

        try
        {
            // Print prompt echo
            Console.Write(settings.Prompt);

            var generator = new TextGenerator(model, tokenizer);
            var totalSw = Stopwatch.StartNew();
            int generated = 0;
            InferenceTimings timings = default;
            FinishReason finishReason = FinishReason.Length;

            await foreach (var token in generator.GenerateStreamingTokensAsync(settings.Prompt, inferenceOptions))
            {
                Console.Write(token.Text);
                if (token.FinishReason is null || token.Text.Length > 0)
                    generated++;
                if (token.FinishReason.HasValue)
                {
                    finishReason = token.FinishReason.Value;
                    timings = token.Timings ?? default;
                }
            }

            totalSw.Stop();
            Console.WriteLine();
            AnsiConsole.WriteLine();

            // Read timings from streaming result
            double loadMs = loadSw.Elapsed.TotalMilliseconds;
            double promptEvalMs = timings.PrefillTimeMs;
            double evalMs = timings.DecodeTimeMs;
            double samplerMs = timings.SamplingTimeMs;
            double totalMs = totalSw.Elapsed.TotalMilliseconds;
            int promptLen = timings.PrefillTokenCount;
            int evalSteps = timings.DecodeTokenCount;

            // Compute metrics
            int totalTokens = promptLen + generated;
            double decodeTokPerSec = evalSteps > 0 ? evalSteps / (evalMs / 1000.0) : 0;
            double prefillTokPerSec = promptLen > 0 ? promptLen / (promptEvalMs / 1000.0) : 0;
            double totalTokPerSec = totalTokens > 0 ? totalTokens / (totalMs / 1000.0) : 0;

            // Memory metrics
            long fileSize = new FileInfo(resolvedPath).Length;
            long modelWeightsBytes = fileSize - gguf.DataSectionOffset;
            long computeBytes = model.ComputeMemoryBytes;
            int cacheSize = Math.Min(promptLen + settings.MaxTokens, config.MaxSequenceLength);
            long kvCacheBytes = (long)config.NumLayers * 2 * cacheSize
                * config.NumKvHeads * config.HeadDim * sizeof(float);
            long totalMemory = modelWeightsBytes + computeBytes + kvCacheBytes;

            // Header grid: title left, hero metric right
            var headerGrid = new Grid();
            headerGrid.AddColumn(new GridColumn().NoWrap());
            headerGrid.AddColumn(new GridColumn().NoWrap().RightAligned());
            headerGrid.AddRow(
                new Markup("[bold]Generation Complete[/]"),
                new Markup($"[bold green]{decodeTokPerSec:F2} tok/s[/]"));

            // Build body lines
            var bodyLines = new List<IRenderable>();
            bodyLines.Add(new Markup("  [bold]Performance[/]"));
            bodyLines.Add(new Markup(PerfLine("Prefill", promptEvalMs, promptLen, prefillTokPerSec)));
            bodyLines.Add(new Markup(PerfLine("Decode", evalMs, evalSteps, decodeTokPerSec)));
            bodyLines.Add(new Markup(PerfLine("Sampling", samplerMs, generated, null)));
            bodyLines.Add(new Markup("  [dim]──────────────────────────────────────────────────────[/]"));
            bodyLines.Add(new Markup(PerfLine("Total", totalMs, totalTokens, totalTokPerSec)));
            bodyLines.Add(new Markup(PerfLine("Load", loadMs, null, null)));
            bodyLines.Add(new Text(""));
            bodyLines.Add(new Markup("  [bold]Memory[/]"));
            bodyLines.Add(new Markup(MemLine("Weights", modelWeightsBytes, "(memory-mapped)")));
            bodyLines.Add(new Markup(MemLine("Compute", computeBytes, null)));
            bodyLines.Add(new Markup(MemLine("KV Cache", kvCacheBytes, $"({cacheSize} slots)")));
            bodyLines.Add(new Markup("  [dim]──────────────────────────────────────────────────────[/]"));
            bodyLines.Add(new Markup(MemLine("Total", totalMemory, null)));
            bodyLines.Add(new Text(""));

            var finishReasonStr = finishReason.ToString().ToLowerInvariant();
            bodyLines.Add(new Markup($"  [dim]{Markup.Escape(finishReasonStr)} | {promptLen} prompt, {generated} generated[/]"));

            // Assemble panel
            var panelContent = new Rows(
                new Text(""),
                headerGrid,
                new Text(""),
                new Rows(bodyLines),
                new Text(""));

            var panel = new Panel(panelContent)
                .Border(BoxBorder.Rounded)
                .Padding(2, 0);

            AnsiConsole.Write(panel);
        }
        finally
        {
            model.Dispose();
            gguf.Dispose();
        }

        return 0;
    }

    private static string PerfLine(string label, double ms, int? tokens, double? tokPerSec)
    {
        var labelPart = $"[dim]{label,-14}[/]";
        var msPart = $"{ms,10:N1} ms";
        var tokensPart = tokens.HasValue ? $"{tokens.Value,6:N0} tokens" : "              ";
        var toksPart = tokPerSec.HasValue ? $"{tokPerSec.Value,10:F2} tok/s" : "";
        return $"  {labelPart} {msPart}   {tokensPart}   {toksPart}";
    }

    private static string MemLine(string label, long bytes, string? annotation)
    {
        var labelPart = $"[dim]{label,-14}[/]";
        var sizePart = $"{FormatHelpers.FormatMiB(bytes),12}";
        var annPart = annotation != null ? $"   [dim]{Markup.Escape(annotation)}[/]" : "";
        return $"  {labelPart} {sizePart}{annPart}";
    }

    private static string InferQuantLabel(string resolvedPath, string? quantFlag)
    {
        if (!string.IsNullOrEmpty(quantFlag))
            return quantFlag;

        var match = Regex.Match(Path.GetFileName(resolvedPath), @"\.(Q[\w]+)\.gguf$", RegexOptions.IgnoreCase);
        return match.Success ? match.Groups[1].Value : "unknown";
    }

    private static string BuildSamplingLabel(Settings settings)
    {
        if (settings.Temperature <= 0)
            return "greedy";

        var parts = new List<string> { $"temp={settings.Temperature:F1}" };
        if (settings.TopK > 0) parts.Add($"top-k={settings.TopK}");
        if (settings.TopP < 1.0f) parts.Add($"top-p={settings.TopP:F2}");
        if (settings.MinP > 0f) parts.Add($"min-p={settings.MinP:F2}");
        if (settings.RepeatPenalty != 1.0f) parts.Add($"rep={settings.RepeatPenalty:F2}");
        if (settings.Seed.HasValue) parts.Add($"seed={settings.Seed.Value}");
        return string.Join(", ", parts);
    }
}
