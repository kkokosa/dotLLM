using System.ComponentModel;
using System.Diagnostics;
using DotLLM.Cli.Helpers;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Runs text generation on a GGUF model: load → encode prompt → stream tokens via TextGenerator.
/// Supports greedy (default) and sampled decoding via composable sampling pipeline.
/// </summary>
internal sealed class RunCommand : Command<RunCommand.Settings>
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
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        if (string.IsNullOrEmpty(settings.Prompt))
        {
            AnsiConsole.MarkupLine("[red]--prompt|-p is required.[/]");
            return 1;
        }

        var resolvedPath = GgufFileResolver.Resolve(settings.Model);
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
        AnsiConsole.MarkupLine($"[grey]Model: {config.Architecture}, {config.NumLayers} layers, {config.HiddenSize} hidden, {config.VocabSize:N0} vocab, {threadingInfo.EffectiveThreadCount} thread(s)[/]");

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

        if (settings.Temperature > 0)
        {
            var parts = new List<string> { $"temp={settings.Temperature:F2}" };
            if (settings.TopK > 0) parts.Add($"top-k={settings.TopK}");
            if (settings.TopP < 1.0f) parts.Add($"top-p={settings.TopP:F2}");
            if (settings.MinP > 0f) parts.Add($"min-p={settings.MinP:F2}");
            if (settings.RepeatPenalty != 1.0f) parts.Add($"repeat-penalty={settings.RepeatPenalty:F2}");
            if (settings.Seed.HasValue) parts.Add($"seed={settings.Seed.Value}");
            AnsiConsole.MarkupLine($"[grey]Sampling: {string.Join(", ", parts)}[/]");
        }
        else
        {
            AnsiConsole.MarkupLine("[grey]Sampling: greedy[/]");
        }

        AnsiConsole.WriteLine();

        try
        {
            // Print prompt echo
            Console.Write(settings.Prompt);

            var generator = new TextGenerator(model, tokenizer);
            var totalSw = Stopwatch.StartNew();

            var response = generator.Generate(settings.Prompt, inferenceOptions,
                onTokenGenerated: tokenId => Console.Write(tokenizer.DecodeToken(tokenId)));

            totalSw.Stop();
            Console.WriteLine();
            AnsiConsole.WriteLine();

            // Read timings from engine response
            var timings = response.Timings;
            double loadMs = loadSw.Elapsed.TotalMilliseconds;
            double promptEvalMs = timings.PrefillTimeMs;
            double evalMs = timings.DecodeTimeMs;
            double samplerMs = timings.SamplingTimeMs;
            double totalMs = totalSw.Elapsed.TotalMilliseconds;
            int promptLen = timings.PrefillTokenCount;
            int generated = response.GeneratedTokenCount;
            int evalSteps = timings.DecodeTokenCount;

            // Performance summary table
            var perfTable = new Table()
                .Border(TableBorder.Rounded)
                .Title("[bold]Performance Summary[/]");

            perfTable.AddColumn(new TableColumn("Phase").LeftAligned());
            perfTable.AddColumn(new TableColumn("Time").RightAligned());
            perfTable.AddColumn(new TableColumn("Tokens").RightAligned());
            perfTable.AddColumn(new TableColumn("ms/token").RightAligned());
            perfTable.AddColumn(new TableColumn("tokens/s").RightAligned());

            // Load
            perfTable.AddRow(
                "Load",
                $"{loadMs:F2} ms",
                Markup.Escape("—"),
                Markup.Escape("—"),
                Markup.Escape("—"));

            // Prompt eval
            if (promptLen > 0)
            {
                double promptMsPerToken = promptEvalMs / promptLen;
                double promptTokPerSec = promptLen / (promptEvalMs / 1000.0);
                perfTable.AddRow(
                    "Prompt eval",
                    $"{promptEvalMs:F2} ms",
                    promptLen.ToString(),
                    $"{promptMsPerToken:F2}",
                    $"{promptTokPerSec:F2}");
            }

            // Eval (decode steps after the first)
            if (evalSteps > 0)
            {
                double evalMsPerToken = evalMs / evalSteps;
                double evalTokPerSec = evalSteps / (evalMs / 1000.0);
                perfTable.AddRow(
                    "Eval",
                    $"{evalMs:F2} ms",
                    $"{evalSteps}",
                    $"{evalMsPerToken:F2}",
                    $"{evalTokPerSec:F2}");
            }
            else
            {
                perfTable.AddRow(
                    "Eval",
                    $"{evalMs:F2} ms",
                    "0",
                    Markup.Escape("—"),
                    Markup.Escape("—"));
            }

            // Sampling
            if (generated > 0)
            {
                double samplerMsPerToken = samplerMs / generated;
                perfTable.AddRow(
                    "Sampling",
                    $"{samplerMs:F2} ms",
                    generated.ToString(),
                    $"{samplerMsPerToken:F2}",
                    Markup.Escape("—"));
            }
            else
            {
                perfTable.AddRow(
                    "Sampling",
                    $"{samplerMs:F2} ms",
                    "0",
                    Markup.Escape("—"),
                    Markup.Escape("—"));
            }

            // Total
            int totalTokens = promptLen + generated;
            double totalTokPerSec = totalTokens > 0 ? totalTokens / (totalMs / 1000.0) : 0;
            perfTable.AddRow(
                "[bold]Total[/]",
                $"[bold]{totalMs:F2} ms[/]",
                $"[bold]{totalTokens}[/]",
                Markup.Escape("—"),
                $"[bold]{totalTokPerSec:F2}[/]");

            AnsiConsole.Write(perfTable);
            AnsiConsole.WriteLine();

            // Memory breakdown table
            long fileSize = new FileInfo(resolvedPath).Length;
            long modelWeightsBytes = fileSize - gguf.DataSectionOffset;
            long computeBytes = model.ComputeMemoryBytes;

            int cacheSize = Math.Min(promptLen + settings.MaxTokens, config.MaxSequenceLength);
            long kvCacheBytes = (long)config.NumLayers * 2 * cacheSize
                * config.NumKvHeads * config.HeadDim * sizeof(float);
            long totalMemory = modelWeightsBytes + computeBytes + kvCacheBytes;

            var memTable = new Table()
                .Border(TableBorder.Rounded)
                .Title("[bold]Memory Breakdown[/]");

            memTable.AddColumn(new TableColumn("Component").LeftAligned());
            memTable.AddColumn(new TableColumn("Size").RightAligned());

            memTable.AddRow("Model weights", $"{FormatHelpers.FormatMiB(modelWeightsBytes)}  [dim](memory-mapped)[/]");
            memTable.AddRow("Compute", FormatHelpers.FormatMiB(computeBytes));
            memTable.AddRow("KV-cache", $"{FormatHelpers.FormatMiB(kvCacheBytes)}  [dim]({cacheSize} slots)[/]");
            memTable.AddRow("[bold]Total[/]", $"[bold]{FormatHelpers.FormatMiB(totalMemory)}[/]");

            AnsiConsole.Write(memTable);
        }
        finally
        {
            model.Dispose();
            gguf.Dispose();
        }

        return 0;
    }
}
