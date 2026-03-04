using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cli.Helpers;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Runs greedy text generation on a GGUF model: load → encode prompt → decode loop → stream tokens.
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

                ctx.Status($"Loading {config.Architecture} model ({config.NumLayers} layers, {config.HiddenSize} hidden)...");
                model = LlamaModel.LoadFromGguf(gguf, config);
            });
        loadSw.Stop();

        AnsiConsole.MarkupLine($"[grey]Model: {config.Architecture}, {config.NumLayers} layers, {config.HiddenSize} hidden, {config.VocabSize:N0} vocab[/]");
        AnsiConsole.MarkupLine("[grey]Warning: No KV-cache — full context reprocessed each step (slow)[/]");
        AnsiConsole.WriteLine();

        try
        {
            // Encode prompt
            int[] promptTokens = tokenizer.Encode(settings.Prompt);
            int promptLen = promptTokens.Length;

            // Build the full token sequence (prompt + generated)
            int maxSeqLen = promptLen + settings.MaxTokens;
            var tokens = new List<int>(maxSeqLen);
            tokens.AddRange(promptTokens);

            // Pre-allocate positions array for the entire generation run
            int[] positions = new int[maxSeqLen];
            for (int i = 0; i < maxSeqLen; i++)
                positions[i] = i;

            // Print prompt echo
            Console.Write(settings.Prompt);

            var totalSw = Stopwatch.StartNew();
            long promptEvalTicks = 0;
            long evalTicks = 0;
            long samplerTicks = 0;
            int generated = 0;

            for (int step = 0; step < settings.MaxTokens; step++)
            {
                int seqLen = tokens.Count;
                var tokenSpan = CollectionsMarshal.AsSpan(tokens);

                // Forward pass — returns [1, vocabSize] logits (last token only)
                long fwdStart = Stopwatch.GetTimestamp();
                using var logitsTensor = model.Forward(tokenSpan, positions.AsSpan(0, seqLen), -1);
                long fwdEnd = Stopwatch.GetTimestamp();

                if (step == 0)
                    promptEvalTicks = fwdEnd - fwdStart;
                else
                    evalTicks += fwdEnd - fwdStart;

                // Greedy argmax on logits
                int vocabSize = config.VocabSize;
                long samplerStart = Stopwatch.GetTimestamp();
                unsafe
                {
                    float* logits = (float*)logitsTensor.DataPointer;

                    int bestId = 0;
                    float bestVal = logits[0];
                    for (int i = 1; i < vocabSize; i++)
                    {
                        if (logits[i] > bestVal)
                        {
                            bestVal = logits[i];
                            bestId = i;
                        }
                    }
                    long samplerEnd = Stopwatch.GetTimestamp();
                    samplerTicks += samplerEnd - samplerStart;

                    // Check EOS
                    if (bestId == tokenizer.EosTokenId)
                        break;

                    tokens.Add(bestId);
                    generated++;

                    // Stream token text
                    string tokenText = tokenizer.DecodeToken(bestId);
                    Console.Write(tokenText);
                }
            }

            totalSw.Stop();
            Console.WriteLine();
            AnsiConsole.WriteLine();

            // Convert ticks to milliseconds
            double tickFreq = Stopwatch.Frequency;
            double loadMs = loadSw.Elapsed.TotalMilliseconds;
            double promptEvalMs = promptEvalTicks / tickFreq * 1000.0;
            double evalMs = evalTicks / tickFreq * 1000.0;
            double samplerMs = samplerTicks / tickFreq * 1000.0;
            double totalMs = totalSw.Elapsed.TotalMilliseconds;

            int evalSteps = generated > 0 ? generated - 1 : 0; // first generated token comes from prompt eval step

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
            long totalMemory = modelWeightsBytes + computeBytes;

            var memTable = new Table()
                .Border(TableBorder.Rounded)
                .Title("[bold]Memory Breakdown[/]");

            memTable.AddColumn(new TableColumn("Component").LeftAligned());
            memTable.AddColumn(new TableColumn("Size").RightAligned());

            memTable.AddRow("Model weights", $"{FormatHelpers.FormatMiB(modelWeightsBytes)}  [dim](memory-mapped)[/]");
            memTable.AddRow("Compute", FormatHelpers.FormatMiB(computeBytes));
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
