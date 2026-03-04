using System.ComponentModel;
using System.Diagnostics;
using DotLLM.Cli.Helpers;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
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
        AnsiConsole.WriteLine();

        try
        {
            // Encode prompt
            int[] promptTokens = tokenizer.Encode(settings.Prompt);
            int promptLen = promptTokens.Length;

            // Pre-allocate positions array and KV-cache (capped at model's max sequence length)
            int cacheSize = Math.Min(promptLen + settings.MaxTokens, config.MaxSequenceLength);
            int[] positions = new int[cacheSize];
            for (int i = 0; i < cacheSize; i++)
                positions[i] = i;

            using var kvCache = new SimpleKvCache(
                config.NumLayers, config.NumKvHeads, config.HeadDim, cacheSize);

            // Print prompt echo
            Console.Write(settings.Prompt);

            var totalSw = Stopwatch.StartNew();
            long promptEvalTicks = 0;
            long evalTicks = 0;
            long samplerTicks = 0;
            int generated = 0;
            int vocabSize = config.VocabSize;

            // Prefill: process all prompt tokens at once
            long fwdStart = Stopwatch.GetTimestamp();
            using var prefillLogits = model.Forward(
                promptTokens, positions.AsSpan(0, promptLen), -1, kvCache);
            long fwdEnd = Stopwatch.GetTimestamp();
            promptEvalTicks = fwdEnd - fwdStart;

            // Generate tokens only when max-tokens > 0
            if (settings.MaxTokens > 0)
            {
                // First generated token from prefill logits
                int lastToken;
                unsafe
                {
                    long samplerStart = Stopwatch.GetTimestamp();
                    float* logitPtr = (float*)prefillLogits.DataPointer;
                    lastToken = ArgMax(logitPtr, vocabSize);
                    samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                }

                if (lastToken != tokenizer.EosTokenId)
                {
                    generated++;
                    Console.Write(tokenizer.DecodeToken(lastToken));

                    // Decode loop: single token per step
                    for (int step = 1; step < settings.MaxTokens; step++)
                    {
                        int pos = promptLen + step - 1;
                        fwdStart = Stopwatch.GetTimestamp();
                        using var logitsTensor = model.Forward(
                            [lastToken], positions.AsSpan(pos, 1), -1, kvCache);
                        fwdEnd = Stopwatch.GetTimestamp();
                        evalTicks += fwdEnd - fwdStart;

                        unsafe
                        {
                            long samplerStart = Stopwatch.GetTimestamp();
                            float* logitPtr = (float*)logitsTensor.DataPointer;
                            lastToken = ArgMax(logitPtr, vocabSize);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }

                        if (lastToken == tokenizer.EosTokenId)
                            break;

                        generated++;
                        Console.Write(tokenizer.DecodeToken(lastToken));
                    }
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
            long kvCacheBytes = kvCache.AllocatedBytes;
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

    private static unsafe int ArgMax(float* logits, int count)
        => System.Numerics.Tensors.TensorPrimitives.IndexOfMax(new ReadOnlySpan<float>(logits, count));
}
