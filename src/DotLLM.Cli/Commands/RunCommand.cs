using System.ComponentModel;
using System.Diagnostics;
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

        AnsiConsole.MarkupLine($"[grey]Model: {config.Architecture}, {config.NumLayers} layers, {config.HiddenSize} hidden, {config.VocabSize:N0} vocab[/]");
        AnsiConsole.MarkupLine("[grey]Warning: No KV-cache — full context reprocessed each step (slow)[/]");
        AnsiConsole.WriteLine();

        try
        {
            // Encode prompt
            int[] promptTokens = tokenizer.Encode(settings.Prompt);
            int promptLen = promptTokens.Length;

            // Build the full token sequence (prompt + generated)
            var tokens = new List<int>(promptLen + settings.MaxTokens);
            tokens.AddRange(promptTokens);

            // Print prompt echo
            Console.Write(settings.Prompt);

            var totalSw = Stopwatch.StartNew();
            int generated = 0;

            for (int step = 0; step < settings.MaxTokens; step++)
            {
                int seqLen = tokens.Count;
                int[] tokenArray = tokens.ToArray();
                int[] positions = new int[seqLen];
                for (int i = 0; i < seqLen; i++)
                    positions[i] = i;

                // Forward pass — returns [seqLen, vocabSize] logits
                using var logitsTensor = model.Forward(tokenArray, positions, -1);

                // Greedy argmax on last token's logits
                int vocabSize = config.VocabSize;
                unsafe
                {
                    float* logits = (float*)logitsTensor.DataPointer;
                    float* lastLogits = logits + (seqLen - 1) * (long)vocabSize;

                    int bestId = 0;
                    float bestVal = lastLogits[0];
                    for (int i = 1; i < vocabSize; i++)
                    {
                        if (lastLogits[i] > bestVal)
                        {
                            bestVal = lastLogits[i];
                            bestId = i;
                        }
                    }

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

            // Timing stats
            double totalSec = totalSw.Elapsed.TotalSeconds;
            double tokPerSec = generated > 0 ? generated / totalSec : 0;

            var statsTable = new Table().Border(TableBorder.Rounded).Title("[bold]Generation Stats[/]");
            statsTable.AddColumn("Metric");
            statsTable.AddColumn("Value");
            statsTable.AddRow("Prompt tokens", promptLen.ToString());
            statsTable.AddRow("Generated tokens", generated.ToString());
            statsTable.AddRow("Total time", $"{totalSec:F2}s");
            statsTable.AddRow("Tokens/sec", $"{tokPerSec:F2}");
            AnsiConsole.Write(statsTable);
        }
        finally
        {
            model.Dispose();
            gguf.Dispose();
        }

        return 0;
    }
}
