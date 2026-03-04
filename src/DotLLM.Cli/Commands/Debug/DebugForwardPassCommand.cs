#if DEBUG
using System.ComponentModel;
using System.Diagnostics;
using DotLLM.Cli.Commands;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands.Debug;

/// <summary>
/// Runs a single forward pass and displays detailed logit diagnostics: model config summary,
/// token IDs, top-10 predicted tokens with softmax probabilities, and logit statistics.
/// </summary>
internal sealed class DebugForwardPassCommand : Command<DebugForwardPassCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<file>")]
        [Description("Path to a GGUF file or HuggingFace repo ID.")]
        public string FilePath { get; set; } = string.Empty;

        [CommandOption("--prompt|-p")]
        [Description("Input text (default: BOS token only).")]
        public string? Prompt { get; set; }
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        var resolvedPath = GgufFileResolver.Resolve(settings.FilePath);
        if (resolvedPath is null)
            return 1;

        using var gguf = GgufFile.Open(resolvedPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        using var model = LlamaModel.LoadFromGguf(gguf, config);

        // Model config summary
        AnsiConsole.Write(new Rule("[bold yellow]Model Config[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var configTable = new Table().Border(TableBorder.Rounded);
        configTable.AddColumn("Property");
        configTable.AddColumn("Value");
        configTable.AddRow("Architecture", config.Architecture.ToString());
        configTable.AddRow("Hidden size", config.HiddenSize.ToString("N0"));
        configTable.AddRow("Layers", config.NumLayers.ToString());
        configTable.AddRow("Attention heads", config.NumAttentionHeads.ToString());
        configTable.AddRow("KV heads", config.NumKvHeads.ToString());
        configTable.AddRow("Vocab size", config.VocabSize.ToString("N0"));

        // Find embedding quant type from GGUF tensors
        if (gguf.TensorsByName.TryGetValue("token_embd.weight", out var embDesc))
            configTable.AddRow("Embedding quant", embDesc.QuantizationType.ToString());
        if (gguf.TensorsByName.TryGetValue("output.weight", out var outDesc))
            configTable.AddRow("Output quant", outDesc.QuantizationType.ToString());

        AnsiConsole.Write(configTable);
        AnsiConsole.WriteLine();

        // Tokenize prompt
        int[] tokenIds;
        if (string.IsNullOrEmpty(settings.Prompt))
        {
            tokenIds = [tokenizer.BosTokenId];
            AnsiConsole.MarkupLine("[grey]No prompt specified — using BOS token only.[/]");
        }
        else
        {
            tokenIds = tokenizer.Encode(settings.Prompt);
        }

        // Token IDs table
        AnsiConsole.Write(new Rule("[bold yellow]Input Tokens[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var tokenTable = new Table().Border(TableBorder.Rounded);
        tokenTable.AddColumn("Position");
        tokenTable.AddColumn("Token ID");
        tokenTable.AddColumn("Text");

        for (int i = 0; i < tokenIds.Length; i++)
        {
            string text = tokenizer.DecodeToken(tokenIds[i]);
            tokenTable.AddRow(i.ToString(), tokenIds[i].ToString(), EscapeTokenText(text).EscapeMarkup());
        }
        AnsiConsole.Write(tokenTable);
        AnsiConsole.WriteLine();

        // Forward pass
        int[] positions = new int[tokenIds.Length];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;

        var sw = Stopwatch.StartNew();
        using var logitsTensor = model.Forward(tokenIds, positions, -1);
        sw.Stop();

        AnsiConsole.Write(new Rule("[bold yellow]Forward Pass[/]").LeftJustified());
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"Time: [bold]{sw.Elapsed.TotalMilliseconds:F1} ms[/]");
        AnsiConsole.MarkupLine($"Logit tensor shape: [bold]{logitsTensor.Shape.ToString().EscapeMarkup()}[/]");
        AnsiConsole.WriteLine();

        // Analyze logits (already last-token-only: [1, vocabSize])
        int vocabSize = config.VocabSize;

        unsafe
        {
            float* lastLogits = (float*)logitsTensor.DataPointer;

            // Compute stats
            float min = lastLogits[0], max = lastLogits[0];
            double sum = 0;
            for (int i = 0; i < vocabSize; i++)
            {
                float v = lastLogits[i];
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
            }
            double mean = sum / vocabSize;

            double sumSqDiff = 0;
            for (int i = 0; i < vocabSize; i++)
            {
                double diff = lastLogits[i] - mean;
                sumSqDiff += diff * diff;
            }
            double stdDev = Math.Sqrt(sumSqDiff / vocabSize);

            // Top-10 by logit value
            var top10 = new (int id, float logit)[10];
            for (int i = 0; i < 10; i++)
                top10[i] = (-1, float.NegativeInfinity);

            for (int i = 0; i < vocabSize; i++)
            {
                float v = lastLogits[i];
                if (v > top10[9].logit)
                {
                    top10[9] = (i, v);
                    // Bubble sort into position
                    for (int j = 9; j > 0 && top10[j].logit > top10[j - 1].logit; j--)
                        (top10[j], top10[j - 1]) = (top10[j - 1], top10[j]);
                }
            }

            // Softmax for top-10 (numerically stable: subtract max)
            double[] softmaxProbs = new double[10];
            double expSum = 0;
            for (int i = 0; i < vocabSize; i++)
                expSum += Math.Exp(lastLogits[i] - max);
            for (int i = 0; i < 10; i++)
                softmaxProbs[i] = Math.Exp(top10[i].logit - max) / expSum;

            // Top-10 table
            AnsiConsole.Write(new Rule("[bold yellow]Top-10 Predicted Tokens (last position)[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var predTable = new Table().Border(TableBorder.Rounded);
            predTable.AddColumn("Rank");
            predTable.AddColumn("Token ID");
            predTable.AddColumn("Text");
            predTable.AddColumn("Logit");
            predTable.AddColumn("Probability");

            for (int i = 0; i < 10; i++)
            {
                if (top10[i].id < 0) break;
                string text = tokenizer.DecodeToken(top10[i].id);
                predTable.AddRow(
                    (i + 1).ToString(),
                    top10[i].id.ToString(),
                    EscapeTokenText(text).EscapeMarkup(),
                    top10[i].logit.ToString("F4"),
                    softmaxProbs[i].ToString("P4"));
            }
            AnsiConsole.Write(predTable);
            AnsiConsole.WriteLine();

            // Logit stats
            AnsiConsole.Write(new Rule("[bold yellow]Logit Statistics[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var statsTable = new Table().Border(TableBorder.Rounded);
            statsTable.AddColumn("Stat");
            statsTable.AddColumn("Value");
            statsTable.AddRow("Min", min.ToString("F4"));
            statsTable.AddRow("Max", max.ToString("F4"));
            statsTable.AddRow("Mean", mean.ToString("F4"));
            statsTable.AddRow("Std Dev", stdDev.ToString("F4"));
            AnsiConsole.Write(statsTable);
        }

        return 0;
    }

    private static string EscapeTokenText(string text)
    {
        return text
            .Replace("\n", "\\n")
            .Replace("\r", "\\r")
            .Replace("\t", "\\t")
            .Replace("\0", "\\0");
    }
}
#endif
