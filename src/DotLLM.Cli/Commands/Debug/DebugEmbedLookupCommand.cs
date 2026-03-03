#if DEBUG
using System.ComponentModel;
using DotLLM.Cli.Commands;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands.Debug;

/// <summary>
/// Inspects the embedding vector for a single token: dequantizes one row from the embedding table
/// and displays vector statistics plus the first and last 16 values.
/// </summary>
internal sealed class DebugEmbedLookupCommand : Command<DebugEmbedLookupCommand.Settings>
{
    /// <summary>Q8_0 block: 2 bytes (Half scale) + 32 bytes (sbyte values).</summary>
    private const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    private const int Q8_0GroupSize = 32;

    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<file>")]
        [Description("Path to a GGUF file or HuggingFace repo ID.")]
        public string FilePath { get; set; } = string.Empty;

        [CommandOption("--token-id|-i")]
        [Description("Token ID to look up (required).")]
        public int TokenId { get; set; } = -1;
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        if (settings.TokenId < 0)
        {
            AnsiConsole.MarkupLine("[red]--token-id|-i is required (non-negative integer).[/]");
            return 1;
        }

        var resolvedPath = GgufFileResolver.Resolve(settings.FilePath);
        if (resolvedPath is null)
            return 1;

        using var gguf = GgufFile.Open(resolvedPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        if (settings.TokenId >= config.VocabSize)
        {
            AnsiConsole.MarkupLine($"[red]Token ID {settings.TokenId} out of range (vocab size: {config.VocabSize}).[/]");
            return 1;
        }

        // Get embedding tensor descriptor
        if (!gguf.TensorsByName.TryGetValue("token_embd.weight", out var embDesc))
        {
            AnsiConsole.MarkupLine("[red]token_embd.weight tensor not found in GGUF file.[/]");
            return 1;
        }

        int hiddenSize = config.HiddenSize;
        nint dataBase = gguf.DataBasePointer;
        nint embPtr = dataBase + (nint)embDesc.DataOffset;
        var qt = embDesc.QuantizationType;

        // Dequantize one row
        float[] embedding = new float[hiddenSize];
        DequantizeRow(embPtr, qt, settings.TokenId, hiddenSize, embedding);

        // Token info
        string tokenText = tokenizer.DecodeToken(settings.TokenId);

        AnsiConsole.Write(new Rule("[bold yellow]Token Info[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var infoTable = new Table().Border(TableBorder.Rounded);
        infoTable.AddColumn("Property");
        infoTable.AddColumn("Value");
        infoTable.AddRow("Token ID", settings.TokenId.ToString());
        infoTable.AddRow("Decoded text", EscapeTokenText(tokenText).EscapeMarkup());
        infoTable.AddRow("Embedding quant type", qt.ToString());
        AnsiConsole.Write(infoTable);
        AnsiConsole.WriteLine();

        // Vector stats
        float min = embedding[0], max = embedding[0];
        double sum = 0, sumSq = 0;
        for (int i = 0; i < hiddenSize; i++)
        {
            float v = embedding[i];
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumSq += (double)v * v;
        }
        double mean = sum / hiddenSize;
        double l2Norm = Math.Sqrt(sumSq);

        AnsiConsole.Write(new Rule("[bold yellow]Vector Statistics[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var statsTable = new Table().Border(TableBorder.Rounded);
        statsTable.AddColumn("Stat");
        statsTable.AddColumn("Value");
        statsTable.AddRow("Dimension", hiddenSize.ToString());
        statsTable.AddRow("Min", min.ToString("F6"));
        statsTable.AddRow("Max", max.ToString("F6"));
        statsTable.AddRow("Mean", mean.ToString("F6"));
        statsTable.AddRow("L2 norm", l2Norm.ToString("F6"));
        AnsiConsole.Write(statsTable);
        AnsiConsole.WriteLine();

        // First 16 and last 16 values
        int showCount = Math.Min(16, hiddenSize);

        AnsiConsole.Write(new Rule($"[bold yellow]First {showCount} values[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var firstTable = new Table().Border(TableBorder.Rounded);
        firstTable.AddColumn("Index");
        firstTable.AddColumn("Value");
        for (int i = 0; i < showCount; i++)
            firstTable.AddRow(i.ToString(), embedding[i].ToString("F6"));
        AnsiConsole.Write(firstTable);
        AnsiConsole.WriteLine();

        if (hiddenSize > showCount)
        {
            int start = hiddenSize - showCount;
            AnsiConsole.Write(new Rule($"[bold yellow]Last {showCount} values[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var lastTable = new Table().Border(TableBorder.Rounded);
            lastTable.AddColumn("Index");
            lastTable.AddColumn("Value");
            for (int i = start; i < hiddenSize; i++)
                lastTable.AddRow(i.ToString(), embedding[i].ToString("F6"));
            AnsiConsole.Write(lastTable);
        }

        return 0;
    }

    private static unsafe void DequantizeRow(nint embPtr, QuantizationType qt, int tokenId, int hiddenSize, Span<float> dest)
    {
        switch (qt)
        {
            case QuantizationType.F32:
            {
                float* src = (float*)embPtr + (long)tokenId * hiddenSize;
                new ReadOnlySpan<float>(src, hiddenSize).CopyTo(dest);
                break;
            }
            case QuantizationType.Q8_0:
            {
                int blocksPerRow = hiddenSize / Q8_0GroupSize;
                long rowOffset = (long)tokenId * blocksPerRow * Q8_0BlockBytes;
                nint rowPtr = embPtr + (nint)rowOffset;
                Dequantize.ToFloat32(rowPtr, hiddenSize, QuantizationType.Q8_0, dest);
                break;
            }
            case QuantizationType.F16:
            {
                Half* src = (Half*)embPtr + (long)tokenId * hiddenSize;
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(
                    new ReadOnlySpan<Half>(src, hiddenSize), dest);
                break;
            }
            default:
                throw new NotSupportedException($"Unsupported embedding quantization type: {qt}");
        }
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
