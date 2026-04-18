#if DEBUG
using System.ComponentModel;
using DotLLM.Cli.Commands;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands.Debug;

/// <summary>
/// Extracts and displays the <see cref="DotLLM.Core.Models.ModelConfig"/> from a GGUF file's metadata.
/// </summary>
internal sealed class DebugGgufConfigCommand : Command<DebugGgufConfigCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<file>")]
        [Description("Path to a GGUF file or HuggingFace repo ID (e.g., Qwen/Qwen3-0.6B-GGUF).")]
        public string FilePath { get; set; } = string.Empty;
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        var resolvedPath = GgufFileResolver.Resolve(settings.FilePath);
        if (resolvedPath is null)
            return 1;

        using var gguf = GgufFile.Open(resolvedPath);

        Core.Models.ModelConfig config;
        try
        {
            config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Failed to extract ModelConfig:[/] {ex.Message.EscapeMarkup()}");
            return 1;
        }

        AnsiConsole.Write(new Rule("[bold yellow]ModelConfig[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Property");
        table.AddColumn("Value");

        table.AddRow("Architecture", config.Architecture.ToString());
        table.AddRow("VocabSize", config.VocabSize.ToString("N0"));
        table.AddRow("HiddenSize", config.HiddenSize.ToString("N0"));
        table.AddRow("IntermediateSize", config.IntermediateSize.ToString("N0"));
        table.AddRow("NumLayers", config.NumLayers.ToString());
        table.AddRow("NumAttentionHeads", config.NumAttentionHeads.ToString());
        table.AddRow("NumKvHeads", config.NumKvHeads.ToString());
        table.AddRow("HeadDim", config.HeadDim.ToString());
        table.AddRow("MaxSequenceLength", config.MaxSequenceLength.ToString("N0"));
        table.AddRow("AttentionType", config.AttentionType.ToString());
        table.AddRow("PositionEncodingType", config.PositionEncodingType.ToString());
        table.AddRow("ActivationFunction", config.ActivationFunction.ToString());
        table.AddRow("NormType", config.NormType.ToString());
        table.AddRow("NormEpsilon", config.NormEpsilon.ToString("G"));
        table.AddRow("TiedEmbeddings", config.TiedEmbeddings.ToString());
        table.AddRow("SlidingWindowSize", config.SlidingWindowSize?.ToString() ?? "(none)");
        string chatTemplateDisplay = config.ChatTemplate is not null
            ? (config.ChatTemplate.Length > 80
                ? string.Concat(config.ChatTemplate.AsSpan(0, 80), "...")
                : config.ChatTemplate)
            : "(none)";
        table.AddRow("ChatTemplate", chatTemplateDisplay.EscapeMarkup());

        AnsiConsole.Write(table);

        if (config.SsmConfig is { } ssm)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[bold yellow]Mamba SSM Config[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var ssmTable = new Table().Border(TableBorder.Rounded);
            ssmTable.AddColumn("Property");
            ssmTable.AddColumn("Value");

            ssmTable.AddRow("DConv", ssm.DConv.ToString());
            ssmTable.AddRow("DInner", ssm.DInner.ToString());
            ssmTable.AddRow("DState", ssm.DState.ToString());
            ssmTable.AddRow("NGroup", ssm.NGroup.ToString());
            ssmTable.AddRow("NHead", ssm.NHead.ToString());
            ssmTable.AddRow("HeadDim (derived)", ssm.HeadDim.ToString());
            ssmTable.AddRow("InputProjectionDim (derived)", ssm.InputProjectionDim.ToString());
            ssmTable.AddRow("ConvDim (derived)", ssm.ConvDim.ToString());

            AnsiConsole.Write(ssmTable);
        }

        if (config.HybridLayout is { } hybrid)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[bold yellow]Hybrid Layer Layout[/]").LeftJustified());
            AnsiConsole.WriteLine();

            int ssmCount = 0, attnCount = 0, ffnCount = 0;
            foreach (var k in hybrid.LayerKind)
            {
                if (k == Core.Models.HybridLayerKind.Ssm) ssmCount++;
                else if (k == Core.Models.HybridLayerKind.Attention) attnCount++;
                else if (k == Core.Models.HybridLayerKind.Ffn) ffnCount++;
            }
            AnsiConsole.MarkupLine(
                $"SSM layers: [green]{ssmCount}[/] | Attention layers: [green]{attnCount}[/] | FFN layers: [green]{ffnCount}[/] | total {hybrid.LayerKind.Length}");
            AnsiConsole.MarkupLine(
                $"Layout: {string.Join("", hybrid.LayerKind.Select(k => k switch {
                    Core.Models.HybridLayerKind.Ssm => 'S',
                    Core.Models.HybridLayerKind.Attention => 'A',
                    Core.Models.HybridLayerKind.Ffn => 'F',
                    _ => '?'
                }))}");
        }

        if (config.RoPEConfig is { } rope)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[bold yellow]RoPE Config[/]").LeftJustified());
            AnsiConsole.WriteLine();

            var ropeTable = new Table().Border(TableBorder.Rounded);
            ropeTable.AddColumn("Property");
            ropeTable.AddColumn("Value");

            ropeTable.AddRow("Theta", rope.Theta.ToString("G"));
            ropeTable.AddRow("DimensionCount", rope.DimensionCount.ToString());
            ropeTable.AddRow("ScalingType", rope.ScalingType.ToString());
            ropeTable.AddRow("ScalingFactor", rope.ScalingFactor.ToString("G"));
            ropeTable.AddRow("OrigMaxSeqLen", rope.OrigMaxSeqLen.ToString());
            ropeTable.AddRow("AttnFactor", rope.AttnFactor.ToString("G"));
            ropeTable.AddRow("BetaFast", rope.BetaFast.ToString("G"));
            ropeTable.AddRow("BetaSlow", rope.BetaSlow.ToString("G"));

            AnsiConsole.Write(ropeTable);
        }

        return 0;
    }
}
#endif
