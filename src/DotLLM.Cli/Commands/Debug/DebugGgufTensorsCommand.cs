#if DEBUG
using System.ComponentModel;
using DotLLM.Cli.Commands;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands.Debug;

/// <summary>
/// Lists all tensor descriptors from a GGUF file: name, shape, quantization type, and data offset.
/// </summary>
internal sealed class DebugGgufTensorsCommand : Command<DebugGgufTensorsCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<file>")]
        [Description("Path to a GGUF file or HuggingFace repo ID (e.g., Qwen/Qwen3-0.6B-GGUF).")]
        public string FilePath { get; set; } = string.Empty;

        [CommandOption("--filter|-f")]
        [Description("Filter tensor names (case-insensitive substring match).")]
        public string? Filter { get; set; }
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        var resolvedPath = GgufFileResolver.Resolve(settings.FilePath);
        if (resolvedPath is null)
            return 1;

        using var gguf = GgufFile.Open(resolvedPath);

        var tensors = gguf.Tensors.AsEnumerable();
        if (!string.IsNullOrEmpty(settings.Filter))
            tensors = tensors.Where(t => t.Name.Contains(settings.Filter, StringComparison.OrdinalIgnoreCase));

        var filtered = tensors.ToList();

        AnsiConsole.Write(new Rule($"[bold yellow]GGUF Tensors ({filtered.Count} of {gguf.Tensors.Count})[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("#");
        table.AddColumn("Name");
        table.AddColumn("Shape");
        table.AddColumn("Quant Type");
        table.AddColumn("Offset");

        int idx = 0;
        foreach (var tensor in filtered)
        {
            table.AddRow(
                (idx++).ToString(),
                tensor.Name.EscapeMarkup(),
                tensor.Shape.ToString().EscapeMarkup(),
                tensor.QuantizationType.ToString(),
                $"0x{tensor.DataOffset:X}");
        }

        AnsiConsole.Write(table);
        return 0;
    }
}
#endif
