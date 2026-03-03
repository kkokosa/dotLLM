#if DEBUG
using System.ComponentModel;
using DotLLM.Cli.Commands;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands.Debug;

/// <summary>
/// Parses and displays the GGUF file header: version, tensor count, metadata count, data section offset, and file size.
/// </summary>
internal sealed class DebugGgufHeaderCommand : Command<DebugGgufHeaderCommand.Settings>
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

        var fileInfo = new FileInfo(resolvedPath);
        using var gguf = GgufFile.Open(resolvedPath);

        AnsiConsole.Write(new Rule("[bold yellow]GGUF Header[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Field");
        table.AddColumn("Value");

        table.AddRow("File", fileInfo.Name);
        table.AddRow("File size", FormatBytes(fileInfo.Length));
        table.AddRow("Version", gguf.Header.Version.ToString());
        table.AddRow("Tensor count", gguf.Header.TensorCount.ToString("N0"));
        table.AddRow("Metadata KV count", gguf.Header.MetadataKvCount.ToString("N0"));
        table.AddRow("Data section offset", $"0x{gguf.DataSectionOffset:X} ({gguf.DataSectionOffset:N0} bytes)");

        AnsiConsole.Write(table);
        return 0;
    }

    private static string FormatBytes(long bytes)
    {
        return bytes switch
        {
            >= 1L << 30 => $"{bytes / (double)(1L << 30):F2} GiB ({bytes:N0} bytes)",
            >= 1L << 20 => $"{bytes / (double)(1L << 20):F2} MiB ({bytes:N0} bytes)",
            >= 1L << 10 => $"{bytes / (double)(1L << 10):F2} KiB ({bytes:N0} bytes)",
            _ => $"{bytes:N0} bytes"
        };
    }
}
#endif
