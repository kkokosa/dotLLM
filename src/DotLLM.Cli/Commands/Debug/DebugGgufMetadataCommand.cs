#if DEBUG
using System.ComponentModel;
using DotLLM.Cli.Commands;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands.Debug;

/// <summary>
/// Lists all GGUF metadata key-value pairs with their types and values.
/// </summary>
internal sealed class DebugGgufMetadataCommand : Command<DebugGgufMetadataCommand.Settings>
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

        AnsiConsole.Write(new Rule($"[bold yellow]GGUF Metadata ({gguf.Metadata.Count} entries)[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Key");
        table.AddColumn("Type");
        table.AddColumn("Value");

        foreach (string key in gguf.Metadata.Keys.OrderBy(k => k))
        {
            if (!gguf.Metadata.TryGetValue(key, out var entry))
                continue;

            string typeStr = entry.Type.ToString();
            string valueStr = FormatValue(entry);

            table.AddRow(key.EscapeMarkup(), typeStr, valueStr.EscapeMarkup());
        }

        AnsiConsole.Write(table);
        return 0;
    }

    private static string FormatValue(GgufMetadataValue entry)
    {
        if (entry.Type == GgufValueType.Array && entry.Value is Array array)
        {
            string elementType = array.GetType().GetElementType()?.Name ?? "?";
            return $"[{elementType}; {array.Length} elements]";
        }

        if (entry.Type == GgufValueType.String && entry.Value is string s)
        {
            const int maxLen = 120;
            return s.Length > maxLen ? string.Concat(s.AsSpan(0, maxLen), "...") : s;
        }

        return entry.Value?.ToString() ?? "(null)";
    }
}
#endif
