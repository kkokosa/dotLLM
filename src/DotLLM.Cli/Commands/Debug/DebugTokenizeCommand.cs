#if DEBUG
using System.ComponentModel;
using DotLLM.Cli.Commands;
using DotLLM.Models.Gguf;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands.Debug;

/// <summary>
/// Tokenizer round-trip diagnostics: tokenize text, display individual tokens, verify decode round-trip.
/// </summary>
internal sealed class DebugTokenizeCommand : Command<DebugTokenizeCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<file>")]
        [Description("Path to a GGUF file or HuggingFace repo ID.")]
        public string FilePath { get; set; } = string.Empty;

        [CommandOption("--text|-t")]
        [Description("Text to tokenize (required).")]
        public string? Text { get; set; }
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        if (string.IsNullOrEmpty(settings.Text))
        {
            AnsiConsole.MarkupLine("[red]--text|-t is required.[/]");
            return 1;
        }

        var resolvedPath = GgufFileResolver.Resolve(settings.FilePath);
        if (resolvedPath is null)
            return 1;

        using var gguf = GgufFile.Open(resolvedPath);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        // Tokenizer info
        AnsiConsole.Write(new Rule("[bold yellow]Tokenizer Info[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var infoTable = new Table().Border(TableBorder.Rounded);
        infoTable.AddColumn("Property");
        infoTable.AddColumn("Value");
        infoTable.AddRow("Vocab size", tokenizer.VocabSize.ToString("N0"));
        infoTable.AddRow("BOS token ID", tokenizer.BosTokenId.ToString());
        infoTable.AddRow("EOS token ID", tokenizer.EosTokenId.ToString());
        AnsiConsole.Write(infoTable);
        AnsiConsole.WriteLine();

        // Encode
        int[] tokenIds = tokenizer.Encode(settings.Text);

        AnsiConsole.Write(new Rule($"[bold yellow]Tokens ({tokenIds.Length} tokens)[/]").LeftJustified());
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

        // Round-trip decode
        string decoded = tokenizer.Decode(tokenIds);

        AnsiConsole.Write(new Rule("[bold yellow]Round-trip[/]").LeftJustified());
        AnsiConsole.WriteLine();

        bool match = settings.Text == decoded;
        AnsiConsole.MarkupLine($"Original:  {settings.Text.EscapeMarkup()}");
        AnsiConsole.MarkupLine($"Decoded:   {decoded.EscapeMarkup()}");

        if (match)
            AnsiConsole.MarkupLine("[green]\u2713 Round-trip match[/]");
        else
            AnsiConsole.MarkupLine("[red]\u2717 Round-trip MISMATCH[/]");

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
