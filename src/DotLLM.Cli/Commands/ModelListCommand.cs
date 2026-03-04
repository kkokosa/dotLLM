using DotLLM.Cli.Helpers;
using DotLLM.HuggingFace;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Lists locally downloaded GGUF models.
/// </summary>
internal sealed class ModelListCommand : Command<ModelListCommand.Settings>
{
    public sealed class Settings : CommandSettings;

    public override int Execute(CommandContext context, Settings settings)
    {
        var models = HuggingFaceDownloader.ListLocalModels();

        if (models.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No locally downloaded models found.[/]");
            AnsiConsole.MarkupLine($"[dim]Models directory: {HuggingFaceDownloader.DefaultModelsDirectory.EscapeMarkup()}[/]");
            return 0;
        }

        var table = new Table();
        table.Border(TableBorder.Rounded);
        table.AddColumn("Repository");
        table.AddColumn("Filename");
        table.AddColumn(new TableColumn("Size").RightAligned());
        table.AddColumn("Downloaded");

        foreach (var model in models.OrderByDescending(m => m.DownloadedAt))
        {
            table.AddRow(
                $"[bold]{model.RepoId.EscapeMarkup()}[/]",
                model.Filename.EscapeMarkup(),
                FormatHelpers.FormatSize(model.SizeBytes),
                model.DownloadedAt.LocalDateTime.ToString("yyyy-MM-dd HH:mm"));
        }

        AnsiConsole.Write(table);
        return 0;
    }

}
