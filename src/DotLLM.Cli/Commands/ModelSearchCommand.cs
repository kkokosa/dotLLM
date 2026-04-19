using System.ComponentModel;
using DotLLM.HuggingFace;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Searches HuggingFace Hub for GGUF models.
/// </summary>
internal sealed class ModelSearchCommand : AsyncCommand<ModelSearchCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<query>")]
        [Description("Search query (e.g. 'llama', 'mistral 7b').")]
        public string Query { get; set; } = string.Empty;

        [CommandOption("--limit")]
        [Description("Maximum number of results.")]
        [DefaultValue(20)]
        public int Limit { get; set; } = 20;

        [CommandOption("--sort")]
        [Description("Sort field: downloads, likes, lastModified.")]
        [DefaultValue("downloads")]
        public string Sort { get; set; } = "downloads";
    }

    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        using var client = new HuggingFaceClient();

        var models = await AnsiConsole.Status()
            .StartAsync("Searching HuggingFace...", async _ =>
                await client.SearchModelsAsync(settings.Query, settings.Limit, settings.Sort));

        if (models.Count == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No GGUF models found.[/]");
            return 0;
        }

        var table = new Table();
        table.Border(TableBorder.Rounded);
        table.AddColumn("Repository");
        table.AddColumn(new TableColumn("Downloads").RightAligned());
        table.AddColumn(new TableColumn("Likes").RightAligned());
        table.AddColumn("Tags");

        foreach (var model in models)
        {
            var tags = model.Tags.Count > 0
                ? string.Join(", ", model.Tags.Take(5))
                : "-";
            table.AddRow(
                $"[bold]{model.Id.EscapeMarkup()}[/]",
                model.Downloads.ToString("N0"),
                model.Likes.ToString("N0"),
                tags.EscapeMarkup());
        }

        AnsiConsole.Write(table);
        return 0;
    }
}
