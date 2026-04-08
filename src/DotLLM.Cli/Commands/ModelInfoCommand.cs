using System.ComponentModel;
using DotLLM.Cli.Helpers;
using DotLLM.HuggingFace;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Shows details for a HuggingFace model repository, including available GGUF files.
/// </summary>
internal sealed class ModelInfoCommand : AsyncCommand<ModelInfoCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<repo-id>")]
        [Description("HuggingFace repository ID (e.g. 'TheBloke/Llama-2-7B-GGUF').")]
        public string RepoId { get; set; } = string.Empty;
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        using var client = new HuggingFaceClient();

        var (model, ggufFiles) = await AnsiConsole.Status()
            .StartAsync("Fetching model info...", async _ =>
            {
                var m = await client.GetModelInfoAsync(settings.RepoId);
                var files = await client.ListGgufFilesAsync(settings.RepoId);
                return (m, files);
            });

        AnsiConsole.Write(new Rule($"[bold]{model.Id.EscapeMarkup()}[/]").LeftJustified());
        AnsiConsole.WriteLine();

        var infoTable = new Table { Border = TableBorder.None, ShowHeaders = false };
        infoTable.AddColumn("Key");
        infoTable.AddColumn("Value");
        infoTable.AddRow("[bold]Author[/]", model.Author?.EscapeMarkup() ?? "-");
        infoTable.AddRow("[bold]Downloads[/]", model.Downloads.ToString("N0"));
        infoTable.AddRow("[bold]Likes[/]", model.Likes.ToString("N0"));
        infoTable.AddRow("[bold]Pipeline[/]", model.PipelineTag?.EscapeMarkup() ?? "-");
        infoTable.AddRow("[bold]Tags[/]", model.Tags.Count > 0
            ? string.Join(", ", model.Tags.Take(10)).EscapeMarkup()
            : "-");
        AnsiConsole.Write(infoTable);

        // Check for locally downloaded files
        var localModels = HuggingFaceDownloader.ListLocalModels()
            .Where(m => m.RepoId.Equals(settings.RepoId, StringComparison.OrdinalIgnoreCase))
            .ToDictionary(m => m.Filename, StringComparer.OrdinalIgnoreCase);

        if (localModels.Count > 0)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[bold]Local Path[/]").LeftJustified());
            var localDir = Path.GetDirectoryName(localModels.Values.First().FullPath)!;
            AnsiConsole.MarkupLine($"  [dim]{localDir.EscapeMarkup()}[/]");
        }

        if (ggufFiles.Count > 0)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.Write(new Rule("[bold]GGUF Files[/]").LeftJustified());

            var fileTable = new Table();
            fileTable.Border(TableBorder.Rounded);
            fileTable.AddColumn("Filename");
            fileTable.AddColumn(new TableColumn("Size").RightAligned());
            fileTable.AddColumn(new TableColumn("Local").Centered());

            foreach (var file in ggufFiles.OrderBy(f => f.Path))
            {
                var isLocal = localModels.ContainsKey(file.Path);
                fileTable.AddRow(
                    file.Path.EscapeMarkup(),
                    FormatHelpers.FormatSize(file.Size),
                    isLocal ? "[green]yes[/]" : "[dim]-[/]");
            }

            AnsiConsole.Write(fileTable);
        }

        return 0;
    }

}
