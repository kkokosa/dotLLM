using System.ComponentModel;
using DotLLM.HuggingFace;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Downloads a GGUF file from a HuggingFace repository.
/// </summary>
internal sealed class ModelPullCommand : AsyncCommand<ModelPullCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<repo-id>")]
        [Description("HuggingFace repository ID (e.g. 'TheBloke/Llama-2-7B-GGUF').")]
        public string RepoId { get; set; } = string.Empty;

        [CommandOption("--file|-f")]
        [Description("Specific GGUF filename to download. If omitted, lists available files for selection.")]
        public string? Filename { get; set; }

        [CommandOption("--dir|-d")]
        [Description("Destination directory. Defaults to ~/.dotllm/models/.")]
        public string? Directory { get; set; }
    }

    protected override async Task<int> ExecuteAsync(CommandContext context, Settings settings, CancellationToken cancellationToken)
    {
        using var client = new HuggingFaceClient();
        using var downloader = new HuggingFaceDownloader();

        var filename = settings.Filename;

        // If no filename specified, list GGUF files and let user pick
        if (string.IsNullOrEmpty(filename))
        {
            var ggufFiles = await AnsiConsole.Status()
                .StartAsync("Fetching file list...", async _ =>
                    await client.ListGgufFilesAsync(settings.RepoId));

            if (ggufFiles.Count == 0)
            {
                AnsiConsole.MarkupLine("[red]No GGUF files found in repository.[/]");
                return 1;
            }

            filename = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("Select a GGUF file to download:")
                    .AddChoices(ggufFiles.Select(f => f.Path)));
        }

        AnsiConsole.MarkupLine($"Downloading [bold]{filename.EscapeMarkup()}[/] from [bold]{settings.RepoId.EscapeMarkup()}[/]...");

        var path = await AnsiConsole.Progress()
            .AutoClear(false)
            .Columns(
                new TaskDescriptionColumn(),
                new ProgressBarColumn(),
                new PercentageColumn(),
                new TransferSpeedColumn(),
                new RemainingTimeColumn())
            .StartAsync(async ctx =>
            {
                var task = ctx.AddTask($"[green]{filename.EscapeMarkup()}[/]", maxValue: 100);
                long? lastTotal = null;

                var progress = new Progress<(long bytesDownloaded, long? totalBytes)>(p =>
                {
                    if (p.totalBytes.HasValue)
                    {
                        if (lastTotal != p.totalBytes.Value)
                        {
                            task.MaxValue = p.totalBytes.Value;
                            lastTotal = p.totalBytes.Value;
                        }
                        task.Value = p.bytesDownloaded;
                    }
                });

                return await downloader.DownloadFileAsync(
                    settings.RepoId, filename, settings.Directory, progress);
            });

        AnsiConsole.MarkupLine($"[green]Saved to:[/] {path.EscapeMarkup()}");
        return 0;
    }
}
