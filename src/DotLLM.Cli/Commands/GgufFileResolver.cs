using DotLLM.HuggingFace;
using Spectre.Console;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Resolves a GGUF file argument to a local file path. Accepts either a direct file path
/// or a HuggingFace repo ID (e.g., "Qwen/Qwen3-0.6B-GGUF") and looks up downloaded models.
/// </summary>
internal static class GgufFileResolver
{
    /// <summary>
    /// Resolves the argument to a local .gguf file path.
    /// Returns null and prints an error if resolution fails.
    /// </summary>
    public static string? Resolve(string fileArg)
    {
        // Direct file path — use as-is.
        if (File.Exists(fileArg))
            return Path.GetFullPath(fileArg);

        // Check if it looks like a repo ID (contains /).
        if (fileArg.Contains('/'))
        {
            string modelsDir = HuggingFaceDownloader.DefaultModelsDirectory;
            string repoDir = Path.Combine(modelsDir, fileArg.Replace('/', Path.DirectorySeparatorChar));

            if (!Directory.Exists(repoDir))
            {
                AnsiConsole.MarkupLine($"[red]Not found as file or downloaded model:[/] {fileArg.EscapeMarkup()}");
                AnsiConsole.MarkupLine($"[grey]Checked: {repoDir.EscapeMarkup()}[/]");
                return null;
            }

            var ggufFiles = Directory.GetFiles(repoDir, "*.gguf");
            switch (ggufFiles.Length)
            {
                case 0:
                    AnsiConsole.MarkupLine($"[red]No .gguf files found in:[/] {repoDir.EscapeMarkup()}");
                    return null;

                case 1:
                    return ggufFiles[0];

                default:
                    AnsiConsole.MarkupLine($"[yellow]Multiple .gguf files in {fileArg.EscapeMarkup()}:[/]");
                    foreach (var f in ggufFiles)
                        AnsiConsole.MarkupLine($"  {Path.GetFileName(f).EscapeMarkup()}");
                    AnsiConsole.MarkupLine("[grey]Specify the full path or repo/file to disambiguate.[/]");
                    return null;
            }
        }

        AnsiConsole.MarkupLine($"[red]File not found:[/] {fileArg.EscapeMarkup()}");
        return null;
    }
}
