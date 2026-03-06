using DotLLM.HuggingFace;
using Spectre.Console;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Resolves a GGUF file argument to a local file path. Accepts either a direct file path,
/// a HuggingFace repo ID (e.g., "Qwen/Qwen3-0.6B-GGUF"), or a repo ID with filename
/// (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q8_0.gguf").
/// </summary>
internal static class GgufFileResolver
{
    /// <summary>
    /// Resolves the argument to a local .gguf file path.
    /// Returns null and prints an error if resolution fails.
    /// </summary>
    /// <param name="fileArg">File path, repo ID, or repo ID with filename.</param>
    /// <param name="quant">Optional quantization filter (e.g., "Q8_0") for disambiguation.</param>
    public static string? Resolve(string fileArg, string? quant = null)
    {
        // Direct file path — use as-is.
        if (File.Exists(fileArg))
            return Path.GetFullPath(fileArg);

        // Check if it looks like a repo ID (contains /).
        if (fileArg.Contains('/'))
        {
            string modelsDir = HuggingFaceDownloader.DefaultModelsDirectory;

            // Try owner/repo/file.gguf form: split into repo ID + filename.
            var resolved = TryResolveRepoFile(modelsDir, fileArg);
            if (resolved is not null)
                return resolved;

            // Standard owner/repo form — look up downloaded models.
            string repoDir = Path.Combine(modelsDir, fileArg.Replace('/', Path.DirectorySeparatorChar));

            if (!Directory.Exists(repoDir))
            {
                AnsiConsole.MarkupLine($"[red]Not found as file or downloaded model:[/] {fileArg.EscapeMarkup()}");
                AnsiConsole.MarkupLine($"[grey]Checked: {repoDir.EscapeMarkup()}[/]");
                return null;
            }

            var ggufFiles = Directory.GetFiles(repoDir, "*.gguf");

            if (ggufFiles.Length == 0)
            {
                AnsiConsole.MarkupLine($"[red]No .gguf files found in:[/] {repoDir.EscapeMarkup()}");
                return null;
            }

            if (ggufFiles.Length == 1)
                return ggufFiles[0];

            // Multiple files — apply --quant filter if provided.
            if (!string.IsNullOrEmpty(quant))
            {
                var filtered = ggufFiles
                    .Where(f => Path.GetFileName(f).Contains(quant, StringComparison.OrdinalIgnoreCase))
                    .ToArray();

                if (filtered.Length == 1)
                    return filtered[0];

                if (filtered.Length == 0)
                {
                    AnsiConsole.MarkupLine($"[red]No .gguf files matching '--quant {quant.EscapeMarkup()}' in {fileArg.EscapeMarkup()}:[/]");
                    foreach (var f in ggufFiles)
                        AnsiConsole.MarkupLine($"  {Path.GetFileName(f).EscapeMarkup()}");
                    return null;
                }

                // Still ambiguous after filter — fall through to disambiguation error with filtered list.
                ggufFiles = filtered;
            }

            // Disambiguation error.
            AnsiConsole.MarkupLine($"[yellow]Multiple .gguf files in {fileArg.EscapeMarkup()}:[/]");
            foreach (var f in ggufFiles)
                AnsiConsole.MarkupLine($"  {Path.GetFileName(f).EscapeMarkup()}");
            AnsiConsole.MarkupLine("[grey]Use --quant <type> to select (e.g., --quant Q8_0).[/]");
            return null;
        }

        AnsiConsole.MarkupLine($"[red]File not found:[/] {fileArg.EscapeMarkup()}");
        return null;
    }

    /// <summary>
    /// Tries to resolve an "owner/repo/file.gguf" path to a local file.
    /// Returns null if the argument doesn't match this form or the file doesn't exist.
    /// </summary>
    private static string? TryResolveRepoFile(string modelsDir, string fileArg)
    {
        // Need at least 3 segments: owner / repo / filename
        var parts = fileArg.Split('/');
        if (parts.Length < 3)
            return null;

        // owner/repo is the first two segments, the rest is the filename (could contain /).
        string owner = parts[0];
        string repo = parts[1];
        string fileName = string.Join(Path.DirectorySeparatorChar.ToString(), parts[2..]);

        string candidatePath = Path.Combine(modelsDir, owner, repo, fileName);
        return File.Exists(candidatePath) ? Path.GetFullPath(candidatePath) : null;
    }
}
