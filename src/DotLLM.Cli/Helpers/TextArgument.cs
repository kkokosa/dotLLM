namespace DotLLM.Cli.Helpers;

/// <summary>
/// Resolves text options that may be supplied either inline or loaded from a file
/// (e.g. <c>--prompt</c> / <c>--prompt-file</c>). The two forms are mutually exclusive.
/// </summary>
/// <remarks>
/// Distinct from the <c>@file</c> convention used by <c>--schema</c>, <c>--grammar</c>, and
/// <c>--tools</c>: those carry JSON/GBNF, which never begins with <c>@</c>. Prompts are free
/// text that plausibly does, so a separate option is used instead of an in-band prefix.
/// </remarks>
internal static class TextArgument
{
    /// <summary>
    /// Resolves an inline value and a file path into a single text value.
    /// </summary>
    /// <param name="inline">Inline option value, or null/empty when not supplied.</param>
    /// <param name="filePath">File option value, or null/empty when not supplied.</param>
    /// <param name="inlineOption">Display name of the inline option, used in error messages.</param>
    /// <param name="fileOption">Display name of the file option, used in error messages.</param>
    /// <param name="required">When true, omitting both options is an error.</param>
    /// <param name="value">Resolved text. Null when neither option was supplied and
    /// <paramref name="required"/> is false.</param>
    /// <param name="error">Human-readable failure reason. Null on success.</param>
    /// <returns>True when resolution succeeded; false when <paramref name="error"/> is set.</returns>
    public static bool TryResolve(
        string? inline,
        string? filePath,
        string inlineOption,
        string fileOption,
        bool required,
        out string? value,
        out string? error)
    {
        value = null;
        error = null;

        bool hasInline = !string.IsNullOrEmpty(inline);
        bool hasFile = !string.IsNullOrEmpty(filePath);

        if (hasInline && hasFile)
        {
            error = $"{inlineOption} and {fileOption} are mutually exclusive.";
            return false;
        }

        if (!hasInline && !hasFile)
        {
            if (required)
            {
                error = $"{inlineOption} or {fileOption} is required.";
                return false;
            }
            return true;
        }

        if (hasInline)
        {
            value = inline;
            return true;
        }

        string text;
        try
        {
            text = File.ReadAllText(filePath!);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                      or ArgumentException or NotSupportedException)
        {
            error = $"Cannot read {fileOption} '{filePath}': {ex.Message}";
            return false;
        }

        text = StripSingleTrailingNewline(text);

        if (text.Length == 0)
        {
            error = $"{fileOption} '{filePath}' is empty.";
            return false;
        }

        value = text;
        return true;
    }

    /// <summary>
    /// Removes at most one trailing newline (<c>\n</c> or <c>\r\n</c>). Editors append a final
    /// newline that the author rarely intends as part of the prompt, and it changes tokenization.
    /// Matches <c>llama.cpp --file</c> behaviour. Interior newlines are preserved, as is a
    /// deliberate blank line at the end (only one newline is removed).
    /// </summary>
    internal static string StripSingleTrailingNewline(string text)
    {
        if (text.EndsWith("\r\n", StringComparison.Ordinal))
            return text[..^2];
        if (text.EndsWith('\n'))
            return text[..^1];
        return text;
    }
}
