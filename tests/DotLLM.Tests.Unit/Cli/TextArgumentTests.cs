using DotLLM.Cli.Helpers;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>
/// Covers <see cref="TextArgument.TryResolve"/>, which backs the mutually exclusive
/// <c>--prompt</c>/<c>--prompt-file</c> and <c>--system</c>/<c>--system-file</c> option pairs.
/// </summary>
public sealed class TextArgumentTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    private string WriteTemp(string content)
    {
        string path = Path.Combine(Path.GetTempPath(), $"dotllm-textarg-{Guid.NewGuid():N}.txt");
        File.WriteAllText(path, content);
        _tempFiles.Add(path);
        return path;
    }

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try { File.Delete(path); } catch (IOException) { /* best effort */ }
        }
    }

    [Fact]
    public void InlineValue_IsReturnedVerbatim()
    {
        bool ok = TextArgument.TryResolve("hello world", null, "--prompt", "--prompt-file",
            required: true, out string? value, out string? error);

        Assert.True(ok);
        Assert.Null(error);
        Assert.Equal("hello world", value);
    }

    [Fact]
    public void InlineValue_StartingWithAt_IsNotTreatedAsPath()
    {
        // The whole point of a separate --prompt-file option: '@' stays literal.
        bool ok = TextArgument.TryResolve("@everyone: hi", null, "--prompt", "--prompt-file",
            required: true, out string? value, out string? error);

        Assert.True(ok);
        Assert.Null(error);
        Assert.Equal("@everyone: hi", value);
    }

    [Fact]
    public void FilePath_ReturnsFileContents()
    {
        string path = WriteTemp("The capital of France is");

        bool ok = TextArgument.TryResolve(null, path, "--prompt", "--prompt-file",
            required: true, out string? value, out string? error);

        Assert.True(ok);
        Assert.Null(error);
        Assert.Equal("The capital of France is", value);
    }

    [Fact]
    public void BothSupplied_IsAnError()
    {
        bool ok = TextArgument.TryResolve("inline", "some/path.txt", "--prompt", "--prompt-file",
            required: true, out string? value, out string? error);

        Assert.False(ok);
        Assert.Null(value);
        Assert.Contains("mutually exclusive", error);
    }

    [Fact]
    public void NeitherSupplied_IsAnError_WhenRequired()
    {
        bool ok = TextArgument.TryResolve(null, null, "--prompt", "--prompt-file",
            required: true, out string? value, out string? error);

        Assert.False(ok);
        Assert.Null(value);
        Assert.Contains("required", error);
    }

    [Fact]
    public void NeitherSupplied_IsOk_WhenOptional()
    {
        bool ok = TextArgument.TryResolve(null, null, "--system", "--system-file",
            required: false, out string? value, out string? error);

        Assert.True(ok);
        Assert.Null(value);
        Assert.Null(error);
    }

    [Fact]
    public void MissingFile_IsAnError()
    {
        string missing = Path.Combine(Path.GetTempPath(), $"dotllm-missing-{Guid.NewGuid():N}.txt");

        bool ok = TextArgument.TryResolve(null, missing, "--prompt", "--prompt-file",
            required: true, out string? value, out string? error);

        Assert.False(ok);
        Assert.Null(value);
        Assert.Contains("--prompt-file", error);
    }

    [Fact]
    public void EmptyFile_IsAnError()
    {
        string path = WriteTemp("");

        bool ok = TextArgument.TryResolve(null, path, "--prompt", "--prompt-file",
            required: true, out string? value, out string? error);

        Assert.False(ok);
        Assert.Null(value);
        Assert.Contains("empty", error);
    }

    [Fact]
    public void FileContainingOnlyANewline_IsTreatedAsEmpty()
    {
        string path = WriteTemp("\n");

        bool ok = TextArgument.TryResolve(null, path, "--prompt", "--prompt-file",
            required: true, out _, out string? error);

        Assert.False(ok);
        Assert.Contains("empty", error);
    }

    [Theory]
    [InlineData("abc\n", "abc")]
    [InlineData("abc\r\n", "abc")]
    [InlineData("abc", "abc")]
    [InlineData("abc\n\n", "abc\n")]          // only ONE newline is stripped
    [InlineData("a\nb\n", "a\nb")]            // interior newlines preserved
    [InlineData("a\r\nb\r\n", "a\r\nb")]
    public void StripSingleTrailingNewline_RemovesAtMostOne(string input, string expected)
    {
        Assert.Equal(expected, TextArgument.StripSingleTrailingNewline(input));
    }

    [Fact]
    public void FileWithTrailingNewline_HasItStripped()
    {
        // Editors append a final newline; it changes tokenization, so it is removed.
        string path = WriteTemp("The capital of France is\n");

        bool ok = TextArgument.TryResolve(null, path, "--prompt", "--prompt-file",
            required: true, out string? value, out _);

        Assert.True(ok);
        Assert.Equal("The capital of France is", value);
    }

    [Fact]
    public void MultiLineFile_PreservesInteriorNewlines()
    {
        string path = WriteTemp("line one\nline two\nline three\n");

        bool ok = TextArgument.TryResolve(null, path, "--prompt", "--prompt-file",
            required: true, out string? value, out _);

        Assert.True(ok);
        Assert.Equal("line one\nline two\nline three", value);
    }

    [Fact]
    public void EmptyStringInline_IsTreatedAsNotSupplied()
    {
        // Spectre binds an omitted option to null, but guard against "" defensively.
        bool ok = TextArgument.TryResolve("", null, "--system", "--system-file",
            required: false, out string? value, out string? error);

        Assert.True(ok);
        Assert.Null(value);
        Assert.Null(error);
    }
}
