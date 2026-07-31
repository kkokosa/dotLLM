using DotLLM.Engine.Evaluation;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

public sealed class CorpusReaderTests
{
    // One token per whitespace-separated word; ids are word lengths, so order is checkable.
    private sealed class WordTokenizer : ITokenizer
    {
        public int[] Encode(string text) =>
            text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(w => w.Length).ToArray();

        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotSupportedException();
        public string DecodeToken(int tokenId) => throw new NotSupportedException();
        public int CountTokens(string text) => Encode(text).Length;
        public int VocabSize => 1024;
        public int BosTokenId => 0;
        public int EosTokenId => 1;
    }

    [Fact]
    public void StreamTokens_ProducesTokensInOrder()
    {
        using var reader = new StringReader("a bb ccc dddd");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer()).ToArray();
        Assert.Equal([1, 2, 3, 4], tokens);
    }

    [Fact]
    public void StreamTokens_HonoursMaxTokens()
    {
        using var reader = new StringReader("a bb ccc dddd eeeee");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer(), maxTokens: 3).ToArray();
        Assert.Equal([1, 2, 3], tokens);
    }

    [Fact]
    public void StreamTokens_DoesNotSplitTokensAcrossChunkBoundaries()
    {
        // A tiny chunk size forces the boundary case: "ccc" must not become "c" + "cc".
        using var reader = new StringReader("a bb ccc dddd eeeee ffffff");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer(), maxTokens: 0, charChunkSize: 4).ToArray();
        Assert.Equal([1, 2, 3, 4, 5, 6], tokens);
    }

    /// <summary>Records the exact strings handed to the tokenizer, so cut points are assertable.</summary>
    private sealed class RecordingTokenizer : ITokenizer
    {
        public List<string> Segments { get; } = [];

        public int[] Encode(string text)
        {
            Segments.Add(text);
            return [text.Length];
        }

        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotSupportedException();
        public string DecodeToken(int tokenId) => throw new NotSupportedException();
        public int CountTokens(string text) => 1;
        public int VocabSize => 1024;
        public int BosTokenId => 0;
        public int EosTokenId => 1;
    }

    [Theory]
    [InlineData("aa   bb")]          // run of spaces
    [InlineData("aa \n\t bb")]       // mixed whitespace run
    [InlineData("aa\n\n\nbb")]       // run of newlines, no spaces at all
    [InlineData("aa\r\nbb")]         // CRLF
    public void StreamTokens_NeverCutsInsideAWhitespaceRun(string text)
    {
        // A GPT-2-style pre-tokenizer treats a whitespace run as one unit. Cutting inside one makes
        // the streamed token stream differ from tokenizing the file in a single pass — a silent
        // divergence that would invalidate every comparison this harness exists to support.
        // Cutting on the last whitespace *character* rather than the run's start does exactly that,
        // and a corpus of newlines with no spaces is not cut at all.
        var tokenizer = new RecordingTokenizer();
        using var reader = new StringReader(text);
        _ = CorpusReader.StreamTokens(reader, tokenizer, maxTokens: 0, charChunkSize: 4).ToArray();

        Assert.Equal(text, string.Concat(tokenizer.Segments));
        Assert.All(tokenizer.Segments, s => Assert.False(
            s.Length > 0 && char.IsWhiteSpace(s[^1]),
            $"segment '{s.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t")}' " +
            "ends in whitespace, so a whitespace run was cut in half"));
    }

    [Fact]
    public void StreamTokens_WhitespaceFreeInputIsEmittedWhole()
    {
        // Documented limitation: with no whitespace there is no safe cut, so the carry accumulates
        // and the corpus is effectively read whole. Asserted so the behaviour is a stated trade
        // rather than a surprise.
        var tokenizer = new RecordingTokenizer();
        using var reader = new StringReader("abcdefghijklmnop");
        _ = CorpusReader.StreamTokens(reader, tokenizer, maxTokens: 0, charChunkSize: 4).ToArray();

        Assert.Single(tokenizer.Segments);
        Assert.Equal("abcdefghijklmnop", tokenizer.Segments[0]);
    }
}
