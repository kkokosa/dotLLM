using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Tokenizers;

/// <summary>
/// Integration tests for <see cref="BpeTokenizer"/> against the real SmolLM-135M Q8_0 GGUF
/// (SentencePiece tokenizer, vocab_size=49152). Uses the shared <see cref="SmallModelFixture"/>
/// so no additional downloads are needed.
/// </summary>
[Collection("SmallModel")]
public class BpeTokenizerIntegrationTests
{
    private readonly SmallModelFixture _fixture;

    public BpeTokenizerIntegrationTests(SmallModelFixture fixture)
    {
        _fixture = fixture;
    }

    private BpeTokenizer LoadTokenizer()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        return GgufBpeTokenizerFactory.Load(gguf.Metadata);
    }

    // -------------------------------------------------------------------------
    // Loading & metadata
    // -------------------------------------------------------------------------

    [Fact]
    public void Load_SmolLM_ReturnsTokenizer()
    {
        BpeTokenizer tok = LoadTokenizer();
        Assert.NotNull(tok);
    }

    [Fact]
    public void VocabSize_MatchesGgufMetadata()
    {
        using var gguf = GgufFile.Open(_fixture.FilePath);
        string[] tokens = gguf.Metadata.GetStringArray("tokenizer.ggml.tokens");
        BpeTokenizer tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        Assert.Equal(tokens.Length, tok.VocabSize);
    }

    [Fact]
    public void BosEosTokenIds_AreWithinVocabRange()
    {
        BpeTokenizer tok = LoadTokenizer();
        Assert.InRange(tok.BosTokenId, 0, tok.VocabSize - 1);
        Assert.InRange(tok.EosTokenId, 0, tok.VocabSize - 1);
    }

    // -------------------------------------------------------------------------
    // Encode smoke tests
    // -------------------------------------------------------------------------

    [Fact]
    public void Encode_HelloWorld_ReturnsNonEmptyArray()
    {
        BpeTokenizer tok = LoadTokenizer();
        int[] ids = tok.Encode("Hello world");
        Assert.NotEmpty(ids);
    }

    [Fact]
    public void Encode_EmptyString_ReturnsEmpty()
    {
        BpeTokenizer tok = LoadTokenizer();
        Assert.Empty(tok.Encode(string.Empty));
    }

    // -------------------------------------------------------------------------
    // Roundtrip tests
    // -------------------------------------------------------------------------

    [Fact]
    public void Roundtrip_AsciiText()
    {
        BpeTokenizer tok = LoadTokenizer();
        const string text = "Hello world";
        Assert.Equal(text, tok.Decode(tok.Encode(text)));
    }

    [Fact]
    public void Roundtrip_MultiWordSentence()
    {
        BpeTokenizer tok = LoadTokenizer();
        const string text = "The quick brown fox jumps over the lazy dog.";
        Assert.Equal(text, tok.Decode(tok.Encode(text)));
    }

    [Fact]
    public void Roundtrip_Unicode()
    {
        BpeTokenizer tok = LoadTokenizer();
        const string text = "café au lait";
        Assert.Equal(text, tok.Decode(tok.Encode(text)));
    }

    [Fact]
    public void Roundtrip_NumbersAndPunctuation()
    {
        BpeTokenizer tok = LoadTokenizer();
        const string text = "1 + 1 = 2";
        Assert.Equal(text, tok.Decode(tok.Encode(text)));
    }
}
