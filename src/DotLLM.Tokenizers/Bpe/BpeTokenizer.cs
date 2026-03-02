namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// Byte-pair encoding tokenizer supporting SentencePiece and tiktoken variants.
/// Vocabulary data is loaded at construction; all encoding and decoding is delegated
/// to the variant-specific <see cref="IBpeEncoding"/> implementation.
/// To add a new variant, implement <see cref="IBpeEncoding"/> and add a factory method here —
/// no modifications to existing code are required.
/// </summary>
public sealed class BpeTokenizer : ITokenizer
{
    private readonly IBpeEncoding _encoding;

    /// <inheritdoc/>
    public int BosTokenId { get; }

    /// <inheritdoc/>
    public int EosTokenId { get; }

    /// <inheritdoc/>
    public int VocabSize { get; }

    private BpeTokenizer(IBpeEncoding encoding, int bosId, int eosId, int vocabSize)
    {
        _encoding = encoding;
        BosTokenId = bosId;
        EosTokenId = eosId;
        VocabSize = vocabSize;
    }

    /// <summary>
    /// Creates a SentencePiece BPE tokenizer (Llama 1/2, Mistral, TinyLlama, SmolLM).
    /// </summary>
    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="scores">Unigram log-probability scores (higher = preferred merge).</param>
    /// <param name="tokenTypes">Per-token type flags (0=normal, 1=unknown, 2=control, 3=byte, 5=user-defined). Null = all normal.</param>
    /// <param name="bosId">Beginning-of-sequence token ID.</param>
    /// <param name="eosId">End-of-sequence token ID.</param>
    /// <param name="addBosSpace">Prepend ▁ to text that doesn't start with a space (matches SentencePiece default).</param>
    public static BpeTokenizer CreateSentencePiece(
        string[] tokens, float[] scores, int[]? tokenTypes,
        int bosId, int eosId, bool addBosSpace = true)
    {
        float[] safeScores = scores.Length == tokens.Length ? scores : new float[tokens.Length];
        return new BpeTokenizer(
            new SentencePieceEncoding(tokens, safeScores, tokenTypes, addBosSpace),
            bosId, eosId, tokens.Length);
    }

    /// <summary>
    /// Creates a tiktoken BPE tokenizer (Llama 3, GPT-4).
    /// </summary>
    /// <param name="tokens">Vocabulary strings indexed by token ID.</param>
    /// <param name="merges">Merge table entries in "A B" format; index = rank (lower = applied first).</param>
    /// <param name="tokenTypes">Per-token type flags. Null = all normal.</param>
    /// <param name="bosId">Beginning-of-sequence token ID.</param>
    /// <param name="eosId">End-of-sequence token ID.</param>
    public static BpeTokenizer CreateTiktoken(
        string[] tokens, string[] merges, int[]? tokenTypes, int bosId, int eosId)
        => new(new Gpt2TiktokenEncoding(tokens, merges, tokenTypes), bosId, eosId, tokens.Length);

    /// <inheritdoc/>
    public int[] Encode(string text) => text.Length == 0 ? [] : _encoding.Encode(text);

    /// <inheritdoc/>
    public string Decode(ReadOnlySpan<int> tokenIds) =>
        tokenIds.IsEmpty ? string.Empty : _encoding.Decode(tokenIds);

    /// <inheritdoc/>
    public string DecodeToken(int tokenId) => _encoding.DecodeToken(tokenId);

    /// <inheritdoc/>
    public int CountTokens(string text) => Encode(text).Length;
}
