namespace DotLLM.Tokenizers;

/// <summary>
/// Tokenizer that encodes text to token IDs and decodes token IDs back to text.
/// </summary>
public interface ITokenizer
{
    /// <summary>Encodes text into token IDs.</summary>
    /// <param name="text">Input text to tokenize.</param>
    /// <returns>Array of token IDs.</returns>
    int[] Encode(string text);

    /// <summary>Decodes a sequence of token IDs back to text.</summary>
    /// <param name="tokenIds">Token IDs to decode.</param>
    /// <returns>Decoded text.</returns>
    string Decode(ReadOnlySpan<int> tokenIds);

    /// <summary>
    /// Decodes a sequence of token IDs back to text, optionally preserving the leading space
    /// that SentencePiece tokenizers normally strip (the inverse of BOS ▁ prepending).
    /// Use <paramref name="stripBosSpace"/> = <c>false</c> when decoding generated continuation
    /// tokens that were NOT encoded with BOS space prepending.
    /// </summary>
    /// <param name="tokenIds">Token IDs to decode.</param>
    /// <param name="stripBosSpace">When true (default), strips the leading space introduced by BOS ▁ prepending.</param>
    /// <returns>Decoded text.</returns>
    string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);

    /// <summary>
    /// Attempts to decode a sequence of token IDs into the caller-provided destination span
    /// without allocating an intermediate <see cref="string"/>. Returns <see langword="true"/>
    /// when the decoded text fit in <paramref name="destination"/> (with the character count
    /// written to <paramref name="charsWritten"/>), or <see langword="false"/> when the
    /// destination was too small.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The default interface method forwards to <see cref="Decode(ReadOnlySpan{int}, bool)"/>
    /// and copies into <paramref name="destination"/> — providing the surface for every
    /// <see cref="ITokenizer"/> implementation, but only the implementations that override
    /// this method gain the zero-allocation benefit. <see cref="DotLLM.Tokenizers.Bpe.BpeTokenizer"/>
    /// overrides it for both the tiktoken (Llama-3, GPT-4, Qwen-2) and SentencePiece (Llama-1/2,
    /// Mistral, TinyLlama) encodings.
    /// </para>
    /// <para>
    /// On a <see langword="false"/> return, the contents of <paramref name="destination"/>
    /// are unspecified — callers must either retry with a larger buffer or fall back to
    /// the allocating <see cref="Decode(ReadOnlySpan{int}, bool)"/> overload.
    /// </para>
    /// </remarks>
    /// <param name="tokenIds">Token IDs to decode.</param>
    /// <param name="stripBosSpace">When <see langword="true"/>, strips the leading space introduced by SentencePiece BOS ▁ prepending.</param>
    /// <param name="destination">Destination character buffer. May be empty when <paramref name="tokenIds"/> is empty.</param>
    /// <param name="charsWritten">Number of characters written to <paramref name="destination"/> on success; <c>0</c> on failure.</param>
    /// <returns><see langword="true"/> if the decoded text fit; otherwise <see langword="false"/>.</returns>
    bool TryDecode(ReadOnlySpan<int> tokenIds, bool stripBosSpace, Span<char> destination, out int charsWritten)
    {
        string decoded = Decode(tokenIds, stripBosSpace);
        if (decoded.Length > destination.Length)
        {
            charsWritten = 0;
            return false;
        }
        decoded.AsSpan().CopyTo(destination);
        charsWritten = decoded.Length;
        return true;
    }

    /// <summary>Decodes a single token ID to its string representation.</summary>
    /// <param name="tokenId">Token ID to decode.</param>
    /// <returns>String representation of the token.</returns>
    string DecodeToken(int tokenId);

    /// <summary>Total vocabulary size.</summary>
    int VocabSize { get; }

    /// <summary>Beginning-of-sequence token ID.</summary>
    int BosTokenId { get; }

    /// <summary>End-of-sequence token ID.</summary>
    int EosTokenId { get; }

    /// <summary>
    /// Counts the number of tokens without performing a full encode.
    /// May be approximate for some tokenizer implementations.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <returns>Token count.</returns>
    int CountTokens(string text);
}
