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
