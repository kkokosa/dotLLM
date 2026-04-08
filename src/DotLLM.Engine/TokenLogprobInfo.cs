namespace DotLLM.Engine;

/// <summary>
/// Log-probability information for a single generated token.
/// Includes the chosen token's logprob and optionally the top-K alternatives.
/// </summary>
/// <param name="TokenId">The sampled token ID.</param>
/// <param name="Token">The decoded token string.</param>
/// <param name="Logprob">Log-probability of the chosen token (natural log).</param>
/// <param name="Bytes">UTF-8 byte representation of the token.</param>
/// <param name="TopLogprobs">Top-K alternative tokens ranked by logprob descending. Null when top_logprobs=0.</param>
public readonly record struct TokenLogprobInfo(
    int TokenId,
    string Token,
    float Logprob,
    byte[]? Bytes,
    TopLogprobEntry[]? TopLogprobs);

/// <summary>
/// A single entry in the top-K logprobs list.
/// </summary>
/// <param name="TokenId">The token ID.</param>
/// <param name="Token">The decoded token string.</param>
/// <param name="Logprob">Log-probability of this token (natural log).</param>
/// <param name="Bytes">UTF-8 byte representation of the token.</param>
public readonly record struct TopLogprobEntry(
    int TokenId,
    string Token,
    float Logprob,
    byte[]? Bytes);
