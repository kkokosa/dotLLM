namespace DotLLM.Core.Sampling;

/// <summary>
/// Determines whether token generation should stop.
/// Examples: EOS token, stop sequence match, max length.
/// </summary>
public interface IStopCondition
{
    /// <summary>
    /// Checks whether generation should stop after the given token.
    /// </summary>
    /// <param name="tokenId">The most recently generated token ID.</param>
    /// <param name="generatedTokens">All token IDs generated so far in this sequence.</param>
    /// <param name="decodedText">The full decoded text generated so far.</param>
    /// <returns>Whether to continue, stop (excluding token), or stop (including token).</returns>
    StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText);
}
