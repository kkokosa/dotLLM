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

    /// <summary>
    /// Span-based overload that avoids materializing a full <see cref="string"/> when the caller
    /// maintains the decoded text incrementally. Defaults to the string overload; implementations
    /// that can operate on a span (such as stop-string suffix checks) should override this.
    /// </summary>
    /// <param name="tokenId">The most recently generated token ID.</param>
    /// <param name="generatedTokens">All token IDs generated so far in this sequence.</param>
    /// <param name="decodedTail">A trailing view of the decoded text — large enough to cover the
    /// longest expected stop pattern. Implementations must not assume the view covers the full
    /// sequence.</param>
    StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, ReadOnlySpan<char> decodedTail)
        => ShouldStop(tokenId, generatedTokens, decodedTail.ToString());
}
