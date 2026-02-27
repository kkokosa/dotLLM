namespace DotLLM.Core.Constraints;

/// <summary>
/// A stateful constraint that restricts which tokens can be sampled at each decoding step.
/// Used for structured output: JSON schema, regex, grammar-guided generation.
/// </summary>
public interface IDecodingConstraint
{
    /// <summary>
    /// Updates internal state after a token has been sampled.
    /// </summary>
    /// <param name="tokenId">The token ID that was selected.</param>
    void Advance(int tokenId);

    /// <summary>
    /// Returns a bitmask of which tokens are allowed at the current state.
    /// </summary>
    /// <returns>Token mask with allowed tokens set.</returns>
    TokenMask GetAllowedTokens();

    /// <summary>
    /// Whether the constraint is fully satisfied (e.g., JSON object is closed).
    /// </summary>
    /// <returns>True if generation can stop with a valid output.</returns>
    bool IsComplete();

    /// <summary>
    /// Creates a snapshot of the current state for speculative decoding rollback.
    /// </summary>
    /// <returns>A deep copy of this constraint at its current state.</returns>
    IDecodingConstraint Clone();

    /// <summary>
    /// Resets the constraint to its initial state.
    /// </summary>
    void Reset();
}
