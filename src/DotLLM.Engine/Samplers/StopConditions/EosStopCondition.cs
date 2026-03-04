using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers.StopConditions;

/// <summary>
/// Stops generation when the end-of-sequence token is produced.
/// The EOS token is excluded from the output.
/// </summary>
public sealed class EosStopCondition : IStopCondition
{
    private readonly int _eosTokenId;

    /// <summary>
    /// Creates a new EOS stop condition.
    /// </summary>
    /// <param name="eosTokenId">The end-of-sequence token ID.</param>
    public EosStopCondition(int eosTokenId)
    {
        _eosTokenId = eosTokenId;
    }

    /// <inheritdoc/>
    public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText)
        => tokenId == _eosTokenId ? StopResult.Stop : StopResult.Continue;
}
