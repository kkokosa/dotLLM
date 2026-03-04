using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers.StopConditions;

/// <summary>
/// Stops generation when the maximum number of tokens has been reached.
/// The final token is included in the output.
/// </summary>
public sealed class MaxTokensStopCondition : IStopCondition
{
    private readonly int _maxTokens;

    /// <summary>
    /// Creates a new max tokens stop condition.
    /// </summary>
    /// <param name="maxTokens">Maximum number of tokens to generate.</param>
    public MaxTokensStopCondition(int maxTokens)
    {
        _maxTokens = maxTokens;
    }

    /// <inheritdoc/>
    public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText)
        => generatedTokens.Count >= _maxTokens ? StopResult.StopInclude : StopResult.Continue;
}
