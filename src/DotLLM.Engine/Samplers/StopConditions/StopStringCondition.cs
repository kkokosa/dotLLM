using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers.StopConditions;

/// <summary>
/// Stops generation when the decoded text ends with a specified stop string.
/// The stop string is excluded from the output.
/// </summary>
public sealed class StopStringCondition : IStopCondition
{
    private readonly string _stopString;

    /// <summary>
    /// Creates a new stop string condition.
    /// </summary>
    /// <param name="stopString">The string that triggers generation stop.</param>
    public StopStringCondition(string stopString)
    {
        _stopString = stopString;
    }

    /// <inheritdoc/>
    public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText)
        => decodedText.EndsWith(_stopString, StringComparison.Ordinal)
            ? StopResult.Stop
            : StopResult.Continue;
}
