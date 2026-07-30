namespace DotLLM.Diagnostics;

/// <summary>
/// Stateless analysis helpers operating over the results returned by
/// <see cref="LogitLensHook.GetResults"/>.
/// </summary>
public static class LogitLensAnalysis
{
    /// <summary>
    /// Returns the index of the earliest layer (in ascending layer order, filtered to a
    /// single token position when <paramref name="tokenPosition"/> is specified) whose
    /// top-1 prediction equals <paramref name="targetTokenId"/>, or <c>null</c> when no
    /// such layer exists.
    /// </summary>
    /// <param name="results">Logit-lens results returned by <see cref="LogitLensHook.GetResults"/>.</param>
    /// <param name="targetTokenId">Token id to look for as the top-1 prediction.</param>
    /// <param name="tokenPosition">Optional token-position filter (commonly the final prompt position).</param>
    public static int? ConvergenceLayer(
        IReadOnlyList<LogitLensResult> results,
        int targetTokenId,
        int? tokenPosition = null)
    {
        ArgumentNullException.ThrowIfNull(results);

        int? earliest = null;
        foreach (var r in results)
        {
            if (tokenPosition is int pos && r.TokenPosition != pos) continue;
            if (r.TopKTokens.Length == 0) continue;
            if (r.TopKTokens[0] != targetTokenId) continue;

            if (earliest is null || r.LayerIndex < earliest.Value)
                earliest = r.LayerIndex;
        }
        return earliest;
    }

    /// <summary>
    /// Returns the per-layer probability of <paramref name="targetTokenId"/> at
    /// <paramref name="tokenPosition"/>, ordered by ascending layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// If <see cref="LogitLensResult.FullProbabilities"/> is available the exact probability
    /// is reported. Otherwise the value is read from <see cref="LogitLensResult.TopKProbabilities"/>
    /// when the token appears in the top-K; tokens outside the top-K with no full distribution
    /// recorded report <c>0f</c> — set <see cref="LogitLensConfig.StoreFullProbabilities"/>
    /// to <c>true</c> for accurate readings on rare tokens.
    /// </para>
    /// </remarks>
    /// <param name="results">Logit-lens results.</param>
    /// <param name="targetTokenId">Token whose confidence trajectory to extract.</param>
    /// <param name="tokenPosition">Token position to filter on.</param>
    public static IReadOnlyList<(int Layer, float Probability)> ConfidenceAcrossLayers(
        IReadOnlyList<LogitLensResult> results,
        int targetTokenId,
        int tokenPosition)
    {
        ArgumentNullException.ThrowIfNull(results);

        var output = new List<(int Layer, float Probability)>();
        foreach (var r in results
                     .Where(r => r.TokenPosition == tokenPosition)
                     .OrderBy(r => r.LayerIndex))
        {
            float p = 0f;
            if (r.FullProbabilities is not null)
            {
                if ((uint)targetTokenId < (uint)r.FullProbabilities.Length)
                    p = r.FullProbabilities[targetTokenId];
            }
            else
            {
                for (int i = 0; i < r.TopKTokens.Length; i++)
                {
                    if (r.TopKTokens[i] == targetTokenId)
                    {
                        p = r.TopKProbabilities[i];
                        break;
                    }
                }
            }
            output.Add((r.LayerIndex, p));
        }
        return output;
    }

    /// <summary>
    /// Returns the rank of <paramref name="targetTokenId"/> in <paramref name="result"/>'s
    /// distribution (0 = top-1). Requires <see cref="LogitLensResult.FullProbabilities"/>
    /// to be available for tokens outside the top-K; returns <c>null</c> when the token is
    /// outside the top-K and no full distribution was recorded.
    /// </summary>
    /// <param name="result">A single per-layer lens result.</param>
    /// <param name="targetTokenId">Token whose rank to look up.</param>
    public static int? RankOf(LogitLensResult result, int targetTokenId)
    {
        ArgumentNullException.ThrowIfNull(result);

        // Always check top-K first — cheap and avoids needing the full distribution
        // when the target is one of the top-K predictions.
        for (int i = 0; i < result.TopKTokens.Length; i++)
            if (result.TopKTokens[i] == targetTokenId) return i;

        if (result.FullProbabilities is null) return null;

        float pTarget = (uint)targetTokenId < (uint)result.FullProbabilities.Length
            ? result.FullProbabilities[targetTokenId]
            : 0f;

        int rank = 0;
        for (int i = 0; i < result.FullProbabilities.Length; i++)
            if (result.FullProbabilities[i] > pTarget) rank++;

        return rank;
    }
}
