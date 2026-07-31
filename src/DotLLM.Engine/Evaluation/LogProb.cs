namespace DotLLM.Engine.Evaluation;

/// <summary>Numerically stable log-softmax over a single row of logits.</summary>
public static class LogProb
{
    /// <summary>
    /// Returns <c>log P(target)</c> in nats under a softmax over <paramref name="logits"/>.
    /// </summary>
    /// <remarks>
    /// Uses the max-shift identity <c>log softmax(x)_t = (x_t - m) - log sum_j exp(x_j - m)</c>
    /// with <c>m = max(x)</c>, so no <c>exp</c> argument is ever positive and overflow is
    /// impossible. Accumulates in <see cref="double"/>: a vocab of 128k float32 terms loses
    /// meaningful precision in float32, and perplexity differences between near-identical runs
    /// are exactly what this harness exists to resolve.
    /// </remarks>
    /// <param name="logits">One row of unnormalized scores.</param>
    /// <param name="target">Index whose log-probability is returned.</param>
    public static double OfTarget(ReadOnlySpan<float> logits, int target)
    {
        if ((uint)target >= (uint)logits.Length)
            throw new ArgumentOutOfRangeException(nameof(target));

        float max = logits[0];
        for (int j = 1; j < logits.Length; j++)
            if (logits[j] > max) max = logits[j];

        double sumExp = 0;
        for (int j = 0; j < logits.Length; j++)
            sumExp += Math.Exp(logits[j] - max);

        return (logits[target] - max) - Math.Log(sumExp);
    }
}
