namespace DotLLM.Diagnostics;

/// <summary>
/// Pure-function numerical helpers used by <see cref="LogitLensHook"/>. Exposed (internal)
/// for testing — none of these need a loaded model.
/// </summary>
internal static class LogitLensMath
{
    /// <summary>
    /// Numerically stable softmax (subtract-max). May operate in-place when
    /// <paramref name="logits"/> and <paramref name="output"/> alias.
    /// </summary>
    /// <param name="logits">Input logits.</param>
    /// <param name="output">Destination probability buffer; must be at least <paramref name="logits"/>.Length.</param>
    public static void Softmax(ReadOnlySpan<float> logits, Span<float> output)
    {
        if (output.Length < logits.Length)
            throw new ArgumentException("output must be at least as long as logits.", nameof(output));

        float maxLogit = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > maxLogit) maxLogit = logits[i];

        // Edge case: empty / all-(-inf) input → produce a uniform distribution if anything.
        if (float.IsNegativeInfinity(maxLogit) || logits.Length == 0)
        {
            float uniform = logits.Length == 0 ? 0f : 1.0f / logits.Length;
            for (int i = 0; i < logits.Length; i++) output[i] = uniform;
            return;
        }

        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            float e = MathF.Exp(logits[i] - maxLogit);
            output[i] = e;
            sum += e;
        }

        float inv = (float)(1.0 / sum);
        for (int i = 0; i < logits.Length; i++) output[i] *= inv;
    }

    /// <summary>
    /// Returns the Shannon entropy of a probability distribution, in nats.
    /// Entries less than or equal to zero contribute zero.
    /// </summary>
    /// <param name="probabilities">A distribution that sums to approximately 1.</param>
    public static float Entropy(ReadOnlySpan<float> probabilities)
    {
        double h = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            float p = probabilities[i];
            if (p > 0f) h -= p * MathF.Log(p);
        }
        return (float)h;
    }

    /// <summary>
    /// Extracts the <paramref name="k"/> highest-probability entries from
    /// <paramref name="probabilities"/> in descending order.
    /// </summary>
    /// <param name="probabilities">Distribution to query.</param>
    /// <param name="k">Number of entries to return; clamped to <paramref name="probabilities"/>.Length.</param>
    /// <param name="indices">Indices of the top entries, descending by probability.</param>
    /// <param name="values">Probabilities for <paramref name="indices"/>, parallel array.</param>
    public static void TopK(ReadOnlySpan<float> probabilities, int k,
        out int[] indices, out float[] values)
    {
        int effective = Math.Clamp(k, 0, probabilities.Length);
        indices = new int[effective];
        values = new float[effective];
        if (effective == 0) return;

        // Simple partial selection — vocab sizes are O(50k-150k) and we run this once per
        // (layer, position) on retrieval; not on the hot path. Heap could be added later.
        var pairs = new (float P, int I)[probabilities.Length];
        for (int i = 0; i < probabilities.Length; i++) pairs[i] = (probabilities[i], i);
        Array.Sort(pairs, static (a, b) => b.P.CompareTo(a.P));

        for (int i = 0; i < effective; i++)
        {
            values[i] = pairs[i].P;
            indices[i] = pairs[i].I;
        }
    }
}
