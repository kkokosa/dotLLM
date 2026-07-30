namespace DotLLM.Diagnostics;

/// <summary>
/// Pure-function numerical helpers used by <see cref="SparseAutoencoder"/> and
/// <see cref="SaeHook"/>. Exposed (internal) for testing — none of these require a
/// loaded model or SAE weights.
/// </summary>
internal static class SaeMath
{
    /// <summary>
    /// Extracts the <paramref name="k"/> highest-magnitude entries from <paramref name="values"/>
    /// in descending order. Magnitude is the raw entry value (post-ReLU non-negative use case).
    /// </summary>
    /// <param name="values">Source vector to query.</param>
    /// <param name="k">Number of entries to return; clamped to <paramref name="values"/>.Length.</param>
    /// <param name="indices">Indices of the top entries, descending by value.</param>
    /// <param name="topValues">Values for <paramref name="indices"/>, parallel array.</param>
    public static void TopK(ReadOnlySpan<float> values, int k, out int[] indices, out float[] topValues)
    {
        int effective = Math.Clamp(k, 0, values.Length);
        indices = new int[effective];
        topValues = new float[effective];
        if (effective == 0) return;

        // Simple full sort — SAE dictionaries are O(1k-65k) features and Encode runs only on
        // diagnostic hook fires (registered explicitly, not on the inference hot path).
        // A min-heap of size K would be O(n log k) and is the obvious follow-up if this
        // ever becomes hot.
        var pairs = new (float V, int I)[values.Length];
        for (int i = 0; i < values.Length; i++) pairs[i] = (values[i], i);
        Array.Sort(pairs, static (a, b) => b.V.CompareTo(a.V));

        for (int i = 0; i < effective; i++)
        {
            topValues[i] = pairs[i].V;
            indices[i] = pairs[i].I;
        }
    }

    /// <summary>
    /// Returns the L2 norm of (<paramref name="a"/> − <paramref name="b"/>).
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector. Must be the same length as <paramref name="a"/>.</param>
    /// <returns>The Euclidean distance between the two vectors.</returns>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    public static float L2Distance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException(
                $"a ({a.Length}) and b ({b.Length}) must have the same length.");

        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            float d = a[i] - b[i];
            sum += (double)d * d;
        }
        return (float)Math.Sqrt(sum);
    }
}
