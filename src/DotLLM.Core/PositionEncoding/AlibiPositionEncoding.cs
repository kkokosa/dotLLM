using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Core.PositionEncoding;

/// <summary>
/// Attention with Linear Biases (ALiBi) position encoding metadata.
/// ALiBi does not mutate Q/K tensors; attention kernels consume <see cref="Slopes"/>
/// and add <c>-slope[h] * (queryPosition - keyPosition)</c> to each visible score.
/// </summary>
public sealed class AlibiPositionEncoding : IPositionEncoding
{
    private float[]? _slopes;

    /// <summary>Per-query-head ALiBi slopes.</summary>
    public ReadOnlySpan<float> Slopes => _slopes;

    /// <summary>
    /// Creates standard ALiBi slopes for <paramref name="numHeads"/>.
    /// Matches the BLOOM/MPT/Falcon-legacy fixed slope schedule.
    /// </summary>
    public static float[] CreateSlopes(int numHeads)
    {
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads));

        var slopes = new float[numHeads];
        int powerOfTwo = PreviousPowerOfTwo(numHeads);
        FillPowerOfTwoSlopes(slopes.AsSpan(0, powerOfTwo), powerOfTwo);

        if (powerOfTwo != numHeads)
        {
            int expanded = powerOfTwo * 2;
            var extra = new float[expanded];
            FillPowerOfTwoSlopes(extra, expanded);

            int dst = powerOfTwo;
            for (int i = 0; dst < numHeads; i += 2)
                slopes[dst++] = extra[i];
        }

        return slopes;
    }

    /// <inheritdoc/>
    public void PrecomputeTables(int maxSeqLen, ModelConfig config)
    {
        if (maxSeqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _slopes = CreateSlopes(config.NumAttentionHeads);
    }

    /// <inheritdoc/>
    public (ITensor Q, ITensor K) Apply(ITensor q, ITensor k, ReadOnlySpan<int> positions)
    {
        if (_slopes is null)
            throw new InvalidOperationException("PrecomputeTables must be called before Apply.");

        // ALiBi is applied inside attention scores, not by rotating or adding to Q/K.
        return (q, k);
    }

    /// <inheritdoc/>
    public void InvalidateCache() => _slopes = null;

    private static int PreviousPowerOfTwo(int value)
    {
        int power = 1;
        while (power <= value / 2)
            power <<= 1;
        return power;
    }

    private static void FillPowerOfTwoSlopes(Span<float> slopes, int numHeads)
    {
        double start = Math.Pow(2.0, -Math.Pow(2.0, -(Math.Log2(numHeads) - 3.0)));
        double ratio = start;
        for (int i = 0; i < slopes.Length; i++)
        {
            slopes[i] = (float)start;
            start *= ratio;
        }
    }
}
