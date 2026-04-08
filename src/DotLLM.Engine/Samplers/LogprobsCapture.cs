using System.Buffers;
using System.Numerics.Tensors;
using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Captures log-probability information from raw logits before the sampler pipeline modifies them.
/// Two-phase design: <see cref="ComputeLogSoftmax"/> runs before sampling (logits still intact),
/// then <see cref="BuildInfo"/> runs after the token is selected.
/// </summary>
public static class LogprobsCapture
{
    /// <summary>
    /// Computes log-softmax over the logit span and returns the rented buffer.
    /// Caller must return the buffer to <see cref="ArrayPool{T}.Shared"/> after use.
    /// log_softmax(x_i) = x_i - log(sum(exp(x_j - max(x)))).
    /// </summary>
    /// <param name="logits">Raw logits (after constraint masking, before sampling).</param>
    /// <returns>A rented float array containing log-softmax values. Length >= logits.Length.</returns>
    public static float[] ComputeLogSoftmax(ReadOnlySpan<float> logits)
    {
        int vocabSize = logits.Length;
        float[] buffer = ArrayPool<float>.Shared.Rent(vocabSize);
        Span<float> output = buffer.AsSpan(0, vocabSize);

        // Numerically stable log-softmax: x_i - max - log(sum(exp(x_j - max)))
        float max = TensorPrimitives.Max(logits);
        TensorPrimitives.Subtract(logits, max, output); // vectorized x_i - max
        float sumExp = 0f;
        for (int i = 0; i < vocabSize; i++)
            sumExp += MathF.Exp(output[i]);
        float logSumExp = MathF.Log(sumExp);
        TensorPrimitives.Subtract((ReadOnlySpan<float>)output, logSumExp, output); // vectorized final subtract

        return buffer;
    }

    /// <summary>
    /// Builds a <see cref="TokenLogprobInfo"/> from pre-computed log-softmax values.
    /// </summary>
    /// <param name="logSoftmax">Log-softmax values (from <see cref="ComputeLogSoftmax"/>).</param>
    /// <param name="vocabSize">Actual vocabulary size (logSoftmax buffer may be larger).</param>
    /// <param name="sampledTokenId">The token selected by sampling.</param>
    /// <param name="topK">Number of top alternatives to include (0 = none).</param>
    /// <param name="tokenizer">Tokenizer for decoding token IDs to strings.</param>
    /// <returns>Log-probability information for the sampled token.</returns>
    public static TokenLogprobInfo BuildInfo(
        ReadOnlySpan<float> logSoftmax, int vocabSize,
        int sampledTokenId, int topK, ITokenizer tokenizer)
    {
        float sampledLogprob = logSoftmax[sampledTokenId];
        string sampledToken = tokenizer.Decode([sampledTokenId], stripBosSpace: false);
        byte[] sampledBytes = Encoding.UTF8.GetBytes(sampledToken);

        TopLogprobEntry[]? topEntries = topK > 0
            ? ExtractTopK(logSoftmax, vocabSize, topK, tokenizer)
            : null;

        return new TokenLogprobInfo(
            sampledTokenId, sampledToken, sampledLogprob, sampledBytes, topEntries);
    }

    /// <summary>
    /// Extracts the top-K tokens by log-probability using a simple insertion-sort approach.
    /// For K &lt;= 20 over vocab sizes of 32K-128K, linear scan with sorted insertion
    /// is faster than a heap.
    /// </summary>
    private static TopLogprobEntry[] ExtractTopK(
        ReadOnlySpan<float> logSoftmax, int vocabSize, int topK, ITokenizer tokenizer)
    {
        // Track top-K token IDs and their logprobs — stackalloc is safe since topK <= 20
        Span<int> topIds = stackalloc int[topK];
        Span<float> topLogprobs = stackalloc float[topK];
        int count = 0;

        for (int i = 0; i < vocabSize; i++)
        {
            float lp = logSoftmax[i];
            if (float.IsNegativeInfinity(lp))
                continue;

            if (count < topK)
            {
                // Fill initial slots, maintaining sorted order (descending)
                int insertAt = count;
                while (insertAt > 0 && lp > topLogprobs[insertAt - 1])
                {
                    topIds[insertAt] = topIds[insertAt - 1];
                    topLogprobs[insertAt] = topLogprobs[insertAt - 1];
                    insertAt--;
                }
                topIds[insertAt] = i;
                topLogprobs[insertAt] = lp;
                count++;
            }
            else if (lp > topLogprobs[count - 1])
            {
                // Replace the smallest in our top-K and re-sort
                int insertAt = count - 1;
                while (insertAt > 0 && lp > topLogprobs[insertAt - 1])
                {
                    topIds[insertAt] = topIds[insertAt - 1];
                    topLogprobs[insertAt] = topLogprobs[insertAt - 1];
                    insertAt--;
                }
                topIds[insertAt] = i;
                topLogprobs[insertAt] = lp;
            }
        }

        var entries = new TopLogprobEntry[count];
        for (int i = 0; i < count; i++)
        {
            string token = tokenizer.Decode([topIds[i]], stripBosSpace: false);
            entries[i] = new TopLogprobEntry(
                topIds[i], token, topLogprobs[i], Encoding.UTF8.GetBytes(token));
        }

        return entries;
    }
}
