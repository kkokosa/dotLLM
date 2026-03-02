using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// RMS (Root Mean Square) normalization kernel.
/// <c>result[i] = (input[i] / rms) * weight[i]</c> where <c>rms = sqrt(mean(input²) + epsilon)</c>.
/// Used in all transformer blocks (pre-norm architecture in Llama/Mistral/Phi).
/// </summary>
public static class RmsNorm
{
    /// <summary>
    /// Computes RMS normalization with per-element weight scaling.
    /// </summary>
    /// <param name="input">Input span of length N.</param>
    /// <param name="weight">Per-element scale weights of length N.</param>
    /// <param name="epsilon">Small constant for numerical stability (typically 1e-5 or 1e-6).</param>
    /// <param name="result">Destination span. Must have length &gt;= N.
    /// May alias <paramref name="input"/> for in-place operation.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> input, ReadOnlySpan<float> weight,
                               float epsilon, Span<float> result)
    {
        // rms = sqrt(sumOfSquares / N + epsilon)
        float sumSq = TensorPrimitives.SumOfSquares(input);
        float rms = MathF.Sqrt(sumSq / input.Length + epsilon);
        float scale = 1.0f / rms;

        // result[i] = input[i] * scale * weight[i]
        TensorPrimitives.Multiply(input, scale, result);
        TensorPrimitives.Multiply(result, weight, result);
    }

    /// <summary>
    /// Scalar reference implementation for correctness verification.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> input, ReadOnlySpan<float> weight,
                                       float epsilon, Span<float> result)
    {
        float sumSq = 0;
        for (int i = 0; i < input.Length; i++)
            sumSq += input[i] * input[i];

        float rms = MathF.Sqrt(sumSq / input.Length + epsilon);
        float scale = 1.0f / rms;

        for (int i = 0; i < input.Length; i++)
            result[i] = input[i] * scale * weight[i];
    }
}
