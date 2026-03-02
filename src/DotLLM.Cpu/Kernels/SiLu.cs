using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// SiLU (Sigmoid Linear Unit) activation kernel: <c>SiLU(x) = x * sigmoid(x)</c>.
/// Used in Llama/Mistral FFN blocks as the gating activation.
/// </summary>
public static class SiLu
{
    /// <summary>
    /// Computes <c>result[i] = input[i] * sigmoid(input[i])</c> for all elements.
    /// Uses <see cref="TensorPrimitives"/> for SIMD-accelerated sigmoid, then element-wise multiply.
    /// </summary>
    /// <param name="input">Input span.</param>
    /// <param name="result">Destination span. Must have length &gt;= <paramref name="input"/>.Length.
    /// Must not alias <paramref name="input"/>.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> input, Span<float> result)
    {
        // sigmoid(input) → result, then result *= input
        TensorPrimitives.Sigmoid(input, result);
        TensorPrimitives.Multiply(input, result, result);
    }

    /// <summary>
    /// Scalar reference implementation for correctness verification.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> input, Span<float> result)
    {
        for (int i = 0; i < input.Length; i++)
        {
            float x = input[i];
            float sigmoid = 1.0f / (1.0f + MathF.Exp(-x));
            result[i] = x * sigmoid;
        }
    }
}
