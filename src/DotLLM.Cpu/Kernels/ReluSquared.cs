using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Squared ReLU activation: <c>y = max(0, x)^2</c>.
/// Used by NVIDIA Nemotron-H FFN layers as a non-gated replacement for SwiGLU.
/// </summary>
public static class ReluSquared
{
    /// <summary>
    /// Computes <c>result[i] = max(0, input[i])^2</c> for all elements.
    /// Safe to alias <paramref name="input"/> and <paramref name="result"/>.
    /// </summary>
    /// <param name="input">Input span.</param>
    /// <param name="result">Destination span; must have length &gt;= <paramref name="input"/>.Length.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> input, Span<float> result)
    {
        // relu(x) = max(x, 0). Both TensorPrimitives ops are element-wise so aliased
        // (input == result) is safe — each lane reads then writes the same position.
        // A previous version did Multiply(input,input,result) first and then a scalar
        // zero-out step that read input[i], but in-place that read saw the already-
        // squared (non-negative) value and the zero-out became a no-op; fixed here.
        TensorPrimitives.Max(input, 0.0f, result);
        TensorPrimitives.Multiply(result, result, result);
    }

    /// <summary>Scalar reference for correctness verification.</summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> input, Span<float> result)
    {
        for (int i = 0; i < input.Length; i++)
        {
            float v = input[i];
            if (v < 0.0f) v = 0.0f;
            result[i] = v * v;
        }
    }
}
