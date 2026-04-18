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
    /// Safe to call in-place (<paramref name="input"/> and <paramref name="result"/> may alias).
    /// </summary>
    /// <param name="input">Input span.</param>
    /// <param name="result">Destination span. Must have length &gt;= <paramref name="input"/>.Length.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> input, Span<float> result)
    {
        // Non-aliased: two-pass using the destination as sigmoid scratch is optimal.
        if (!input.Overlaps(result))
        {
            TensorPrimitives.Sigmoid(input, result);
            TensorPrimitives.Multiply(input, result, result);
            return;
        }

        // Aliased (in-place): process in stack-local tiles so sigmoid doesn't stomp the input
        // before the multiply reads it. Tile = 256 floats keeps the sigmoid intermediate in L1.
        const int TileSize = 256;
        Span<float> sigBuf = stackalloc float[TileSize];
        int i = 0;
        int length = input.Length;
        for (; i + TileSize <= length; i += TileSize)
        {
            var inTile = input.Slice(i, TileSize);
            var outTile = result.Slice(i, TileSize);
            TensorPrimitives.Sigmoid(inTile, sigBuf);
            TensorPrimitives.Multiply(inTile, sigBuf, outTile);
        }
        if (i < length)
        {
            int tail = length - i;
            var inTile = input.Slice(i, tail);
            var outTile = result.Slice(i, tail);
            var sigTail = sigBuf.Slice(0, tail);
            TensorPrimitives.Sigmoid(inTile, sigTail);
            TensorPrimitives.Multiply(inTile, sigTail, outTile);
        }
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
