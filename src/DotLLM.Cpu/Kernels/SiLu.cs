using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// SiLU (Sigmoid Linear Unit) activation kernel: <c>SiLU(x) = x * sigmoid(x)</c>.
/// Used in Llama/Mistral FFN blocks as the gating activation.
/// </summary>
public static class SiLu
{
    /// <summary>Tile size. 256 floats = 1024 bytes — keeps the sigmoid intermediate in L1.</summary>
    private const int TileSize = 256;

    /// <summary>
    /// Computes <c>result[i] = input[i] * sigmoid(input[i])</c> for all elements.
    /// Uses <see cref="TensorPrimitives"/> for SIMD-accelerated sigmoid, then element-wise multiply.
    /// Any overlap between <paramref name="input"/> and <paramref name="result"/> is supported:
    /// the exact in-place case (<c>result</c> starting at the same address as <c>input</c>) and
    /// shifted overlaps, which are handled memmove-style.
    /// </summary>
    /// <param name="input">Input span.</param>
    /// <param name="result">Destination span. Must have length &gt;= <paramref name="input"/>.Length.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> input, Span<float> result)
    {
        // Non-aliased: two-pass using the destination as sigmoid scratch is optimal.
        // `elementOffset` is the start of `result` relative to `input`, in elements.
        if (!input.Overlaps(result, out int elementOffset))
        {
            TensorPrimitives.Sigmoid(input, result);
            TensorPrimitives.Multiply(input, result, result);
            return;
        }

        if (elementOffset != 0)
        {
            ExecuteShifted(input, result, elementOffset);
            return;
        }

        // Exact in-place alias: process in stack-local tiles so sigmoid doesn't stomp the input
        // before the multiply reads it. Each tile writes exactly the elements it just read.
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
    /// Handles a shifted overlap between <paramref name="input"/> and <paramref name="result"/>,
    /// where writing a tile would clobber input elements a later tile still has to read.
    /// Each tile is snapshotted into a stack buffer before it is written, and tiles are walked in
    /// the memmove-safe direction: backwards when <paramref name="elementOffset"/> is positive
    /// (destination ahead of source), forwards otherwise.
    /// </summary>
    [SkipLocalsInit]
    private static void ExecuteShifted(ReadOnlySpan<float> input, Span<float> result, int elementOffset)
    {
        Span<float> srcBuf = stackalloc float[TileSize];
        Span<float> sigBuf = stackalloc float[TileSize];
        int length = input.Length;

        if (elementOffset > 0)
        {
            for (int i = length; i > 0;)
            {
                int tile = Math.Min(TileSize, i);
                i -= tile;
                ExecuteTile(input.Slice(i, tile), result.Slice(i, tile), srcBuf, sigBuf);
            }
        }
        else
        {
            for (int i = 0; i < length; i += TileSize)
            {
                int tile = Math.Min(TileSize, length - i);
                ExecuteTile(input.Slice(i, tile), result.Slice(i, tile), srcBuf, sigBuf);
            }
        }
    }

    /// <summary>
    /// Computes one tile through disjoint stack scratch so the write to
    /// <paramref name="outTile"/> can never disturb the values being read.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ExecuteTile(ReadOnlySpan<float> inTile, Span<float> outTile,
                                    Span<float> srcBuf, Span<float> sigBuf)
    {
        Span<float> src = srcBuf.Slice(0, inTile.Length);
        Span<float> sig = sigBuf.Slice(0, inTile.Length);
        inTile.CopyTo(src);
        TensorPrimitives.Sigmoid(src, sig);
        TensorPrimitives.Multiply((ReadOnlySpan<float>)src, sig, outTile);
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
