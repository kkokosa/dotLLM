using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Element-wise addition kernel. Delegates to <see cref="TensorPrimitives"/>
/// which is already SIMD-accelerated across all supported platforms.
/// </summary>
public static class Add
{
    /// <summary>
    /// Computes <c>result[i] = a[i] + b[i]</c> for all elements.
    /// </summary>
    /// <param name="a">First input span.</param>
    /// <param name="b">Second input span. Must have the same length as <paramref name="a"/>.</param>
    /// <param name="result">Destination span. Must have length &gt;= <paramref name="a"/>.Length.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        TensorPrimitives.Add(a, b, result);
    }
}
