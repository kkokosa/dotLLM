using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Fast approximate math functions using IEEE-754 bit-manipulation tricks.
/// Intended for attention softmax where full precision is unnecessary — errors in exp
/// get normalized away when dividing by the sum.
/// <para>
/// The core trick (Schraudolph 1999): <c>exp(x) ≈ reinterpret_as_float((int)(x * C0 + C1))</c> where
/// <c>C0 = 2^23 / ln(2)</c> maps x into the IEEE-754 exponent field and <c>C1</c> is a bias constant.
/// This yields ~1-2% max relative error in ~3 SIMD ops vs ~12 for a polynomial approximation.
/// </para>
/// </summary>
public static class FastMath
{
    /// <summary>Scale factor: 2^23 / ln(2). Maps x/ln(2) into the IEEE-754 exponent bit field.</summary>
    private const float C0 = 12102203.0f;

    /// <summary>
    /// Bias constant: (127 - 0.0579) * 2^23. Minimax-optimized for softmax use (minimizes
    /// relative error over [-88, 0] range typical of attention scores after max-subtraction).
    /// </summary>
    private const float C1 = 1064866805.0f;

    /// <summary>Lower clamp: exp(-87.3) ≈ 1.3e-38 (near float min normal). Prevents negative int bits.</summary>
    private const float MinClamp = -87.3f;

    /// <summary>Upper clamp: exp(88.7) ≈ 3.0e+38 (near float max). Prevents overflow.</summary>
    private const float MaxClamp = 88.7f;

    /// <summary>
    /// Scalar fast approximate exp. ~1-2% max relative error.
    /// Clamped to [-87.3, 88.7] for general use.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float FastExp(float x)
    {
        x = Math.Clamp(x, MinClamp, MaxClamp);
        int bits = (int)(x * C0 + C1);
        return Unsafe.BitCast<int, float>(bits);
    }

    /// <summary>
    /// Fused shift + fast exp + store + sum in a single pass.
    /// Computes: <c>output[i] = fast_exp(input[i] + offset)</c>, returns <c>sum(output)</c>.
    /// Replaces separate <c>TensorPrimitives.Add + Exp + Sum</c> with one pass over the data.
    /// </summary>
    /// <param name="input">Input span (e.g., attention scores for one tile).</param>
    /// <param name="output">Output span. May alias <paramref name="input"/> for in-place operation.</param>
    /// <param name="offset">Additive offset (typically <c>-max</c> for numerical stability).</param>
    /// <returns>Sum of all exponentiated values.</returns>
    /// <remarks>
    /// The vectorized paths only clamp the lower bound (<c>-87.3f</c>). This is safe because the intended
    /// use is attention softmax where <paramref name="offset"/> is <c>-max(input)</c>, guaranteeing
    /// <c>input[i] + offset ≤ 0</c>. Callers passing a positive offset must ensure
    /// <c>input[i] + offset ≤ 88.7f</c> to avoid integer overflow in the bit trick.
    /// </remarks>
    [SkipLocalsInit]
    public static float ExpSumAndStore(ReadOnlySpan<float> input, Span<float> output, float offset)
    {
        int length = input.Length;
        ref float src = ref MemoryMarshal.GetReference(input);
        ref float dst = ref MemoryMarshal.GetReference(output);

        if (Avx512F.IsSupported)
            return ExpSumAndStoreAvx512(ref src, ref dst, length, offset);

        if (Avx2.IsSupported)
            return ExpSumAndStoreAvx2(ref src, ref dst, length, offset);

        return ExpSumAndStoreScalar(ref src, ref dst, length, offset);
    }

    /// <summary>
    /// Fast approximate softmax using IEEE-754 bit-manipulation exp.
    /// For attention scores where full precision is unnecessary.
    /// Standard <see cref="Softmax.Execute"/> should be used for sampling softmax.
    /// </summary>
    /// <param name="input">Input span (logits/scores).</param>
    /// <param name="result">Destination span. May alias <paramref name="input"/> for in-place operation.</param>
    [SkipLocalsInit]
    public static void Softmax(ReadOnlySpan<float> input, Span<float> result)
    {
        float max = TensorPrimitives.Max(input);
        float sum = ExpSumAndStore(input, result, -max);
        TensorPrimitives.Multiply(result, 1.0f / sum, result);
    }

    // ──────────────────── AVX-512 path ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ExpSumAndStoreAvx512(ref float src, ref float dst, int length, float offset)
    {
        var offsetVec = Vector512.Create(offset);
        var c0Vec = Vector512.Create(C0);
        var c1Vec = Vector512.Create(C1);
        var minVec = Vector512.Create(MinClamp);
        var sumVec = Vector512<float>.Zero;

        nuint i = 0;
        nuint vecLen = (nuint)(length & ~15); // 16 floats per iteration

        for (; i < vecLen; i += 16)
        {
            var x = Vector512.LoadUnsafe(ref src, i);
            var shifted = Avx512F.Add(x, offsetVec);
            shifted = Avx512F.Max(shifted, minVec);
            var y = Avx512F.FusedMultiplyAdd(shifted, c0Vec, c1Vec);
            var bits = Avx512F.ConvertToVector512Int32WithTruncation(y);
            var exp = bits.AsSingle();
            Vector512.StoreUnsafe(exp, ref dst, i);
            sumVec = Avx512F.Add(sumVec, exp);
        }

        float sum = Vector512.Sum(sumVec);

        // Scalar tail
        for (; i < (nuint)length; i++)
        {
            float val = Unsafe.Add(ref src, i) + offset;
            val = MathF.Max(val, MinClamp);
            int bits = (int)(val * C0 + C1);
            float exp = Unsafe.BitCast<int, float>(bits);
            Unsafe.Add(ref dst, i) = exp;
            sum += exp;
        }

        return sum;
    }

    // ──────────────────── AVX2 path ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ExpSumAndStoreAvx2(ref float src, ref float dst, int length, float offset)
    {
        var offsetVec = Vector256.Create(offset);
        var c0Vec = Vector256.Create(C0);
        var c1Vec = Vector256.Create(C1);
        var minVec = Vector256.Create(MinClamp);
        var sumVec = Vector256<float>.Zero;

        nuint i = 0;
        nuint vecLen = (nuint)(length & ~7); // 8 floats per iteration

        for (; i < vecLen; i += 8)
        {
            var x = Vector256.LoadUnsafe(ref src, i);
            var shifted = Avx.Add(x, offsetVec);
            shifted = Avx.Max(shifted, minVec);
            Vector256<float> y;
            if (Fma.IsSupported)
                y = Fma.MultiplyAdd(shifted, c0Vec, c1Vec);
            else
                y = Avx.Add(Avx.Multiply(shifted, c0Vec), c1Vec);
            var bits = Avx.ConvertToVector256Int32WithTruncation(y);
            var exp = bits.AsSingle();
            Vector256.StoreUnsafe(exp, ref dst, i);
            sumVec = Avx.Add(sumVec, exp);
        }

        float sum = Vector256.Sum(sumVec);

        // Scalar tail
        for (; i < (nuint)length; i++)
        {
            float val = Unsafe.Add(ref src, i) + offset;
            val = MathF.Max(val, MinClamp);
            int bits = (int)(val * C0 + C1);
            float exp = Unsafe.BitCast<int, float>(bits);
            Unsafe.Add(ref dst, i) = exp;
            sum += exp;
        }

        return sum;
    }

    // ──────────────────── Scalar fallback ────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ExpSumAndStoreScalar(ref float src, ref float dst, int length, float offset)
    {
        float sum = 0f;

        for (int i = 0; i < length; i++)
        {
            float val = Unsafe.Add(ref src, (nuint)i) + offset;
            float exp = FastExp(val);
            Unsafe.Add(ref dst, (nuint)i) = exp;
            sum += exp;
        }

        return sum;
    }
}
