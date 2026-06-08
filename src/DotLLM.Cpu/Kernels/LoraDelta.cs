using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// LoRA delta accumulation: <c>y += alpha × (x · B) · A</c>.
/// </summary>
/// <remarks>
/// <para>
/// Tensor layouts (row-major F32, matching <see cref="MatMul.GemmF32(float*, float*, float*, int, int, int)"/>):
/// </para>
/// <list type="bullet">
///   <item><c>x</c>: <c>[seqLen, inputDim]</c> — the same input that fed the base projection.</item>
///   <item><c>B</c>: <c>[rank, inputDim]</c> — LoRA down-projection weight, stored row-major
///     so <c>tmp[t, r] = sum_i x[t, i] · B[r, i]</c> matches the standard
///     "weight matrix as <c>[outputDim, inputDim]</c>" convention dotLLM uses everywhere.</item>
///   <item><c>A</c>: <c>[outputDim, rank]</c> — LoRA up-projection weight (same convention).</item>
///   <item><c>y</c>: <c>[seqLen, outputDim]</c> — base-projection output; LoRA delta added in-place.</item>
/// </list>
/// <para>
/// Implementation: two stacked GEMMs through a small <c>[seqLen, rank]</c>
/// scratch — for typical r∈[8, 64] each is a thin matmul that <see cref="MatMul.GemmF32(float*, float*, float*, int, int, int)"/>
/// already handles efficiently. We rent the scratch from <see cref="ArrayPool{T}"/>
/// so there is no per-call native allocation.
/// </para>
/// </remarks>
public static unsafe class LoraDelta
{
    /// <summary>
    /// Accumulates <c>y += scale × (x · B) · A</c> in-place. Mathematically:
    /// <c>tmp[t, r] = sum_i x[t, i] · B[r, i]</c>; then
    /// <c>y[t, o] += scale × sum_r A[o, r] · tmp[t, r]</c>.
    /// </summary>
    /// <param name="x">Input pointer, row-major <c>[seqLen, inputDim]</c>.</param>
    /// <param name="bWeight">B (down-proj) pointer, row-major <c>[rank, inputDim]</c>.</param>
    /// <param name="aWeight">A (up-proj) pointer, row-major <c>[outputDim, rank]</c>.</param>
    /// <param name="y">Output pointer (read-modify-write), row-major <c>[seqLen, outputDim]</c>.</param>
    /// <param name="seqLen">Number of input tokens in this call.</param>
    /// <param name="inputDim">Projection input dimension.</param>
    /// <param name="outputDim">Projection output dimension.</param>
    /// <param name="rank">LoRA rank (typical 8..64).</param>
    /// <param name="scale">Scaling factor — typically <c>alpha / rank</c>.</param>
    [SkipLocalsInit]
    public static void Apply(float* x, float* bWeight, float* aWeight, float* y,
                             int seqLen, int inputDim, int outputDim, int rank, float scale)
    {
        if (seqLen <= 0 || rank <= 0) return;

        // Stage 1: tmp[t, r] = sum_i x[t, i] · B[r, i].
        // GemmF32 contracts as C[N, M] = B[N, K] × A[M, K]^T, so we pass
        //   a = bWeight   (M=rank, K=inputDim)
        //   b = x         (N=seqLen, K=inputDim)
        //   c = tmp       (N=seqLen, M=rank)
        int tmpElems = seqLen * rank;
        float[] tmpBuf = ArrayPool<float>.Shared.Rent(tmpElems);
        try
        {
            fixed (float* tmp = tmpBuf)
            {
                MatMul.GemmF32(bWeight, x, tmp, rank, inputDim, seqLen);

                // Stage 2: y[t, o] += scale × sum_r A[o, r] · tmp[t, r].
                // We reuse a small per-token scratch for the A·tmp product
                // and add it into y. For typical small rank+outputDim this
                // stays in L1; a per-token GemvF32 keeps the inner loop
                // straight and reuses dotLLM's existing F32 SIMD path.
                int deltaScratchElems = outputDim;
                float[] deltaBuf = ArrayPool<float>.Shared.Rent(deltaScratchElems);
                try
                {
                    fixed (float* delta = deltaBuf)
                    {
                        for (int t = 0; t < seqLen; t++)
                        {
                            // delta[o] = sum_r A[o, r] · tmp[t, r]
                            MatMul.GemvF32(aWeight, tmp + t * rank, delta, outputDim, rank);

                            // y[t, o] += scale * delta[o] via TensorPrimitives.
                            var deltaSpan = new ReadOnlySpan<float>(delta, outputDim);
                            var ySpan = new Span<float>(y + t * outputDim, outputDim);
                            TensorPrimitives.MultiplyAdd(deltaSpan, scale, ySpan, ySpan);
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(deltaBuf);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tmpBuf);
        }
    }

    /// <summary>
    /// Convenience overload using <see cref="ReadOnlySpan{T}"/> / <see cref="Span{T}"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Apply(ReadOnlySpan<float> x, ReadOnlySpan<float> bWeight, ReadOnlySpan<float> aWeight,
                             Span<float> y, int seqLen, int inputDim, int outputDim, int rank, float scale)
    {
        if (x.Length < seqLen * inputDim)
            throw new ArgumentException($"x span too small: {x.Length} < {seqLen * inputDim}", nameof(x));
        if (bWeight.Length < rank * inputDim)
            throw new ArgumentException($"bWeight span too small: {bWeight.Length} < {rank * inputDim}", nameof(bWeight));
        if (aWeight.Length < outputDim * rank)
            throw new ArgumentException($"aWeight span too small: {aWeight.Length} < {outputDim * rank}", nameof(aWeight));
        if (y.Length < seqLen * outputDim)
            throw new ArgumentException($"y span too small: {y.Length} < {seqLen * outputDim}", nameof(y));

        fixed (float* xPtr = x)
        fixed (float* bPtr = bWeight)
        fixed (float* aPtr = aWeight)
        fixed (float* yPtr = y)
        {
            Apply(xPtr, bPtr, aPtr, yPtr, seqLen, inputDim, outputDim, rank, scale);
        }
    }
}
