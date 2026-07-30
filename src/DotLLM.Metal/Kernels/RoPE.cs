using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Rotary Position Embedding (RoPE) accelerated via Metal GPU.
/// Direct translation of the CUDA rope_f32 kernel — computes cos/sin on the fly from theta.
/// </summary>
public static class RoPE
{
    /// <summary>
    /// Applies RoPE to query and key tensors. Convenience overload — rotates all
    /// <paramref name="headDim"/> dimensions (<c>ropeDim = headDim</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext ctx,
        Span<float> q,
        Span<float> k,
        ReadOnlySpan<int> positions,
        int numHeads,
        int numKvHeads,
        int headDim,
        float theta,
        RoPEType ropeType = RoPEType.Norm)
        => Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, headDim, theta, ropeType);

    /// <summary>
    /// Applies RoPE to query and key tensors with partial rotation support.
    /// </summary>
    /// <param name="ctx">The Metal context.</param>
    /// <param name="q">Query tensor <c>[seqLen, numHeads × headDim]</c>, modified in-place.</param>
    /// <param name="k">Key tensor <c>[seqLen, numKvHeads × headDim]</c>, modified in-place.</param>
    /// <param name="positions">Position index per token. Length determines seqLen.</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Full dimension per head.</param>
    /// <param name="ropeDim">Number of dimensions to rotate (must be even, &lt;= <paramref name="headDim"/>).</param>
    /// <param name="theta">Base frequency (e.g. 10000 for Llama 2, 500000 for Llama 3).</param>
    /// <param name="ropeType"><see cref="RoPEType.Norm"/> for Llama/Mistral, <see cref="RoPEType.NeoX"/> for Qwen/Phi.</param>
    public static void Execute(
        MetalContext ctx,
        Span<float> q,
        Span<float> k,
        ReadOnlySpan<int> positions,
        int numHeads,
        int numKvHeads,
        int headDim,
        int ropeDim,
        float theta,
        RoPEType ropeType = RoPEType.Norm)
    {
        int seqLen = positions.Length;

        if (ropeDim <= 0 || ropeDim % 2 != 0)
            throw new ArgumentException($"ropeDim must be a positive even number, got {ropeDim}.", nameof(ropeDim));
        if (ropeDim > headDim)
            throw new ArgumentException($"ropeDim ({ropeDim}) must be <= headDim ({headDim}).", nameof(ropeDim));
        if (q.Length != seqLen * numHeads * headDim)
            throw new ArgumentException($"q.Length ({q.Length}) must equal seqLen × numHeads × headDim ({seqLen * numHeads * headDim}).", nameof(q));
        if (k.Length != seqLen * numKvHeads * headDim)
            throw new ArgumentException($"k.Length ({k.Length}) must equal seqLen × numKvHeads × headDim ({seqLen * numKvHeads * headDim}).", nameof(k));

        if (seqLen == 0) return;

        int ropeTypeInt = ropeType == RoPEType.NeoX ? 1 : 0;

        unsafe
        {
            fixed (float* pQ  = q)
            fixed (float* pK  = k)
            fixed (int*   pPos = positions)
            {
                int code = MetalNative.RoPEF32(
                    ctx.Handle,
                    pQ, pK, pPos,
                    seqLen, numHeads, numKvHeads, headDim, ropeDim,
                    theta, ropeTypeInt);

                if (code != 0)
                    throw new InvalidOperationException($"Metal rope_f32 failed with code {code}.");
            }
        }
    }
}

/// <summary>
/// Rotary Position Embedding (FP16) accelerated via Metal GPU.
/// Reads/writes <c>half</c> tensors; all trigonometric computation is done in FP32.
/// Port of <c>rope_f16.cu</c>.
/// </summary>
public static class RoPEF16
{
    /// <summary>
    /// Applies RoPE to FP16 query and key tensors. Convenience overload — rotates all
    /// <paramref name="headDim"/> dimensions (<c>ropeDim = headDim</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext ctx,
        Span<Half> q,
        Span<Half> k,
        ReadOnlySpan<int> positions,
        int numHeads,
        int numKvHeads,
        int headDim,
        float theta,
        RoPEType ropeType = RoPEType.Norm)
        => Execute(ctx, q, k, positions, numHeads, numKvHeads, headDim, headDim, theta, ropeType);

    /// <summary>
    /// Applies RoPE to FP16 query and key tensors with partial rotation support.
    /// </summary>
    /// <param name="ctx">The Metal context.</param>
    /// <param name="q">Query tensor <c>[seqLen, numHeads × headDim]</c>, modified in-place.</param>
    /// <param name="k">Key tensor <c>[seqLen, numKvHeads × headDim]</c>, modified in-place.</param>
    /// <param name="positions">Position index per token. Length determines seqLen.</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Full dimension per head.</param>
    /// <param name="ropeDim">Number of dimensions to rotate (must be even, &lt;= <paramref name="headDim"/>).</param>
    /// <param name="theta">Base frequency (e.g. 10000 for Llama 2, 500000 for Llama 3).</param>
    /// <param name="ropeType"><see cref="RoPEType.Norm"/> for Llama/Mistral, <see cref="RoPEType.NeoX"/> for Qwen/Phi.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext ctx,
        Span<Half> q,
        Span<Half> k,
        ReadOnlySpan<int> positions,
        int numHeads,
        int numKvHeads,
        int headDim,
        int ropeDim,
        float theta,
        RoPEType ropeType = RoPEType.Norm)
    {
        int seqLen = positions.Length;

        if (ropeDim <= 0 || ropeDim % 2 != 0)
            throw new ArgumentException($"ropeDim must be a positive even number, got {ropeDim}.", nameof(ropeDim));
        if (ropeDim > headDim)
            throw new ArgumentException($"ropeDim ({ropeDim}) must be <= headDim ({headDim}).", nameof(ropeDim));
        if (q.Length != seqLen * numHeads * headDim)
            throw new ArgumentException($"q.Length ({q.Length}) must equal seqLen × numHeads × headDim ({seqLen * numHeads * headDim}).", nameof(q));
        if (k.Length != seqLen * numKvHeads * headDim)
            throw new ArgumentException($"k.Length ({k.Length}) must equal seqLen × numKvHeads × headDim ({seqLen * numKvHeads * headDim}).", nameof(k));

        if (seqLen == 0) return;

        int ropeTypeInt = ropeType == RoPEType.NeoX ? 1 : 0;

        var qU = MemoryMarshal.Cast<Half, ushort>(q);
        var kU = MemoryMarshal.Cast<Half, ushort>(k);

        unsafe
        {
            fixed (ushort* pQ  = qU)
            fixed (ushort* pK  = kU)
            fixed (int*    pPos = positions)
            {
                int code = MetalNative.RoPEF16(
                    ctx.Handle,
                    pQ, pK, pPos,
                    seqLen, numHeads, numKvHeads, headDim, ropeDim,
                    theta, ropeTypeInt);

                if (code != 0)
                    throw new InvalidOperationException($"Metal rope_f16 failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers and does not check buffer lengths.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Execute(
        MetalContext ctx,
        nint q,
        nint k,
        nint positions,
        int numHeads,
        int numKvHeads,
        int headDim,
        int ropeDim,
        float theta,
        int seqLen,
        RoPEType ropeType = RoPEType.Norm)
    {
        if (seqLen == 0) return;

        int ropeTypeInt = ropeType == RoPEType.NeoX ? 1 : 0;

        int code = MetalNative.RoPEF16(
            ctx.Handle,
            (ushort*)q, (ushort*)k, (int*)positions,
            seqLen, numHeads, numKvHeads, headDim, ropeDim,
            theta, ropeTypeInt);

        if (code != 0)
            throw new InvalidOperationException($"Metal rope_f16 failed with code {code}.");
    }

}
