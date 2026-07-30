using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Tiled scaled dot-product attention with FP16 Q/K/V/output, FP32 accumulation,
/// online softmax, causal masking, GQA head broadcast, and optional sliding-window attention.
/// Accelerated via Metal GPU. Direct translation of attention.cu (attention_f16 kernel).
/// </summary>
public static class AttentionF16
{
    /// <summary>
    /// Computes <c>output = softmax((Q @ K^T) / sqrt(headDim) + causalMask) @ V</c>
    /// with FP16 inputs/output, FP32 accumulation, and GQA head broadcast.
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="q">Query tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">Key tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">Value tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="output">Output tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Number of query positions.</param>
    /// <param name="seqKv">Number of key/value positions.</param>
    /// <param name="numHeads">Number of query heads.</param>
    /// <param name="numKvHeads">Number of key/value heads. Must divide <paramref name="numHeads"/>.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="positionOffset">Position of the first query token. 0 for prefill.</param>
    /// <param name="slidingWindow">Sliding window size. 0 = full causal attention.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext       ctx,
        ReadOnlySpan<Half> q,
        ReadOnlySpan<Half> k,
        ReadOnlySpan<Half> v,
        Span<Half>         output,
        int seqQ,
        int seqKv,
        int numHeads,
        int numKvHeads,
        int headDim,
        int positionOffset,
        int slidingWindow = 0)
    {
        ValidateArgs(seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset,
                     q.Length, k.Length, v.Length, output.Length);

        ReadOnlySpan<ushort> qRaw  = MemoryMarshal.Cast<Half, ushort>(q);
        ReadOnlySpan<ushort> kRaw  = MemoryMarshal.Cast<Half, ushort>(k);
        ReadOnlySpan<ushort> vRaw  = MemoryMarshal.Cast<Half, ushort>(v);
        Span<ushort>         outRaw = MemoryMarshal.Cast<Half, ushort>(output);

        unsafe
        {
            fixed (ushort* pQ   = qRaw)
            fixed (ushort* pK   = kRaw)
            fixed (ushort* pV   = vRaw)
            fixed (ushort* pOut = outRaw)
            {
                int code = MetalNative.AttentionF16(
                    ctx.Handle,
                    pQ, pK, pV, pOut,
                    seqQ, seqKv, numHeads, numKvHeads, headDim,
                    positionOffset, slidingWindow);

                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal attention_f16 failed with code {code}.");
            }
        }
    }

    private static void ValidateArgs(
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset,
        int qLen, int kLen, int vLen, int outLen)
    {
        if (seqQ <= 0)       throw new ArgumentOutOfRangeException(nameof(seqQ));
        if (seqKv <= 0)      throw new ArgumentOutOfRangeException(nameof(seqKv));
        if (numHeads <= 0)   throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numKvHeads <= 0 || numHeads % numKvHeads != 0)
            throw new ArgumentException($"numKvHeads ({numKvHeads}) must divide numHeads ({numHeads}).");
        if (headDim <= 0)    throw new ArgumentOutOfRangeException(nameof(headDim));
        if (positionOffset < 0) throw new ArgumentOutOfRangeException(nameof(positionOffset));

        int expQ  = seqQ  * numHeads    * headDim;
        int expKv = seqKv * numKvHeads  * headDim;
        if (qLen != expQ)   throw new ArgumentException($"q.Length ({qLen}) must equal {expQ}.");
        if (kLen != expKv)  throw new ArgumentException($"k.Length ({kLen}) must equal {expKv}.");
        if (vLen != expKv)  throw new ArgumentException($"v.Length ({vLen}) must equal {expKv}.");
        if (outLen != expQ) throw new ArgumentException($"output.Length ({outLen}) must equal {expQ}.");
    }

    /// <summary>
    /// Forward-pass overload: takes raw <see cref="nint"/> pointers and does not check buffer lengths.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void Execute(
        MetalContext ctx,
        nint q,
        nint k,
        nint v,
        nint output,
        int seqQ,
        int seqKv,
        int numHeads,
        int numKvHeads,
        int headDim,
        int positionOffset,
        int slidingWindow = 0)
    {
        int code = MetalNative.AttentionF16(
            ctx.Handle,
            (ushort*)q, (ushort*)k, (ushort*)v, (ushort*)output,
            seqQ, seqKv, numHeads, numKvHeads, headDim,
            positionOffset, slidingWindow);

        if (code != 0)
        {
            throw new InvalidOperationException($"Metal attention_f16 failed with code {code}.");
        }
    }

    /// <summary>
    /// Attention using persistent K/V MTLBuffers from the given cache layer.
    /// K/V must already have been written to the cache
    /// before calling this. seqKv is taken from <paramref name="cache"/>.CurrentLength.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void ExecuteWithKvCache(
        MetalContext ctx,
        MetalKvCache cache,
        int          layer,
        nint         q,
        nint         output,
        int          seqQ,
        int          numHeads,
        int          numKvHeads,
        int          headDim,
        int          positionOffset,
        int          slidingWindow = 0)
    {
        int code = MetalNative.AttentionF16WithKvCache(
            ctx.Handle, cache.Handle,
            (ushort*)q, (ushort*)output,
            layer, seqQ, numHeads, numKvHeads, headDim,
            positionOffset, slidingWindow);

        if (code != 0)
            throw new InvalidOperationException(
                $"Metal attention_f16_kvcache failed with code {code}.");
    }
}

/// <summary>
/// Tiled scaled dot-product attention with FP32 Q/K/V/output, online softmax,
/// causal masking, GQA head broadcast, and optional sliding-window attention.
/// Accelerated via Metal GPU. Direct translation of attention_f32.cu.
/// </summary>
public static class AttentionF32
{
    /// <summary>
    /// Computes <c>output = softmax((Q @ K^T) / sqrt(headDim) + causalMask) @ V</c>
    /// with online (numerically stable) softmax and GQA head broadcast.
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="q">Query tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">Key tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">Value tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="output">Output tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Number of query positions.</param>
    /// <param name="seqKv">Number of key/value positions (total context length).</param>
    /// <param name="numHeads">Number of query heads.</param>
    /// <param name="numKvHeads">Number of key/value heads. Must divide <paramref name="numHeads"/>.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="positionOffset">
    /// Position of the first query token in the sequence.
    /// 0 for prefill; number of cached tokens for decode.
    /// </param>
    /// <param name="slidingWindow">
    /// Sliding window size. 0 (default) means full causal attention.
    /// </param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext        ctx,
        ReadOnlySpan<float> q,
        ReadOnlySpan<float> k,
        ReadOnlySpan<float> v,
        Span<float>         output,
        int seqQ,
        int seqKv,
        int numHeads,
        int numKvHeads,
        int headDim,
        int positionOffset,
        int slidingWindow = 0)
    {
        if (seqQ <= 0)
            throw new ArgumentOutOfRangeException(nameof(seqQ), "seqQ must be > 0.");
        if (seqKv <= 0)
            throw new ArgumentOutOfRangeException(nameof(seqKv), "seqKv must be > 0.");
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be > 0.");
        if (numKvHeads <= 0 || numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numKvHeads ({numKvHeads}) must be > 0 and divide numHeads ({numHeads}).",
                nameof(numKvHeads));
        if (headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), "headDim must be > 0.");
        if (positionOffset < 0)
            throw new ArgumentOutOfRangeException(nameof(positionOffset), "positionOffset must be >= 0.");

        int expectedQ      = seqQ  * numHeads    * headDim;
        int expectedKv     = seqKv * numKvHeads  * headDim;
        int expectedOutput = seqQ  * numHeads    * headDim;

        if (q.Length != expectedQ)
            throw new ArgumentException(
                $"q.Length ({q.Length}) must equal seqQ × numHeads × headDim ({expectedQ}).", nameof(q));
        if (k.Length != expectedKv)
            throw new ArgumentException(
                $"k.Length ({k.Length}) must equal seqKv × numKvHeads × headDim ({expectedKv}).", nameof(k));
        if (v.Length != expectedKv)
            throw new ArgumentException(
                $"v.Length ({v.Length}) must equal seqKv × numKvHeads × headDim ({expectedKv}).", nameof(v));
        if (output.Length != expectedOutput)
            throw new ArgumentException(
                $"output.Length ({output.Length}) must equal seqQ × numHeads × headDim ({expectedOutput}).",
                nameof(output));

        unsafe
        {
            fixed (float* pQ = q)
            fixed (float* pK = k)
            fixed (float* pV = v)
            fixed (float* pOut = output)
            {
                int code = MetalNative.AttentionF32(
                    ctx.Handle,
                    pQ, pK, pV, pOut,
                    seqQ, seqKv, numHeads, numKvHeads, headDim,
                    positionOffset, slidingWindow);

                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal attention_f32 failed with code {code}.");
            }
        }
    }
}
