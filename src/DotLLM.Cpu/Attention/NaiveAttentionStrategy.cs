using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Cpu.Attention;

/// <summary>
/// O(N^2) reference attention strategy for CPU. Extracts pointers from <see cref="ITensor"/>
/// inputs and delegates to the <see cref="Kernels.Attention"/> kernel.
/// Suitable for correctness verification and small sequence lengths.
/// </summary>
public sealed class NaiveAttentionStrategy : IAttentionStrategy
{
    private readonly int _headDim;

    /// <summary>
    /// Creates a new <see cref="NaiveAttentionStrategy"/> with the specified head dimension.
    /// </summary>
    /// <param name="headDim">Dimension per attention head. Used for head count computation and stride layout.</param>
    public NaiveAttentionStrategy(int headDim)
    {
        if (headDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "headDim must be positive.");
        _headDim = headDim;
    }

    /// <inheritdoc/>
    public bool SupportsPagedKvCache => false;

    /// <inheritdoc/>
    public int? RequiredComputeCapability => null;

    /// <inheritdoc/>
    public ITensor ComputeAttention(ITensor q, ITensor k, ITensor v, ITensor? mask, float scale)
    {
        if (mask is not null)
            throw new NotSupportedException(
                "NaiveAttentionStrategy only supports causal masking. " +
                "Explicit attention masks require a different IAttentionStrategy implementation.");

        if (q.DType != DType.Float32 || k.DType != DType.Float32 || v.DType != DType.Float32)
            throw new ArgumentException("NaiveAttentionStrategy currently supports Float32 tensors only.");

        // Infer dimensions from tensor shapes.
        // Q: [seqQ, numHeads * headDim], K/V: [seqKv, numKvHeads * headDim]
        var qShape = q.Shape;
        var kShape = k.Shape;

        int seqQ = qShape[0];
        int seqKv = kShape[0];

        int headDim = _headDim;
        int numHeads = qShape[1] / headDim;
        int numKvHeads = kShape[1] / headDim;

        // Allocate output tensor with same layout as Q.
        long outputElements = q.ElementCount;
        nint outputPtr;
        unsafe
        {
            outputPtr = (nint)NativeMemory.AlignedAlloc(
                (nuint)(outputElements * sizeof(float)), 64);
        }

        try
        {
            unsafe
            {
                var qSpan = new ReadOnlySpan<float>((void*)q.DataPointer, (int)q.ElementCount);
                var kSpan = new ReadOnlySpan<float>((void*)k.DataPointer, (int)k.ElementCount);
                var vSpan = new ReadOnlySpan<float>((void*)v.DataPointer, (int)v.ElementCount);
                var outSpan = new Span<float>((void*)outputPtr, (int)outputElements);

                // positionOffset for causal mask: when seqQ < seqKv (decode), offset = seqKv - seqQ.
                int positionOffset = seqKv - seqQ;

                Kernels.Attention.Execute(qSpan, kSpan, vSpan, outSpan,
                                          seqQ, seqKv, numHeads, numKvHeads, headDim,
                                          positionOffset, scale);
            }

            // Wrap output in a tensor. Caller owns disposal.
            var outputTensor = new UnmanagedTensor(qShape, DType.Float32, q.DeviceId, outputPtr);
            outputPtr = 0; // Ownership transferred
            return outputTensor;
        }
        finally
        {
            // Free only if ownership was NOT transferred (i.e., exception path).
            unsafe
            {
                if (outputPtr != 0)
                    NativeMemory.AlignedFree((void*)outputPtr);
            }
        }
    }
}
