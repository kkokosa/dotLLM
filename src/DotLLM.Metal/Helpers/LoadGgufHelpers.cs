using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Metal.Interop;
using DotLLM.Models.Gguf;

namespace DotLLM.Metal.Helpers;

/// <summary>
/// Provides helper methods for loading GGUF tensors into memory for processing.
/// Specifically, it handles conversion of 1-dimensional tensors to FP16 when required.
/// GGUF tensors may be stored in FP32 and require conversion for compatibility with certain processing kernels.
/// </summary>
public static class LoadGgufHelpers
{
    /// <summary>
    /// Loads a 1-D GGUF tensor as FP16.
    /// Norm weights and biases are commonly stored as FP32 in GGUF, while Metal
    /// kernels consume FP16 pointers.
    /// </summary>
    public static unsafe nint LoadVectorFp16(MetalContext ctx, GgufFile gguf, string tensorName, List<nint> owned)
    {
        var desc = gguf.TensorsByName[tensorName];
        int elementCount = checked((int)desc.Shape.ElementCount);
        nint src = gguf.DataBasePointer + (nint)desc.DataOffset;

        switch (desc.QuantizationType)
        {
            case QuantizationType.F16:
                return src;

            case QuantizationType.F32:
            {
                nint dst = AllocFp16Aligned(ctx, elementCount);
                var srcSpan = new ReadOnlySpan<float>((void*)src, elementCount);
                var dstSpan = new Span<Half>((void*)dst, elementCount);

                for (int i = 0; i < elementCount; i++)
                    dstSpan[i] = (Half)srcSpan[i];

                owned.Add(dst);
                return dst;
            }

            default:
                throw new NotSupportedException(
                    $"Tensor '{tensorName}' has unsupported vector format {desc.QuantizationType}. " +
                    "Expected F16 or F32.");
        }
    }

    /// <summary>
    /// Loads an optional 1-D GGUF tensor as FP16. Returns 0 when absent.
    /// </summary>
    public static nint TryLoadVectorFp16(MetalContext ctx, GgufFile gguf, string tensorName, List<nint> owned)
    {
        return gguf.TensorsByName.ContainsKey(tensorName)
            ? LoadVectorFp16(ctx, gguf, tensorName, owned)
            : 0;
    }

    /// <summary>
    /// Allocates memory aligned to a 64-byte boundary, sufficient to store the given number of FP16 elements.
    /// </summary>
    /// <param name="ctx">Metal context for memory allocation</param>
    /// <param name="elementCount">The number of FP16 elements to allocate memory for.</param>
    /// <returns>A pointer to the allocated, aligned memory block.</returns>
    public static nint AllocFp16Aligned(MetalContext ctx, int elementCount)
    {
        nuint bytes = checked((nuint)elementCount * sizeof(ushort));

        return MetalNative.AllocShared(ctx.Handle, bytes);
    }

    /// <summary>
    /// Frees an FP16 buffer previously returned by <see cref="AllocFp16Aligned"/>.
    /// The buffer was allocated via <see cref="MetalNative.AllocShared"/>, so it
    /// MUST be released via <see cref="MetalNative.FreeShared"/> — calling
    /// <c>NativeMemory.AlignedFree</c> here would leak the backing MTLBuffer
    /// (still referenced by the context's shared_buffers map) and corrupt the
    /// Metal allocator's bookkeeping.
    /// </summary>
    /// <param name="ctx">The Metal context that owned the allocation.</param>
    /// <param name="ptr">The pointer to free. No-op when zero.</param>
    public static void FreeFp16(MetalContext ctx, nint ptr)
    {
        if (ptr != 0)
            MetalNative.FreeShared(ctx.Handle, ptr);
    }
}
