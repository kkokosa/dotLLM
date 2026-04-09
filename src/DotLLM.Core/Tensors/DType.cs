using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace DotLLM.Core.Tensors;

/// <summary>
/// Data type descriptor for tensor elements. Immutable value type with pre-defined static instances.
/// </summary>
/// <param name="Name">Human-readable type name.</param>
/// <param name="SizeInBytes">Size of a single element in bytes. 0 for sub-byte quantized types.</param>
/// <param name="IsQuantized">Whether this is a quantized (non-IEEE) format.</param>
/// <param name="BlockByteSize">Size of one quantization block in bytes. 0 for non-quantized types.</param>
/// <param name="BlockElementCount">Number of elements per quantization block. 1 for non-quantized types.</param>
public readonly record struct DType(string Name, int SizeInBytes, bool IsQuantized,
    int BlockByteSize = 0, int BlockElementCount = 1)
{
    /// <summary>32-bit IEEE float.</summary>
    public static readonly DType Float32 = new("float32", 4, false);

    /// <summary>16-bit IEEE float.</summary>
    public static readonly DType Float16 = new("float16", 2, false);

    /// <summary>16-bit brain float.</summary>
    public static readonly DType BFloat16 = new("bfloat16", 2, false);

    /// <summary>Signed 8-bit integer.</summary>
    public static readonly DType Int8 = new("int8", 1, false);

    /// <summary>Unsigned 8-bit integer.</summary>
    public static readonly DType UInt8 = new("uint8", 1, false);

    /// <summary>32-bit signed integer.</summary>
    public static readonly DType Int32 = new("int32", 4, false);

    /// <summary>4-bit quantized, group size 32, no min.</summary>
    public static readonly DType Q4_0 = new("q4_0", 0, true, BlockByteSize: 18, BlockElementCount: 32);

    /// <summary>4-bit quantized, group size 32, with min.</summary>
    public static readonly DType Q4_1 = new("q4_1", 0, true, BlockByteSize: 20, BlockElementCount: 32);

    /// <summary>8-bit quantized, group size 32.</summary>
    public static readonly DType Q8_0 = new("q8_0", 0, true, BlockByteSize: 34, BlockElementCount: 32);

    /// <summary>4-bit K-quant.</summary>
    public static readonly DType Q4_K = new("q4_k", 0, true, BlockByteSize: 144, BlockElementCount: 256);

    /// <summary>5-bit K-quant.</summary>
    public static readonly DType Q5_K = new("q5_k", 0, true, BlockByteSize: 176, BlockElementCount: 256);

    /// <summary>6-bit K-quant.</summary>
    public static readonly DType Q6_K = new("q6_k", 0, true, BlockByteSize: 210, BlockElementCount: 256);

    /// <summary>
    /// Computes the total byte count for <paramref name="elementCount"/> elements of this type.
    /// Handles both quantized (block-based) and non-quantized (per-element) types correctly.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public long ComputeByteCount(long elementCount)
    {
        Debug.Assert(!IsQuantized || elementCount % BlockElementCount == 0,
            $"Quantized element count {elementCount} is not a multiple of block size {BlockElementCount}.");
        return IsQuantized ? elementCount / BlockElementCount * BlockByteSize : elementCount * SizeInBytes;
    }

    /// <inheritdoc/>
    public override string ToString() => Name;
}
