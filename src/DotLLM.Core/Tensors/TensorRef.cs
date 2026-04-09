using System.Runtime.CompilerServices;

namespace DotLLM.Core.Tensors;

/// <summary>
/// Zero-allocation tensor reference for hot-path use. A lightweight value type with flat
/// dimension fields — no <see cref="TensorShape"/> (which allocates an <c>int[]</c>), no
/// interface dispatch (unlike <see cref="ITensor"/>). Used in the inference loop where
/// every allocation matters.
/// </summary>
/// <remarks>
/// <see cref="TensorView"/> remains for <see cref="ITensor"/>-based APIs (diagnostics, generic consumers).
/// <see cref="TensorRef"/> is strictly for the inner decode loop where zero GC pressure is required.
/// </remarks>
public readonly record struct TensorRef
{
    /// <summary>First dimension (typically seqLen).</summary>
    public int Dim0 { get; }

    /// <summary>Second dimension (typically kvStride). 0 for 1D tensors.</summary>
    public int Dim1 { get; }

    /// <summary>Element data type.</summary>
    public DType DType { get; }

    /// <summary>Device where data resides. -1 = CPU, 0..N = GPU index.</summary>
    public int DeviceId { get; }

    /// <summary>Pointer to the start of tensor data.</summary>
    public nint DataPointer { get; }

    /// <summary>Total number of elements.</summary>
    public long ElementCount { get; }

    /// <summary>Total size of tensor data in bytes.</summary>
    public long ByteCount
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => DType.ComputeByteCount(ElementCount);
    }

    /// <summary>
    /// Creates a 2D tensor reference (e.g., [seqLen, kvStride]).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef(int dim0, int dim1, DType dtype, int deviceId, nint dataPointer)
    {
        Dim0 = dim0;
        Dim1 = dim1;
        DType = dtype;
        DeviceId = deviceId;
        DataPointer = dataPointer;
        ElementCount = (long)dim0 * dim1;
    }

    /// <summary>
    /// Creates a 1D tensor reference (e.g., [count]).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public TensorRef(int dim0, DType dtype, int deviceId, nint dataPointer)
    {
        Dim0 = dim0;
        Dim1 = 0;
        DType = dtype;
        DeviceId = deviceId;
        DataPointer = dataPointer;
        ElementCount = dim0;
    }
}
