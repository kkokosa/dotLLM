using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DotLLM.Core.Tensors;

/// <summary>
/// Tensor implementation backed by 64-byte-aligned unmanaged memory.
/// Owns the allocation and frees it on disposal. Thread-safe disposal via interlocked exchange.
/// </summary>
public sealed class UnmanagedTensor : ITensor
{
    private nint _ptr;
    private readonly long _elementCount;

    /// <inheritdoc/>
    public TensorShape Shape { get; }

    /// <inheritdoc/>
    public DType DType { get; }

    /// <inheritdoc/>
    public int DeviceId { get; }

    /// <inheritdoc/>
    public nint DataPointer => _ptr;

    /// <inheritdoc/>
    public TensorMetadata Metadata => new(Shape, DType, DeviceId, DataPointer);

    /// <inheritdoc/>
    public long ElementCount => _elementCount;

    /// <inheritdoc/>
    public long ByteCount => _elementCount * DType.SizeInBytes;

    /// <summary>
    /// Wraps an existing aligned pointer in a tensor. The tensor takes ownership of the pointer.
    /// </summary>
    /// <param name="shape">Tensor shape.</param>
    /// <param name="dtype">Element data type.</param>
    /// <param name="deviceId">Device ID (-1 for CPU).</param>
    /// <param name="ptr">Pointer to 64-byte-aligned memory. Ownership transfers to this tensor.</param>
    public UnmanagedTensor(TensorShape shape, DType dtype, int deviceId, nint ptr)
    {
        Shape = shape;
        DType = dtype;
        DeviceId = deviceId;
        _ptr = ptr;
        _elementCount = shape.ElementCount;
    }

    /// <summary>
    /// Allocates a new tensor with 64-byte-aligned unmanaged memory.
    /// </summary>
    /// <param name="shape">Tensor shape.</param>
    /// <param name="dtype">Element data type. Must have non-zero <see cref="DType.SizeInBytes"/>.</param>
    /// <param name="deviceId">Device ID (-1 for CPU).</param>
    /// <returns>A newly allocated tensor. Caller owns disposal.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe UnmanagedTensor Allocate(TensorShape shape, DType dtype, int deviceId = -1)
    {
        long byteCount = shape.ElementCount * dtype.SizeInBytes;
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)byteCount, 64);
        return new UnmanagedTensor(shape, dtype, deviceId, ptr);
    }

    /// <summary>Releases the unmanaged memory via the finalizer if <see cref="Dispose"/> was not called.</summary>
    ~UnmanagedTensor()
    {
        FreeMemory();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        FreeMemory();
        GC.SuppressFinalize(this);
    }

    private unsafe void FreeMemory()
    {
        nint ptr = Interlocked.Exchange(ref _ptr, 0);
        if (ptr != 0)
            NativeMemory.AlignedFree((void*)ptr);
    }
}
