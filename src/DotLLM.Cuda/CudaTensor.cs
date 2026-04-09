using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU-resident tensor backed by <c>cuMemAlloc_v2</c>. Owns the device memory
/// and frees it on disposal. <see cref="DataPointer"/> is an opaque device pointer
/// — it must not be dereferenced from C#.
/// </summary>
public sealed class CudaTensor : ITensor
{
    private nint _ptr;

    /// <inheritdoc/>
    public TensorShape Shape { get; }

    /// <inheritdoc/>
    public DType DType { get; }

    /// <inheritdoc/>
    public int DeviceId { get; }

    /// <inheritdoc/>
    public nint DataPointer => _ptr;

    /// <inheritdoc/>
    public TensorMetadata Metadata => new(Shape, DType, DeviceId, _ptr);

    /// <inheritdoc/>
    public long ElementCount => Shape.ElementCount;

    /// <inheritdoc/>
    public long ByteCount { get; }

    private CudaTensor(TensorShape shape, DType dtype, int deviceId, nint ptr, long byteCount)
    {
        Shape = shape;
        DType = dtype;
        DeviceId = deviceId;
        _ptr = ptr;
        ByteCount = byteCount;
    }

    /// <summary>
    /// Allocates a GPU tensor of the given shape and data type on the current context's device.
    /// </summary>
    /// <param name="shape">Tensor shape.</param>
    /// <param name="dtype">Element data type.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    public static CudaTensor Allocate(TensorShape shape, DType dtype, int deviceId)
    {
        long bytes = dtype.ComputeByteCount(shape.ElementCount);
        if (bytes <= 0)
            throw new ArgumentException($"Cannot allocate tensor with {shape.ElementCount} elements of {dtype} (computed byte count={bytes}).");

        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        return new CudaTensor(shape, dtype, deviceId, ptr, bytes);
    }

    /// <summary>
    /// Allocates a GPU tensor with an explicit byte count (for quantized types where
    /// <see cref="DType.SizeInBytes"/> is 0).
    /// </summary>
    public static CudaTensor AllocateBytes(TensorShape shape, DType dtype, int deviceId, long byteCount)
    {
        if (byteCount <= 0)
            throw new ArgumentException($"Byte count must be positive, got {byteCount}.");

        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)byteCount).ThrowOnError();
        return new CudaTensor(shape, dtype, deviceId, ptr, byteCount);
    }

    /// <summary>
    /// Wraps an existing device pointer as a non-owning tensor view.
    /// The caller is responsible for the lifetime of the pointer.
    /// </summary>
    internal static CudaTensor WrapExisting(TensorShape shape, DType dtype, int deviceId, nint devicePtr, long byteCount)
    {
        return new CudaTensor(shape, dtype, deviceId, devicePtr, byteCount) { _ownsMemory = false };
    }

    private bool _ownsMemory = true;

    /// <inheritdoc/>
    public void Dispose()
    {
        nint ptr = Interlocked.Exchange(ref _ptr, 0);
        if (ptr != 0 && _ownsMemory)
            CudaDriverApi.cuMemFree_v2(ptr);
    }
}
