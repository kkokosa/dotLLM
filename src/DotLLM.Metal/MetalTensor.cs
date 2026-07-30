using DotLLM.Core.Tensors;

namespace DotLLM.Metal;

/// <summary>
/// Non-owning tensor view over Metal-accessible native memory managed
/// elsewhere (for example <see cref="MetalForwardState"/> or the KV-cache).
/// Disposing only clears the pointer — it never frees the backing memory.
/// </summary>
public sealed class MetalTensor : ITensor
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

    /// <inheritdoc/>
    public MetalTensor(nint ptr, int elementCount, int deviceId = 0)
        : this(new TensorShape(elementCount), DType.Float32, deviceId, ptr)
    {
    }

    /// <summary>
    /// Create a non-owning MetalTensor view over a native pointer.
    /// </summary>
    /// <param name="shape">Tensor shape.</param>
    /// <param name="dtype">Element data type.</param>
    /// <param name="deviceId">Device placement id.</param>
    /// <param name="ptr">Pointer to native memory owned elsewhere.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="ptr"/> is null.</exception>
    public MetalTensor(TensorShape shape, DType dtype, int deviceId, nint ptr)
    {
        if (ptr == 0)
            throw new ArgumentException("Tensor pointer cannot be null.", nameof(ptr));

        Shape = shape;
        DType = dtype;
        DeviceId = deviceId;
        _ptr = ptr;
        ByteCount = dtype.ComputeByteCount(shape.ElementCount);
    }

    /// <summary>
    /// Clears the pointer. The backing memory is owned elsewhere and is not freed.
    /// </summary>
    public void Dispose() => _ptr = 0;
}
