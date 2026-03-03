namespace DotLLM.Core.Tensors;

/// <summary>
/// Non-owning tensor that views memory-mapped data (e.g., GGUF weight tensors).
/// <see cref="Dispose"/> is a no-op — the underlying <c>GgufFile</c> owns the mmap lifetime.
/// Callers must ensure the source memory outlives this tensor.
/// </summary>
public sealed class MappedTensor : ITensor
{
    private readonly long _elementCount;

    /// <inheritdoc/>
    public TensorShape Shape { get; }

    /// <inheritdoc/>
    public DType DType { get; }

    /// <inheritdoc/>
    public int DeviceId { get; }

    /// <inheritdoc/>
    public nint DataPointer { get; }

    /// <inheritdoc/>
    public TensorMetadata Metadata => new(Shape, DType, DeviceId, DataPointer);

    /// <inheritdoc/>
    public long ElementCount => _elementCount;

    /// <inheritdoc/>
    public long ByteCount => _elementCount * DType.SizeInBytes;

    /// <summary>
    /// Creates a non-owning tensor view over the given pointer.
    /// </summary>
    /// <param name="shape">Tensor shape.</param>
    /// <param name="dtype">Element data type.</param>
    /// <param name="dataPointer">Pointer to the tensor data. Must remain valid for this tensor's lifetime.</param>
    /// <param name="deviceId">Device ID (-1 for CPU).</param>
    public MappedTensor(TensorShape shape, DType dtype, nint dataPointer, int deviceId = -1)
    {
        Shape = shape;
        DType = dtype;
        DataPointer = dataPointer;
        DeviceId = deviceId;
        _elementCount = shape.ElementCount;
    }

    /// <summary>No-op. The underlying mmap owner is responsible for memory lifetime.</summary>
    public void Dispose()
    {
        // Intentionally empty — mmap lifetime is managed by GgufFile.
    }
}
