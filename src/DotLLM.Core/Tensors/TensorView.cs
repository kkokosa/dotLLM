namespace DotLLM.Core.Tensors;

/// <summary>
/// Non-owning view over an existing tensor buffer. <see cref="Dispose"/> is a no-op —
/// the underlying memory is owned by the source (e.g., a KV-cache buffer).
/// </summary>
public sealed class TensorView : ITensor
{
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
    public long ElementCount { get; }

    /// <inheritdoc/>
    public long ByteCount => DType.ComputeByteCount(ElementCount);

    /// <summary>
    /// Creates a non-owning view over an existing data pointer.
    /// </summary>
    /// <param name="shape">Tensor shape.</param>
    /// <param name="dtype">Element data type.</param>
    /// <param name="deviceId">Device ID (-1 for CPU).</param>
    /// <param name="dataPointer">Pointer to existing data. Not freed on disposal.</param>
    public TensorView(TensorShape shape, DType dtype, int deviceId, nint dataPointer)
    {
        Shape = shape;
        DType = dtype;
        DeviceId = deviceId;
        DataPointer = dataPointer;
        ElementCount = shape.ElementCount;
    }

    /// <summary>No-op — this view does not own the underlying memory.</summary>
    public void Dispose()
    {
        // Intentionally empty: non-owning view.
    }
}
