using DotLLM.Core.Tensors;

namespace DotLLM.Ascend;

/// <summary>
/// Contiguous floating-point device storage owned by an <see cref="AscendSession"/>.
/// The device pointer is opaque and must never be dereferenced on the CPU.
/// </summary>
public sealed class AscendTensor : ITensor
{
    private nint _devicePtr;
    internal AscendSession Session { get; }
    internal TensorShape StorageShape { get; }
    internal bool IsDisposed => _devicePtr == 0;

    internal AscendTensor(AscendSession session, TensorShape shape, nint devicePtr, long bytes, DType dtype)
    {
        Session = session;
        StorageShape = shape;
        _devicePtr = devicePtr;
        ByteCount = bytes;
        DType = dtype;
    }

    /// <inheritdoc/>
    public TensorShape Shape => new(StorageShape.Dimensions);

    /// <inheritdoc/>
    public DType DType { get; }

    /// <inheritdoc/>
    public int DeviceId => Session.DeviceId;

    /// <inheritdoc/>
    public nint DataPointer => _devicePtr;

    /// <inheritdoc/>
    public TensorMetadata Metadata => new(Shape, DType, DeviceId, DataPointer);

    /// <inheritdoc/>
    public long ElementCount => ByteCount / (DType == DType.Float16 ? 2 : 4);

    /// <inheritdoc/>
    public long ByteCount { get; }

    internal void MarkDisposed() => _devicePtr = 0;

    /// <inheritdoc/>
    public void Dispose() => Session.Release(this);
}
