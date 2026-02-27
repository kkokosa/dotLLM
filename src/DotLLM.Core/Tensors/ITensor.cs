namespace DotLLM.Core.Tensors;

/// <summary>
/// A tensor: a typed, shaped block of data on a specific device.
/// Owns its underlying memory and releases it on disposal.
/// </summary>
public interface ITensor : IDisposable
{
    /// <summary>Shape of the tensor.</summary>
    TensorShape Shape { get; }

    /// <summary>Element data type.</summary>
    DType DType { get; }

    /// <summary>Device where data resides. -1 = CPU, 0..N = GPU index.</summary>
    int DeviceId { get; }

    /// <summary>Pointer to the start of tensor data in unmanaged or device memory.</summary>
    nint DataPointer { get; }

    /// <summary>Combined metadata snapshot.</summary>
    TensorMetadata Metadata { get; }

    /// <summary>Total number of elements.</summary>
    long ElementCount { get; }

    /// <summary>Total size of tensor data in bytes.</summary>
    long ByteCount { get; }
}
