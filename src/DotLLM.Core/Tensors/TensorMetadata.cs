namespace DotLLM.Core.Tensors;

/// <summary>
/// Metadata describing a tensor's shape, data type, device placement, and memory location.
/// Pure value type — no ownership semantics.
/// </summary>
/// <param name="Shape">Shape of the tensor.</param>
/// <param name="DType">Element data type.</param>
/// <param name="DeviceId">Device where data resides. -1 = CPU, 0..N = GPU index.</param>
/// <param name="DataPointer">Pointer to the start of tensor data in unmanaged or device memory.</param>
public readonly record struct TensorMetadata(
    TensorShape Shape,
    DType DType,
    int DeviceId,
    nint DataPointer);
