using DotLLM.Core.Backends;
using DotLLM.Core.Tensors;

namespace DotLLM.Cpu;

/// <summary>
/// Single-device CPU backend. Allocates tensors via <see cref="UnmanagedTensor.Allocate"/>.
/// Multi-device operations are not supported and throw <see cref="NotSupportedException"/>.
/// </summary>
public sealed class CpuBackend : IBackend
{
    /// <summary>CPU backend always has exactly one device.</summary>
    public int DeviceCount => 1;

    /// <inheritdoc/>
    public ITensor AllocateOnDevice(int deviceId, TensorShape shape, DType dtype)
    {
        if (deviceId != -1)
            throw new ArgumentException($"CpuBackend only supports deviceId -1 (CPU), got {deviceId}.", nameof(deviceId));

        return UnmanagedTensor.Allocate(shape, dtype, deviceId);
    }

    /// <inheritdoc/>
    public void CopyBetweenDevices(ITensor source, ITensor destination) =>
        throw new NotSupportedException("CpuBackend does not support cross-device copies.");

    /// <inheritdoc/>
    public void AllReduce(ReadOnlySpan<ITensor> tensors) =>
        throw new NotSupportedException("CpuBackend does not support AllReduce.");

    /// <inheritdoc/>
    public void Send(ITensor tensor, int targetDevice) =>
        throw new NotSupportedException("CpuBackend does not support Send.");

    /// <inheritdoc/>
    public ITensor Receive(int sourceDevice, TensorShape shape, DType dtype) =>
        throw new NotSupportedException("CpuBackend does not support Receive.");

    /// <summary>No resources to release.</summary>
    public void Dispose()
    {
        // No owned resources.
    }
}
