using DotLLM.Core.Backends;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// GPU backend using NVIDIA CUDA via Driver API P/Invoke.
/// Supports single-device operations; multi-device (AllReduce/Send/Receive)
/// throws <see cref="NotSupportedException"/> until Step 51 (NCCL).
/// </summary>
public sealed class CudaBackend : IBackend
{
    /// <inheritdoc/>
    public int DeviceCount { get; }

    /// <summary>Creates a CUDA backend, initializing the driver and enumerating devices.</summary>
    public CudaBackend()
    {
        CudaLibraryResolver.Register();
        CudaDriverApi.cuInit(0).ThrowOnError();
        CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
        DeviceCount = count;
    }

    /// <inheritdoc/>
    public ITensor AllocateOnDevice(int deviceId, TensorShape shape, DType dtype)
    {
        if (deviceId < 0)
            throw new ArgumentException($"CudaBackend requires deviceId >= 0, got {deviceId}.", nameof(deviceId));
        if (deviceId >= DeviceCount)
            throw new ArgumentException($"Device {deviceId} not available (have {DeviceCount} GPU(s)).", nameof(deviceId));

        return CudaTensor.Allocate(shape, dtype, deviceId);
    }

    /// <inheritdoc/>
    public void CopyBetweenDevices(ITensor source, ITensor destination)
    {
        if (source.ByteCount != destination.ByteCount)
            throw new ArgumentException(
                $"Source ({source.ByteCount} bytes) and destination ({destination.ByteCount} bytes) sizes differ.");

        nuint bytes = (nuint)source.ByteCount;

        if (source.DeviceId == -1 && destination.DeviceId >= 0)
        {
            // Host → Device
            CudaDriverApi.cuMemcpyHtoD_v2(destination.DataPointer, source.DataPointer, bytes).ThrowOnError();
        }
        else if (source.DeviceId >= 0 && destination.DeviceId == -1)
        {
            // Device → Host
            CudaDriverApi.cuMemcpyDtoH_v2(destination.DataPointer, source.DataPointer, bytes).ThrowOnError();
        }
        else if (source.DeviceId >= 0 && destination.DeviceId >= 0)
        {
            // Device → Device
            CudaDriverApi.cuMemcpyDtoD_v2(destination.DataPointer, source.DataPointer, bytes).ThrowOnError();
        }
        else
        {
            throw new ArgumentException("CPU-to-CPU copy is not a CudaBackend operation.");
        }
    }

    /// <inheritdoc/>
    public void AllReduce(ReadOnlySpan<ITensor> tensors) =>
        throw new NotSupportedException("CudaBackend does not support AllReduce (requires NCCL, Step 51).");

    /// <inheritdoc/>
    public void Send(ITensor tensor, int targetDevice) =>
        throw new NotSupportedException("CudaBackend does not support Send (requires NCCL, Step 51).");

    /// <inheritdoc/>
    public ITensor Receive(int sourceDevice, TensorShape shape, DType dtype) =>
        throw new NotSupportedException("CudaBackend does not support Receive (requires NCCL, Step 51).");

    /// <inheritdoc/>
    public void Dispose()
    {
        // No owned resources — contexts and streams are managed by CudaTransformerModel.
    }
}
