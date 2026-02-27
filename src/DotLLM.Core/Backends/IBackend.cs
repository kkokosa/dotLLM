using DotLLM.Core.Tensors;

namespace DotLLM.Core.Backends;

/// <summary>
/// Abstraction over a compute backend (CPU, CUDA, ROCm). Manages device memory and cross-device operations.
/// </summary>
public interface IBackend : IDisposable
{
    /// <summary>Number of available compute devices.</summary>
    int DeviceCount { get; }

    /// <summary>Allocates a tensor on the specified device.</summary>
    /// <param name="deviceId">Target device. -1 = CPU, 0..N = GPU.</param>
    /// <param name="shape">Shape of the tensor to allocate.</param>
    /// <param name="dtype">Element data type.</param>
    /// <returns>A newly allocated, uninitialized tensor.</returns>
    ITensor AllocateOnDevice(int deviceId, TensorShape shape, DType dtype);

    /// <summary>Copies tensor data between devices (host-to-device, device-to-host, device-to-device).</summary>
    /// <param name="source">Source tensor.</param>
    /// <param name="destination">Destination tensor. Must have compatible shape and dtype.</param>
    void CopyBetweenDevices(ITensor source, ITensor destination);

    /// <summary>All-reduce operation across multiple tensors (e.g., for tensor parallelism via NCCL).</summary>
    /// <param name="tensors">Tensors to reduce — one per participating device.</param>
    void AllReduce(ReadOnlySpan<ITensor> tensors);

    /// <summary>Sends a tensor to another device.</summary>
    /// <param name="tensor">Tensor to send.</param>
    /// <param name="targetDevice">Destination device ID.</param>
    void Send(ITensor tensor, int targetDevice);

    /// <summary>Receives a tensor from another device.</summary>
    /// <param name="sourceDevice">Source device ID.</param>
    /// <param name="shape">Expected shape.</param>
    /// <param name="dtype">Expected data type.</param>
    /// <returns>The received tensor on the current device.</returns>
    ITensor Receive(int sourceDevice, TensorShape shape, DType dtype);
}
