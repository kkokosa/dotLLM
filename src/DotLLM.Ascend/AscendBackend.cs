using DotLLM.Ascend.Interop;
using DotLLM.Core.Backends;
using DotLLM.Core.Tensors;

namespace DotLLM.Ascend;

/// <summary>Explicit single-device FP16 allocation/copy backend. Distributed operations are unsupported.</summary>
public sealed class AscendBackend : IBackend
{
    private readonly AscendSession _session;
    /// <inheritdoc/>
    public int DeviceCount { get; }
    /// <summary>The ordinal bound to this backend instance.</summary>
    public int DeviceId => _session.DeviceId;

    /// <summary>Creates a context bound to the specified visible NPU ordinal.</summary>
    public AscendBackend(int deviceId = 0)
    {
        _session = AscendSession.Create(deviceId);
        try
        {
            AscendNative.Check(AscendNative.aclrtGetDeviceCount(out uint count), "aclrtGetDeviceCount");
            DeviceCount = checked((int)count);
        }
        catch { _session.Dispose(); throw; }
    }

    /// <inheritdoc/>
    public ITensor AllocateOnDevice(int deviceId, TensorShape shape, DType dtype)
    {
        _session.ThrowIfUnavailable();
        if (dtype != DType.Float16) throw new NotSupportedException("Ascend backend storage is FP16.");
        if (deviceId == -1) return UnmanagedTensor.Allocate(shape, dtype);
        if (deviceId != DeviceId) throw new ArgumentException("Allocation must target this backend's bound NPU.");
        return _session.AllocateFloat16(shape);
    }

    /// <inheritdoc/>
    public unsafe void CopyBetweenDevices(ITensor source, ITensor destination)
    {
        ArgumentNullException.ThrowIfNull(source); ArgumentNullException.ThrowIfNull(destination);
        _session.ThrowIfUnavailable();
        if (source.DType != DType.Float16 || destination.DType != DType.Float16 ||
            source.Shape != destination.Shape || source.DataPointer == 0 || destination.DataPointer == 0)
            throw new ArgumentException("Copies require live FP16 tensors with matching shapes.");
        if (source is AscendTensor s && destination is AscendTensor d)
            _session.CopyFloat16(s, 0, d, 0, source.ElementCount);
        else if (source.DeviceId == -1 && source is UnmanagedTensor && destination is AscendTensor upload)
            _session.Upload(upload, new ReadOnlySpan<Half>((void*)source.DataPointer, checked((int)source.ElementCount)));
        else if (source is AscendTensor download && destination.DeviceId == -1 && destination is UnmanagedTensor)
            _session.Download(download, new Span<Half>((void*)destination.DataPointer, checked((int)destination.ElementCount)));
        else
            throw new ArgumentException("Copies require Ascend-owned device storage and owned unmanaged host storage.");
    }

    /// <inheritdoc/>
    public void AllReduce(ReadOnlySpan<ITensor> tensors)
        => throw new NotSupportedException("Ascend distributed collectives are not implemented.");
    /// <inheritdoc/>
    public void Send(ITensor tensor, int targetDevice)
        => throw new NotSupportedException("Ascend inter-device send is not implemented.");
    /// <inheritdoc/>
    public ITensor Receive(int sourceDevice, TensorShape shape, DType dtype)
        => throw new NotSupportedException("Ascend inter-device receive is not implemented.");
    /// <inheritdoc/>
    public void Dispose() => _session.Dispose();
}
