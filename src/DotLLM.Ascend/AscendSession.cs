using DotLLM.Ascend.Interop;
using DotLLM.Core.Tensors;

namespace DotLLM.Ascend;

/// <summary>
/// Owns an explicit NPU context, stream, and its tensors. Operations are serialized
/// and synchronous during bring-up. Dispose releases all remaining tensors.
/// DotLLM must exclusively own AscendCL initialization in this process.
/// </summary>
public sealed partial class AscendSession : IDisposable
{
    private static readonly object RuntimeGate = new();
    private static int _runtimeOwners;
    private readonly object _gate = new();
    private readonly HashSet<AscendTensor> _tensors = [];
    private nint _context;
    private nint _stream;
    private bool _disposed;
    private bool _faulted;

    /// <summary>The logical NPU ordinal bound at creation.</summary>
    public int DeviceId { get; }

    private AscendSession(int deviceId) => DeviceId = deviceId;

    /// <summary>Initializes AscendCL and creates a context on a logical device.</summary>
    public static AscendSession Create(int deviceId = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
        if (!OperatingSystem.IsLinux() || IntPtr.Size != 8)
            throw new PlatformNotSupportedException("Ascend bring-up requires 64-bit Linux.");
        AscendNative.Register();
        uint version = AscendNative.AbiVersion();
        if (version != 3)
            throw new NotSupportedException($"Ascend native ABI {version}; expected 3. Rebuild the native bridge.");
        var session = new AscendSession(deviceId);
        lock (RuntimeGate)
        {
            if (_runtimeOwners == 0)
                AscendNative.Check(AscendNative.aclInit(0), "aclInit");
            _runtimeOwners++;
        }
        try
        {
            AscendNative.Check(AscendNative.aclrtGetDeviceCount(out uint count), "aclrtGetDeviceCount");
            if ((uint)deviceId >= count)
                throw new ArgumentOutOfRangeException(nameof(deviceId), $"Only {count} logical NPUs are visible.");
            AscendNative.Check(AscendNative.aclrtCreateContext(out session._context, deviceId), "aclrtCreateContext");
            session.MakeCurrent();
            AscendNative.Check(AscendNative.aclrtCreateStream(out session._stream), "aclrtCreateStream");
            return session;
        }
        catch
        {
            // Preserve the initialization failure even if cleanup also fails.
            try { session.Dispose(); } catch (AscendException) { }
            throw;
        }
    }

    private void MakeCurrent()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_faulted)
            throw new InvalidOperationException("This Ascend session failed a device operation; dispose and recreate it.");
        AscendNative.Check(AscendNative.aclrtSetCurrentContext(_context), "aclrtSetCurrentContext");
    }

    /// <summary>Allocates a contiguous FP16 tensor on this session's device.</summary>
    public AscendTensor AllocateFloat16(TensorShape shape) => Allocate(shape, DType.Float16);

    /// <summary>Allocates FP32 storage for numerically stable intermediate computations.</summary>
    public AscendTensor AllocateFloat32(TensorShape shape) => Allocate(shape, DType.Float32);

    private AscendTensor Allocate(TensorShape shape, DType dtype)
    {
        // TensorShape exposes mutable dimension arrays; take an independent metadata snapshot.
        var ownedShape = new TensorShape(shape.Dimensions);
        long count = 1;
        foreach (int dimension in ownedShape.Dimensions)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimension);
            count = checked(count * dimension);
        }
        long bytes = checked(count * (dtype == DType.Float16 ? 2 : 4));
        lock (_gate)
        {
            MakeCurrent();
            AscendNative.Check(AscendNative.aclrtMalloc(out nint devicePtr, checked((nuint)bytes), 0), "aclrtMalloc");
            try
            {
                var tensor = new AscendTensor(this, ownedShape, devicePtr, bytes, dtype);
                _tensors.Add(tensor);
                return tensor;
            }
            catch
            {
                AscendNative.aclrtFree(devicePtr);
                throw;
            }
        }
    }

    private void Validate(AscendTensor tensor)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!ReferenceEquals(tensor.Session, this))
            throw new ArgumentException("Tensor belongs to a different Ascend session.", nameof(tensor));
        ObjectDisposedException.ThrowIf(tensor.IsDisposed, tensor);
    }

    /// <summary>Returns free and total device memory in bytes in this context.</summary>
    public (long FreeBytes, long TotalBytes) GetMemoryInfo()
    {
        lock (_gate)
        {
            MakeCurrent();
            AscendNative.Check(AscendNative.aclrtGetMemInfo(0, out nuint free, out nuint total), "aclrtGetMemInfo");
            return (checked((long)free), checked((long)total));
        }
    }

    internal void ThrowIfUnavailable()
    {
        lock (_gate) MakeCurrent();
    }

    internal void Release(AscendTensor tensor)
    {
        lock (_gate)
        {
            if (tensor.IsDisposed) return;
            ObjectDisposedException.ThrowIf(_disposed, this);
            AscendNative.Check(AscendNative.aclrtSetCurrentContext(_context), "aclrtSetCurrentContext");
            AscendNative.Check(AscendNative.aclrtFree(tensor.DataPointer), "aclrtFree");
            tensor.MarkDisposed();
            _tensors.Remove(tensor);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            int firstError = 0;
            string failedOperation = "";
            void Record(int status, string operation)
            {
                if (status == 0 || firstError != 0) return;
                firstError = status;
                failedOperation = operation;
            }
            // Attempt every cleanup, but report failures rather than silently claiming success.
            if (_context != 0) Record(AscendNative.aclrtSetCurrentContext(_context), "aclrtSetCurrentContext");
            if (_stream != 0) Record(AscendNative.aclrtSynchronizeStream(_stream), "aclrtSynchronizeStream");
            foreach (var tensor in _tensors)
            {
                Record(AscendNative.aclrtFree(tensor.DataPointer), "aclrtFree");
                tensor.MarkDisposed();
            }
            _tensors.Clear();
            if (_stream != 0) Record(AscendNative.aclrtDestroyStream(_stream), "aclrtDestroyStream");
            if (_context != 0) Record(AscendNative.aclrtDestroyContext(_context), "aclrtDestroyContext");
            _stream = _context = 0;
            _disposed = true;
            lock (RuntimeGate)
            {
                if (--_runtimeOwners == 0) Record(AscendNative.aclFinalize(), "aclFinalize");
            }
            if (firstError != 0)
                throw new AscendException(failedOperation, firstError, "Failed during session cleanup.");
        }
    }
}
