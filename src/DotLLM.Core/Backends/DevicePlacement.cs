namespace DotLLM.Core.Backends;

/// <summary>
/// Identifies the device where a tensor or operation is placed.
/// </summary>
/// <param name="DeviceId">-1 for CPU, 0..N for GPU index.</param>
public readonly record struct DevicePlacement(int DeviceId)
{
    /// <summary>CPU placement.</summary>
    public static readonly DevicePlacement Cpu = new(-1);

    /// <summary>First GPU (device 0).</summary>
    public static readonly DevicePlacement Gpu0 = new(0);

    /// <summary>Creates a GPU placement for the given device index.</summary>
    /// <param name="index">Zero-based GPU index.</param>
    public static DevicePlacement Gpu(int index) => new(index);

    /// <summary>Whether this placement targets a CPU.</summary>
    public bool IsCpu => DeviceId == -1;

    /// <summary>Whether this placement targets a GPU.</summary>
    public bool IsGpu => DeviceId >= 0;

    /// <inheritdoc/>
    public override string ToString() => IsCpu ? "cpu" : $"gpu:{DeviceId}";
}
