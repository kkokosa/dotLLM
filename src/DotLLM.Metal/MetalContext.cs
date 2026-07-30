using DotLLM.Metal.Interop;

namespace DotLLM.Metal;

/// <summary>
/// Owns a native Metal context: device, command queue, and compiled pipeline cache.
/// Pipelines are compiled once on first use and reused across calls.
/// </summary>
/// <remarks>
/// Create once at startup, share across operations, dispose at shutdown.
/// </remarks>
public sealed class MetalContext : IDisposable
{
    private nint _handle;

    /// <summary>
    /// Initializes a new Metal context backed by the system default GPU.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no Metal-capable device is found or context allocation fails.
    /// </exception>
    public MetalContext()
    {
        _handle = MetalNative.CreateContext();
        if (_handle == 0)
            throw new InvalidOperationException("Failed to create Metal context. No Metal-capable device found.");
    }

    /// <summary>Opaque native handle passed to kernel calls.</summary>
    internal nint Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_handle == 0, this);
            return _handle;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_handle != 0)
        {
            MetalNative.DestroyContext(_handle);
            _handle = 0;
        }
    }
}
