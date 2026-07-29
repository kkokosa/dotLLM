namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Minimal lookup surface shared by single-file
/// (<see cref="SafetensorsFile"/>) and future multi-shard safetensors
/// readers. Lets weight loaders accept either shape without branching on
/// the concrete type.
/// </summary>
/// <remarks>
/// <para>
/// The interface intentionally mirrors exactly what consumers (dense
/// transformer loader, MLA / MoE per-layer loaders) actually call:
/// enumeration, by-name lookup, and zero-copy data access. Anything
/// format-specific (header length, data section offset, individual shard
/// files) stays on the concrete type so callers that genuinely need it
/// can down-cast.
/// </para>
/// <para>
/// Implementations must keep the backing memory-mapped regions alive
/// until <see cref="System.IDisposable.Dispose"/> is called — the
/// pointers returned by <see cref="GetTensorPointer(string)"/> and
/// spans returned by <see cref="GetTensorSpan(string)"/> are only valid
/// for that lifetime.
/// </para>
/// </remarks>
public interface ISafetensorsTensorSource : IDisposable
{
    /// <summary>Tensor descriptors in declaration order (flat union across all shards).</summary>
    IReadOnlyList<SafetensorsTensorDescriptor> Tensors { get; }

    /// <summary>Tensor descriptors indexed by name for O(1) lookup.</summary>
    IReadOnlyDictionary<string, SafetensorsTensorDescriptor> TensorsByName { get; }

    /// <summary>
    /// Returns a pointer to the first byte of the named tensor's raw data
    /// in its owning memory-mapped region. Throws if the tensor is unknown.
    /// </summary>
    /// <param name="name">The tensor's fully-qualified name.</param>
    /// <returns>Pointer to the first byte of the tensor data.</returns>
    nint GetTensorPointer(string name);

    /// <summary>
    /// Returns a <see cref="ReadOnlySpan{Byte}"/> over the raw bytes of the
    /// named tensor's data. Valid until this source is disposed. Throws if
    /// the tensor is unknown or its byte count exceeds <c>Int32.MaxValue</c>.
    /// </summary>
    /// <param name="name">The tensor's fully-qualified name.</param>
    /// <returns>Span over the tensor's raw bytes.</returns>
    ReadOnlySpan<byte> GetTensorSpan(string name);
}
