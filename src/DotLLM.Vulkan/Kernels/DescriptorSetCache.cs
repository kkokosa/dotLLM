using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Small lookup-by-buffer-handles cache of populated descriptor sets. One
/// instance per kernel — trades a little host-side memory for eliminating
/// <c>vkAllocateDescriptorSets</c> + <c>vkUpdateDescriptorSets</c> on the
/// hot forward path whenever a kernel is called with the same buffer set
/// as a previous call.
/// </summary>
/// <remarks>
/// <para>
/// The cache keys on the descriptor set's buffer handles — not on any
/// push-constant values, because the Vulkan spec lets the same set be
/// rebound under different push constants, and the kernel re-issues
/// <c>vkCmdPushConstants</c> per call anyway. Within one forward pass
/// every kernel is called with many distinct buffer tuples (one per
/// layer × one per projection), but across forwards the tuples repeat —
/// weights are fixed, activation scratch is fixed, only the token sequence
/// changes. So the cache warms up in the first forward and stays warm for
/// the life of the model.
/// </para>
/// <para>
/// Structure is a linear probe over a fixed-capacity array of slots. With
/// <c>Capacity = 256</c> we comfortably cover SmolLM-135M's 211 matmul
/// descriptor variants per forward and still fit larger models; the
/// linear scan cost is trivial compared to a single
/// <c>vkAllocateDescriptorSets</c> round-trip. On overflow the cache
/// resets its own backing pool and drops every entry — this is a slow
/// path that only hits when a caller runs more than <c>Capacity</c>
/// distinct buffer tuples per kernel.
/// </para>
/// </remarks>
internal sealed class DescriptorSetCache
{
    /// <summary>Fixed cache slot count. Must be &gt;= the expected number of distinct buffer tuples per forward.</summary>
    internal const int Capacity = 256;

    /// <summary>Hard upper bound on buffers per descriptor set — matches the widest kernel (attention, 4 bindings).</summary>
    private const int MaxBuffersPerSet = 4;

    private readonly VulkanDevice _device;
    private readonly nint _pool;
    private readonly nint _setLayout;
    private readonly int _buffersPerSet;

    // Parallel arrays indexed by slot. _keys[i] holds MaxBuffersPerSet nints;
    // unused trailing slots are zero. _sets[i] is the descriptor set handle
    // populated for that key; 0 means empty.
    private readonly nint[] _keys;
    private readonly nint[] _sets;
    private int _count;

    public DescriptorSetCache(VulkanDevice device, nint pool, nint setLayout, int buffersPerSet)
    {
        if (buffersPerSet <= 0 || buffersPerSet > MaxBuffersPerSet)
            throw new ArgumentOutOfRangeException(nameof(buffersPerSet));
        _device = device;
        _pool = pool;
        _setLayout = setLayout;
        _buffersPerSet = buffersPerSet;
        _keys = new nint[Capacity * MaxBuffersPerSet];
        _sets = new nint[Capacity];
    }

    /// <summary>
    /// Returns a populated descriptor set for <paramref name="buffers"/> —
    /// allocates + writes one on first call, reuses it thereafter. The
    /// caller owns the cache lifetime; <see cref="Reset"/> drops every
    /// entry (e.g. on pool exhaustion).
    /// </summary>
    public nint GetOrCreate(ReadOnlySpan<nint> buffers)
    {
        if (buffers.Length != _buffersPerSet)
            throw new ArgumentException(
                $"Expected {_buffersPerSet} buffers, got {buffers.Length}.", nameof(buffers));

        // Linear scan — 256 entries × up-to-4 pointer comparisons is
        // ~a microsecond, well below vkAllocateDescriptorSets latency.
        for (int i = 0; i < _count; i++)
        {
            if (Matches(i, buffers))
                return _sets[i];
        }

        // Miss — allocate + write + insert.
        if (_count >= Capacity)
        {
            // Cache full. Reset the entire pool and drop all entries.
            // The caller's kernel code is then free to allocate fresh.
            Reset();
        }

        nint set = KernelSupport.AllocateDescriptorSet(_device, _pool, _setLayout);
        KernelSupport.WriteBufferBindings(_device, set, buffers);

        int slot = _count;
        int baseIdx = slot * MaxBuffersPerSet;
        for (int j = 0; j < _buffersPerSet; j++)
            _keys[baseIdx + j] = buffers[j];
        _sets[slot] = set;
        _count++;
        return set;
    }

    /// <summary>
    /// Forgets every cached entry and resets the underlying descriptor
    /// pool. Call when the caller has externally invalidated the sets
    /// (e.g. the kernel's scratch buffers were re-allocated).
    /// </summary>
    public void Reset()
    {
        VulkanApi.vkResetDescriptorPool(_device.Handle, _pool, 0)
            .ThrowOnError("vkResetDescriptorPool DescriptorSetCache");
        Array.Clear(_keys);
        Array.Clear(_sets);
        _count = 0;
    }

    private bool Matches(int slot, ReadOnlySpan<nint> buffers)
    {
        int baseIdx = slot * MaxBuffersPerSet;
        for (int j = 0; j < _buffersPerSet; j++)
        {
            if (_keys[baseIdx + j] != buffers[j])
                return false;
        }
        return true;
    }
}
