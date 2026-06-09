using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Represents a Vulkan logical device bound to a single physical GPU plus a
/// compute queue and command pool. Owns the instance, device, and allocator
/// state; disposal tears everything down in reverse order.
/// </summary>
/// <remarks>
/// Scaffold semantics — proof-of-pipeline only:
/// <list type="bullet">
///   <item>No fence-based pipelining. Submits are synchronous (<c>vkQueueWaitIdle</c>).</item>
///   <item>No staging buffers. Device memory is allocated <c>HostVisible|HostCoherent</c>
///     so uploads/downloads hit the same VRAM region — fine for small tests, not for
///     large model weights. A proper arena + staging ring lands with the first real kernel.</item>
///   <item>Single queue. Multi-queue (transfer/compute separation) is deferred.</item>
/// </list>
/// </remarks>
public sealed class VulkanDevice : IDisposable
{
    private nint _instance;
    private nint _physicalDevice;
    private nint _device;
    private nint _queue;
    private nint _commandPool;
    private bool _disposed;

    /// <summary>Device name (e.g. "AMD Radeon RX 7900 XT", "NVIDIA GeForce RTX 4090").</summary>
    public string DeviceName { get; }

    /// <summary>PCI vendor ID (0x10DE = NVIDIA, 0x1002 = AMD, 0x8086 = Intel).</summary>
    public uint VendorId { get; }

    /// <summary>Vulkan device type (discrete, integrated, virtual, CPU).</summary>
    public int DeviceType { get; }

    /// <summary>Queue family index selected for compute.</summary>
    public uint QueueFamilyIndex { get; }

    internal nint Handle => _device;
    internal nint Queue => _queue;
    internal nint CommandPool => _commandPool;
    internal nint PhysicalDevice => _physicalDevice;

    private VulkanDevice(
        nint instance, nint physical, nint device, nint queue,
        nint commandPool, string name, uint vendor, int type, uint queueFamily)
    {
        _instance = instance;
        _physicalDevice = physical;
        _device = device;
        _queue = queue;
        _commandPool = commandPool;
        DeviceName = name;
        VendorId = vendor;
        DeviceType = type;
        QueueFamilyIndex = queueFamily;
    }

    /// <summary>
    /// Probes whether a Vulkan loader is present and whether <c>vkCreateInstance</c>
    /// succeeds on this machine. Does not throw.
    /// </summary>
    public static bool IsAvailable()
    {
        try
        {
            string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "vulkan-1.dll"
                : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                    ? "libvulkan.dylib"
                    : "libvulkan.so.1";
            if (!NativeLibrary.TryLoad(lib, out nint handle))
                return false;
            NativeLibrary.Free(handle);

            return ProbeInstance();
        }
        catch
        {
            return false;
        }
    }

    // Isolated so the JIT only resolves VulkanApi P/Invokes when the loader is confirmed present.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool ProbeInstance()
    {
        VulkanLibraryResolver.Register();
        nint inst = CreateInstance();
        if (inst == 0) return false;
        try
        {
            uint count = 0;
            int r = VulkanApi.vkEnumeratePhysicalDevices(inst, ref count, null);
            return r >= 0 && count > 0;
        }
        finally
        {
            VulkanApi.vkDestroyInstance(inst, 0);
        }
    }

    /// <summary>
    /// Creates a Vulkan device bound to the first suitable GPU.
    /// Selection order: discrete GPU (preferring AMD/NVIDIA over Intel) → integrated → first available.
    /// </summary>
    public static VulkanDevice Create()
    {
        VulkanLibraryResolver.Register();
        nint instance = CreateInstance();
        if (instance == 0)
            throw new VulkanException(-3, "vkCreateInstance failed — no Vulkan loader or driver available.");

        try
        {
            nint physical = SelectPhysicalDevice(instance, out string name, out uint vendor, out int type);
            uint queueFamily = SelectComputeQueueFamily(physical);
            nint device = CreateLogicalDevice(physical, queueFamily);

            VulkanApi.vkGetDeviceQueue(device, queueFamily, 0, out nint queue);

            var cpInfo = new VkCommandPoolCreateInfo
            {
                sType = VkStructureType.CommandPoolCreateInfo,
                flags = VkCommandPoolCreateFlags.ResetCommandBuffer,
                queueFamilyIndex = queueFamily,
            };
            VulkanApi.vkCreateCommandPool(device, cpInfo, 0, out nint pool)
                .ThrowOnError("vkCreateCommandPool");

            // Transfer ownership of instance to the device on success.
            var result = new VulkanDevice(instance, physical, device, queue, pool, name, vendor, type, queueFamily);
            instance = 0;
            return result;
        }
        finally
        {
            if (instance != 0)
                VulkanApi.vkDestroyInstance(instance, 0);
        }
    }

    private static nint CreateInstance()
    {
        // VK_MAKE_API_VERSION(0, 1, 2, 0) = Vulkan 1.2
        const uint apiVersion = (1u << 22) | (2u << 12);

        // Note: pApplicationName / pEngineName left null — we don't need strings.
        var appInfo = new VkApplicationInfo
        {
            sType = VkStructureType.ApplicationInfo,
            apiVersion = apiVersion,
        };

        unsafe
        {
            VkInstanceCreateInfo ci = default;
            ci.sType = VkStructureType.InstanceCreateInfo;
            ci.pApplicationInfo = (nint)(&appInfo);
            int r = VulkanApi.vkCreateInstance(ci, 0, out nint inst);
            return r >= 0 ? inst : 0;
        }
    }

    private static nint SelectPhysicalDevice(
        nint instance, out string name, out uint vendor, out int type)
    {
        uint count = 0;
        VulkanApi.vkEnumeratePhysicalDevices(instance, ref count, null)
            .ThrowOnError("vkEnumeratePhysicalDevices (count)");
        if (count == 0)
            throw new VulkanException(-3, "No Vulkan physical devices found.");

        var devices = new nint[count];
        VulkanApi.vkEnumeratePhysicalDevices(instance, ref count, devices)
            .ThrowOnError("vkEnumeratePhysicalDevices");

        // Score every device. Prefer: discrete > integrated > other/CPU.
        // Within discrete, prefer AMD/NVIDIA over Intel (Intel rarely has dGPUs,
        // but if one is present it's often weaker than an AMD/NVIDIA dGPU).
        nint bestDev = 0;
        int bestScore = int.MinValue;
        string bestName = "unknown";
        uint bestVendor = 0;
        int bestType = 0;

        foreach (var dev in devices)
        {
            VulkanApi.vkGetPhysicalDeviceProperties(dev, out var props);
            string devName = ReadDeviceName(props);
            int score = ScoreDevice(props.deviceType, props.vendorID);

            if (score > bestScore)
            {
                bestScore = score;
                bestDev = dev;
                bestName = devName;
                bestVendor = props.vendorID;
                bestType = props.deviceType;
            }
        }

        name = bestName;
        vendor = bestVendor;
        type = bestType;
        return bestDev;
    }

    // Vendor IDs are PCI SIG assignments. 0x10DE=NVIDIA, 0x1002=AMD, 0x8086=Intel, 0x13B5=ARM, 0x5143=Qualcomm.
    private static int ScoreDevice(int deviceType, uint vendorId)
    {
        int typeScore = deviceType switch
        {
            VkPhysicalDeviceType.DiscreteGpu => 1000,
            VkPhysicalDeviceType.IntegratedGpu => 500,
            VkPhysicalDeviceType.VirtualGpu => 100,
            _ => 0,
        };
        int vendorScore = vendorId switch
        {
            0x10DE => 20, // NVIDIA
            0x1002 => 20, // AMD
            0x8086 => 10, // Intel — lower preference when a dGPU is also present
            _ => 5,
        };
        return typeScore + vendorScore;
    }

    private static unsafe string ReadDeviceName(VkPhysicalDeviceProperties props)
    {
        byte* p = props.deviceName;
        int len = 0;
        while (len < 256 && p[len] != 0) len++;
        return Encoding.UTF8.GetString(p, len);
    }

    private static uint SelectComputeQueueFamily(nint physical)
    {
        uint count = 0;
        VulkanApi.vkGetPhysicalDeviceQueueFamilyProperties(physical, ref count, null);
        if (count == 0)
            throw new VulkanException(-3, "Physical device reports zero queue families.");

        var families = new VkQueueFamilyProperties[count];
        VulkanApi.vkGetPhysicalDeviceQueueFamilyProperties(physical, ref count, families);

        // Pick the first family that supports COMPUTE. A dedicated compute-only
        // queue (compute without graphics) is nice-to-have but not required for
        // this scaffold.
        for (uint i = 0; i < count; i++)
        {
            if ((families[i].queueFlags & VkQueueFlags.Compute) != 0)
                return i;
        }
        throw new VulkanException(-3, "No queue family with COMPUTE capability.");
    }

    private static unsafe nint CreateLogicalDevice(nint physical, uint queueFamily)
    {
        float priority = 1.0f;

        var qci = new VkDeviceQueueCreateInfo
        {
            sType = VkStructureType.DeviceQueueCreateInfo,
            queueFamilyIndex = queueFamily,
            queueCount = 1,
            pQueuePriorities = (nint)(&priority),
        };

        VkDeviceCreateInfo ci = default;
        ci.sType = VkStructureType.DeviceCreateInfo;
        ci.queueCreateInfoCount = 1;
        ci.pQueueCreateInfos = (nint)(&qci);

        VulkanApi.vkCreateDevice(physical, ci, 0, out nint dev)
            .ThrowOnError("vkCreateDevice");
        return dev;
    }

    // ────────────────────────────────────────────────────────────────
    // Buffer & memory helpers
    // ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Device-owned buffer + backing memory. Caller owns the <see cref="IDisposable"/>.
    /// </summary>
    public sealed class Buffer : IDisposable
    {
        private readonly VulkanDevice _device;
        private nint _buffer;
        private nint _memory;

        /// <summary>Buffer size in bytes.</summary>
        public long Size { get; }

        /// <summary>Underlying <c>VkBuffer</c> handle.</summary>
        public nint Handle => _buffer;

        internal Buffer(VulkanDevice device, nint buffer, nint memory, long size)
        {
            _device = device;
            _buffer = buffer;
            _memory = memory;
            Size = size;
        }

        /// <summary>Underlying <c>VkDeviceMemory</c> handle.</summary>
        public nint Memory => _memory;

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_buffer != 0)
            {
                VulkanApi.vkDestroyBuffer(_device._device, _buffer, 0);
                _buffer = 0;
            }
            if (_memory != 0)
            {
                VulkanApi.vkFreeMemory(_device._device, _memory, 0);
                _memory = 0;
            }
        }
    }

    /// <summary>
    /// Allocates a storage buffer of <paramref name="bytes"/> bytes backed by
    /// host-visible, host-coherent device memory. The returned buffer can be
    /// mapped directly from the host — use for activations / scratch the
    /// forward pass reads/writes from the host between kernel launches.
    /// </summary>
    public Buffer Allocate(long bytes) => AllocateInternal(bytes, deviceLocal: false);

    /// <summary>
    /// Allocates a storage buffer of <paramref name="bytes"/> bytes backed by
    /// device-local memory. The buffer is <b>not</b> host-mappable; use this
    /// for immutable weights and the KV cache, populating the contents via
    /// <see cref="UploadToDeviceLocal"/> (weights) or <c>vkCmdCopyBuffer</c>
    /// between a host-visible source and this device-local destination
    /// (KV cache update path).
    /// </summary>
    /// <remarks>
    /// On discrete GPUs this puts the data in VRAM — reads from a compute
    /// shader hit the driver's native tiled layout rather than going over
    /// PCIe / DF at host-memory bandwidth. On UMA parts (iGPU, APU) the
    /// bytes still physically sit in shared DDR, but the driver picks a
    /// swizzled storage layout that reads significantly faster from a
    /// compute shader than host-coherent linear memory. Always measure.
    /// </remarks>
    public Buffer AllocateDeviceLocal(long bytes) => AllocateInternal(bytes, deviceLocal: true);

    private Buffer AllocateInternal(long bytes, bool deviceLocal)
    {
        if (bytes <= 0) throw new ArgumentOutOfRangeException(nameof(bytes));

        var bci = new VkBufferCreateInfo
        {
            sType = VkStructureType.BufferCreateInfo,
            size = (ulong)bytes,
            usage = VkBufferUsageFlags.StorageBuffer
                  | VkBufferUsageFlags.TransferSrc
                  | VkBufferUsageFlags.TransferDst,
            sharingMode = VkSharingMode.Exclusive,
        };
        VulkanApi.vkCreateBuffer(_device, bci, 0, out nint buffer)
            .ThrowOnError("vkCreateBuffer");

        VulkanApi.vkGetBufferMemoryRequirements(_device, buffer, out var req);

        VkMemoryPropertyFlags required = deviceLocal
            ? VkMemoryPropertyFlags.DeviceLocal
            : VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent;

        // On UMA drivers (AMD integrated, Intel) every memory type may expose
        // DEVICE_LOCAL + HOST_VISIBLE simultaneously. For weights we prefer a
        // strictly device-local-only type (driver is free to use a tiled /
        // swizzled layout — see AllocateDeviceLocal remarks). Fall back to a
        // DEVICE_LOCAL-that-is-also-host-visible type when the GPU only
        // exposes the combined pool (older Intel, some mobile).
        uint typeIndex;
        if (deviceLocal)
        {
            if (!TryFindMemoryType(req.memoryTypeBits,
                    required: VkMemoryPropertyFlags.DeviceLocal,
                    excluded: VkMemoryPropertyFlags.HostVisible,
                    out typeIndex))
            {
                typeIndex = FindMemoryType(req.memoryTypeBits, VkMemoryPropertyFlags.DeviceLocal);
            }
        }
        else
        {
            typeIndex = FindMemoryType(req.memoryTypeBits, required);
        }

        var mai = new VkMemoryAllocateInfo
        {
            sType = VkStructureType.MemoryAllocateInfo,
            allocationSize = req.size,
            memoryTypeIndex = typeIndex,
        };
        int allocResult = VulkanApi.vkAllocateMemory(_device, mai, 0, out nint memory);
        if (allocResult < 0)
        {
            VulkanApi.vkDestroyBuffer(_device, buffer, 0);
            allocResult.ThrowOnError("vkAllocateMemory");
        }

        int bindResult = VulkanApi.vkBindBufferMemory(_device, buffer, memory, 0);
        if (bindResult < 0)
        {
            VulkanApi.vkFreeMemory(_device, memory, 0);
            VulkanApi.vkDestroyBuffer(_device, buffer, 0);
            bindResult.ThrowOnError("vkBindBufferMemory");
        }

        return new Buffer(this, buffer, memory, bytes);
    }

    /// <summary>
    /// Copies <paramref name="source"/> bytes from host memory into
    /// <paramref name="dst"/> (which may be device-local, i.e. not
    /// host-mappable) via an intermediate <paramref name="staging"/> buffer.
    /// Records a <c>vkCmdCopyBuffer</c> on a transient command buffer and
    /// waits on a fence. <paramref name="staging"/> must be host-visible
    /// host-coherent and at least <paramref name="source"/>.Length bytes.
    /// </summary>
    /// <remarks>
    /// This is the weight-upload path. Callers pre-allocate one staging
    /// buffer sized for the largest single weight row/matrix and reuse it
    /// across all <c>vkCmdCopyBuffer</c> uploads — saves the per-upload
    /// <c>vkAllocateMemory</c>/<c>vkCreateBuffer</c> cost that would dominate
    /// at 30 layers × 7 matrices.
    /// </remarks>
    public unsafe void UploadToDeviceLocal(ReadOnlySpan<byte> source, Buffer staging, Buffer dst)
    {
        if (source.Length > staging.Size)
            throw new ArgumentException("Staging buffer too small.", nameof(staging));
        if (source.Length > dst.Size)
            throw new ArgumentException("Destination buffer too small.", nameof(dst));

        // 1. Copy host → staging.
        VulkanApi.vkMapMemory(_device, staging.Memory, 0, (ulong)source.Length, 0, out nint mapped)
            .ThrowOnError("vkMapMemory staging");
        try
        {
            source.CopyTo(new Span<byte>((void*)mapped, source.Length));
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device, staging.Memory);
        }

        // 2. Record + submit staging → dst copy, wait on fence.
        CopyBufferSynchronous(staging, dst, (ulong)source.Length);
    }

    /// <summary>
    /// Records a one-shot <c>vkCmdCopyBuffer</c> from offset 0 of
    /// <paramref name="src"/> to offset 0 of <paramref name="dst"/> and waits
    /// for it on a fence. Used by the device-local weight-upload path.
    /// </summary>
    public void CopyBufferSynchronous(Buffer src, Buffer dst, ulong size)
        => CopyBufferRangeSynchronous(src, dst, srcOffset: 0, dstOffset: 0, size: size);

    /// <summary>
    /// Records a one-shot <c>vkCmdCopyBuffer</c> between arbitrary offsets
    /// and waits for it on a fence. Used by the synchronous KV-cache update
    /// path (the fence-pipelined path uses <c>vkCmdCopyBuffer</c> directly
    /// against the forward pass's shared command buffer).
    /// </summary>
    public unsafe void CopyBufferRangeSynchronous(Buffer src, Buffer dst, ulong srcOffset, ulong dstOffset, ulong size)
    {
        var cbai = new VkCommandBufferAllocateInfo
        {
            sType = VkStructureType.CommandBufferAllocateInfo,
            commandPool = _commandPool,
            level = VkCommandBufferLevel.Primary,
            commandBufferCount = 1,
        };
        VulkanApi.vkAllocateCommandBuffers(_device, cbai, out nint cmdBuf)
            .ThrowOnError("vkAllocateCommandBuffers CopyBufferRangeSynchronous");

        var fenceCi = new VkFenceCreateInfo { sType = VkStructureType.FenceCreateInfo };
        VulkanApi.vkCreateFence(_device, fenceCi, 0, out nint fence)
            .ThrowOnError("vkCreateFence CopyBufferRangeSynchronous");

        try
        {
            var begin = new VkCommandBufferBeginInfo
            {
                sType = VkStructureType.CommandBufferBeginInfo,
                flags = VkCommandBufferUsageFlags.OneTimeSubmit,
            };
            VulkanApi.vkBeginCommandBuffer(cmdBuf, begin).ThrowOnError("vkBeginCommandBuffer CopyBufferRangeSynchronous");

            var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);

            VulkanApi.vkEndCommandBuffer(cmdBuf).ThrowOnError("vkEndCommandBuffer CopyBufferRangeSynchronous");

            var submit = new VkSubmitInfo
            {
                sType = VkStructureType.SubmitInfo,
                commandBufferCount = 1,
                pCommandBuffers = (nint)(&cmdBuf),
            };
            VulkanApi.vkQueueSubmit(_queue, 1, submit, fence).ThrowOnError("vkQueueSubmit CopyBufferRangeSynchronous");

            nint fenceLocal = fence;
            VulkanApi.vkWaitForFences(_device, 1, fenceLocal, waitAll: 1, ulong.MaxValue)
                .ThrowOnError("vkWaitForFences CopyBufferRangeSynchronous");
        }
        finally
        {
            VulkanApi.vkDestroyFence(_device, fence, 0);
            VulkanApi.vkFreeCommandBuffers(_device, _commandPool, 1, cmdBuf);
        }
    }

    private unsafe uint FindMemoryType(uint typeBits, VkMemoryPropertyFlags required)
    {
        if (TryFindMemoryType(typeBits, required, excluded: default, out uint idx))
            return idx;
        throw new VulkanException(-3,
            $"No memory type satisfies typeBits=0x{typeBits:X8} and flags={required}.");
    }

    private unsafe bool TryFindMemoryType(
        uint typeBits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags excluded,
        out uint memoryTypeIndex)
    {
        VulkanApi.vkGetPhysicalDeviceMemoryProperties(_physicalDevice, out var mem);
        // memoryTypes is an array of 8-byte entries: u32 propertyFlags, u32 heapIndex.
        uint* types = (uint*)mem.memoryTypes;
        for (uint i = 0; i < mem.memoryTypeCount; i++)
        {
            if ((typeBits & (1u << (int)i)) == 0) continue;
            var flags = (VkMemoryPropertyFlags)types[i * 2];
            if ((flags & required) != required) continue;
            if (excluded != default && (flags & excluded) != 0) continue;
            memoryTypeIndex = i;
            return true;
        }
        memoryTypeIndex = 0;
        return false;
    }

    /// <summary>Copies <paramref name="source"/> from host memory into the start of <paramref name="dst"/>.</summary>
    public unsafe void Upload(ReadOnlySpan<float> source, Buffer dst)
    {
        long bytes = (long)source.Length * sizeof(float);
        if (bytes > dst.Size)
            throw new ArgumentException("Source larger than destination buffer.", nameof(source));

        VulkanApi.vkMapMemory(_device, dst.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory");
        try
        {
            var destSpan = new Span<float>((void*)mapped, source.Length);
            source.CopyTo(destSpan);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device, dst.Memory);
        }
    }

    /// <summary>
    /// Copies raw <paramref name="source"/> bytes from host memory into the start of <paramref name="dst"/>.
    /// Used for quantized weight blobs (Q8_0, Q4_K, etc.) where the GPU sees the
    /// data as <c>uint[]</c> and the shader extracts bytes.
    /// </summary>
    public unsafe void Upload(ReadOnlySpan<byte> source, Buffer dst)
    {
        if (source.Length > dst.Size)
            throw new ArgumentException("Source larger than destination buffer.", nameof(source));

        VulkanApi.vkMapMemory(_device, dst.Memory, 0, (ulong)source.Length, 0, out nint mapped)
            .ThrowOnError("vkMapMemory");
        try
        {
            var destSpan = new Span<byte>((void*)mapped, source.Length);
            source.CopyTo(destSpan);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device, dst.Memory);
        }
    }

    /// <summary>Copies from the start of <paramref name="src"/> into <paramref name="destination"/> host memory.</summary>
    public unsafe void Download(Buffer src, Span<float> destination)
    {
        long bytes = (long)destination.Length * sizeof(float);
        if (bytes > src.Size)
            throw new ArgumentException("Destination larger than source buffer.", nameof(destination));

        VulkanApi.vkMapMemory(_device, src.Memory, 0, (ulong)bytes, 0, out nint mapped)
            .ThrowOnError("vkMapMemory");
        try
        {
            var srcSpan = new ReadOnlySpan<float>((void*)mapped, destination.Length);
            srcSpan.CopyTo(destination);
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device, src.Memory);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_device != 0)
        {
            VulkanApi.vkDeviceWaitIdle(_device);
        }
        if (_commandPool != 0)
        {
            VulkanApi.vkDestroyCommandPool(_device, _commandPool, 0);
            _commandPool = 0;
        }
        if (_device != 0)
        {
            VulkanApi.vkDestroyDevice(_device, 0);
            _device = 0;
        }
        if (_instance != 0)
        {
            VulkanApi.vkDestroyInstance(_instance, 0);
            _instance = 0;
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Forward-pass command submission
    // ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Reusable command-buffer + fence pair used by the fence-pipelined
    /// forward pass. One instance per <see cref="VulkanTransformerModel"/>;
    /// <see cref="Begin"/> resets and opens the buffer, <see cref="SubmitAndWait"/>
    /// submits and waits on the fence, leaving both ready for the next forward.
    /// </summary>
    public sealed class SubmitContext : IDisposable
    {
        private readonly VulkanDevice _device;
        private nint _cmdBuf;
        private nint _fence;
        private bool _disposed;

        /// <summary>Underlying command buffer. Valid between <see cref="Begin"/> and <see cref="SubmitAndWait"/>.</summary>
        public nint CommandBuffer => _cmdBuf;

        internal SubmitContext(VulkanDevice device, nint cmdBuf, nint fence)
        {
            _device = device;
            _cmdBuf = cmdBuf;
            _fence = fence;
        }

        /// <summary>
        /// Resets the command buffer (and the fence) and opens the buffer for
        /// recording. Call once at the start of each forward pass.
        /// </summary>
        public void Begin()
        {
            VulkanApi.vkResetCommandBuffer(_cmdBuf, 0).ThrowOnError("vkResetCommandBuffer");
            var begin = new VkCommandBufferBeginInfo
            {
                sType = VkStructureType.CommandBufferBeginInfo,
                flags = VkCommandBufferUsageFlags.OneTimeSubmit,
            };
            VulkanApi.vkBeginCommandBuffer(_cmdBuf, begin).ThrowOnError("vkBeginCommandBuffer");
        }

        /// <summary>
        /// Ends the command buffer, submits on the queue, waits on the fence,
        /// resets the fence for reuse. Call once at the end of each forward
        /// pass.
        /// </summary>
        public unsafe void SubmitAndWait()
        {
            VulkanApi.vkEndCommandBuffer(_cmdBuf).ThrowOnError("vkEndCommandBuffer");

            nint cmdBufLocal = _cmdBuf;
            var submit = new VkSubmitInfo
            {
                sType = VkStructureType.SubmitInfo,
                commandBufferCount = 1,
                pCommandBuffers = (nint)(&cmdBufLocal),
            };
            VulkanApi.vkQueueSubmit(_device._queue, 1, submit, _fence).ThrowOnError("vkQueueSubmit SubmitContext");

            nint fenceLocal = _fence;
            VulkanApi.vkWaitForFences(_device._device, 1, fenceLocal, waitAll: 1, ulong.MaxValue)
                .ThrowOnError("vkWaitForFences SubmitContext");
            VulkanApi.vkResetFences(_device._device, 1, fenceLocal).ThrowOnError("vkResetFences SubmitContext");
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (_fence != 0)
            {
                VulkanApi.vkDestroyFence(_device._device, _fence, 0);
                _fence = 0;
            }
            if (_cmdBuf != 0)
            {
                nint local = _cmdBuf;
                VulkanApi.vkFreeCommandBuffers(_device._device, _device._commandPool, 1, local);
                _cmdBuf = 0;
            }
        }
    }

    /// <summary>
    /// Allocates one command buffer and one fence bound to the compute
    /// queue. The returned <see cref="SubmitContext"/> is intended to live
    /// for the lifetime of the caller (e.g. <see cref="VulkanTransformerModel"/>)
    /// and be reused <see cref="SubmitContext.Begin"/>-&gt;record-&gt;
    /// <see cref="SubmitContext.SubmitAndWait"/> once per forward pass.
    /// </summary>
    public SubmitContext CreateSubmitContext()
    {
        var cbai = new VkCommandBufferAllocateInfo
        {
            sType = VkStructureType.CommandBufferAllocateInfo,
            commandPool = _commandPool,
            level = VkCommandBufferLevel.Primary,
            commandBufferCount = 1,
        };
        VulkanApi.vkAllocateCommandBuffers(_device, cbai, out nint cmdBuf)
            .ThrowOnError("vkAllocateCommandBuffers CreateSubmitContext");

        var fenceCi = new VkFenceCreateInfo { sType = VkStructureType.FenceCreateInfo };
        int r = VulkanApi.vkCreateFence(_device, fenceCi, 0, out nint fence);
        if (r < 0)
        {
            nint local = cmdBuf;
            VulkanApi.vkFreeCommandBuffers(_device, _commandPool, 1, local);
            r.ThrowOnError("vkCreateFence CreateSubmitContext");
        }

        return new SubmitContext(this, cmdBuf, fence);
    }
}
