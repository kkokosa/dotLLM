using System.Runtime.InteropServices;

namespace DotLLM.Vulkan.Interop;

// Vulkan uses int32 "structure type" tags (sType) on every struct to permit
// forward extension. Only the tags we actually use are listed here.
internal static class VkStructureType
{
    internal const int ApplicationInfo = 0;
    internal const int InstanceCreateInfo = 1;
    internal const int DeviceQueueCreateInfo = 2;
    internal const int DeviceCreateInfo = 3;
    internal const int SubmitInfo = 4;
    internal const int MemoryAllocateInfo = 5;
    internal const int MappedMemoryRange = 6;
    internal const int BindSparseInfo = 7;
    internal const int FenceCreateInfo = 8;
    internal const int BufferCreateInfo = 12;
    internal const int ShaderModuleCreateInfo = 16;
    internal const int PipelineLayoutCreateInfo = 30;
    internal const int ComputePipelineCreateInfo = 29;
    internal const int PipelineShaderStageCreateInfo = 18;
    internal const int DescriptorSetLayoutCreateInfo = 32;
    internal const int DescriptorPoolCreateInfo = 33;
    internal const int DescriptorSetAllocateInfo = 34;
    internal const int WriteDescriptorSet = 35;
    internal const int CommandPoolCreateInfo = 39;
    internal const int CommandBufferAllocateInfo = 40;
    internal const int CommandBufferBeginInfo = 42;
}

// VkPhysicalDeviceType (chosen enum values)
internal static class VkPhysicalDeviceType
{
    internal const int Other = 0;
    internal const int IntegratedGpu = 1;
    internal const int DiscreteGpu = 2;
    internal const int VirtualGpu = 3;
    internal const int Cpu = 4;
}

// VkBufferUsageFlagBits (bitflags)
[Flags]
internal enum VkBufferUsageFlags : uint
{
    TransferSrc = 0x00000001,
    TransferDst = 0x00000002,
    StorageBuffer = 0x00000020,
}

// VkMemoryPropertyFlagBits (bitflags)
[Flags]
internal enum VkMemoryPropertyFlags : uint
{
    DeviceLocal = 0x00000001,
    HostVisible = 0x00000002,
    HostCoherent = 0x00000004,
    HostCached = 0x00000008,
}

[Flags]
internal enum VkMemoryHeapFlags : uint
{
    DeviceLocal = 0x00000001,
}

[Flags]
internal enum VkQueueFlags : uint
{
    Graphics = 0x00000001,
    Compute = 0x00000002,
    Transfer = 0x00000004,
    SparseBinding = 0x00000008,
}

internal static class VkDescriptorType
{
    internal const int StorageBuffer = 7;
}

internal static class VkShaderStageFlags
{
    internal const uint Compute = 0x00000020;
}

internal static class VkCommandPoolCreateFlags
{
    internal const uint ResetCommandBuffer = 0x00000002;
}

internal static class VkCommandBufferLevel
{
    internal const int Primary = 0;
}

internal static class VkCommandBufferUsageFlags
{
    internal const uint OneTimeSubmit = 0x00000001;
}

internal static class VkSharingMode
{
    internal const int Exclusive = 0;
}

internal static class VkPipelineBindPoint
{
    internal const int Compute = 1;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkApplicationInfo
{
    internal int sType;
    internal nint pNext;
    internal nint pApplicationName;
    internal uint applicationVersion;
    internal nint pEngineName;
    internal uint engineVersion;
    internal uint apiVersion;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkInstanceCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal nint pApplicationInfo;
    internal uint enabledLayerCount;
    internal nint ppEnabledLayerNames;
    internal uint enabledExtensionCount;
    internal nint ppEnabledExtensionNames;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDeviceQueueCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint queueFamilyIndex;
    internal uint queueCount;
    internal nint pQueuePriorities;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDeviceCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint queueCreateInfoCount;
    internal nint pQueueCreateInfos;
    internal uint enabledLayerCount;
    internal nint ppEnabledLayerNames;
    internal uint enabledExtensionCount;
    internal nint ppEnabledExtensionNames;
    internal nint pEnabledFeatures;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkQueueFamilyProperties
{
    internal VkQueueFlags queueFlags;
    internal uint queueCount;
    internal uint timestampValidBits;
    // VkExtent3D minImageTransferGranularity
    internal uint minTransferWidth;
    internal uint minTransferHeight;
    internal uint minTransferDepth;
}

// VkPhysicalDeviceProperties is a large struct with VkPhysicalDeviceLimits
// and VkPhysicalDeviceSparseProperties tails. We only need the header fields
// (apiVersion..deviceName). The tail is reserved as an oversized byte buffer
// to ensure the native callee has enough space to write without blowing the
// stack — we never read those bytes.
//
// Upper-bound size: Vulkan 1.3 reports the total is 824 bytes; rounding up
// to 2048 gives plenty of headroom across any future extension and avoids
// maintenance when minor versions add fields at the tail.
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceProperties
{
    internal uint apiVersion;
    internal uint driverVersion;
    internal uint vendorID;
    internal uint deviceID;
    internal int deviceType;
    internal fixed byte deviceName[256]; // VK_MAX_PHYSICAL_DEVICE_NAME_SIZE
    internal fixed byte pipelineCacheUUID[16];
    // Limits + SparseProperties tail — intentionally oversized.
    internal fixed byte tail[2048];
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct VkPhysicalDeviceMemoryProperties
{
    internal uint memoryTypeCount;
    // 32 * VkMemoryType, each 2 u32 words: [i*2] = propertyFlags, [i*2+1] = heapIndex.
    // Declared as uint (not byte) so consumers index naturally-aligned words rather than
    // reinterpreting a byte buffer through a uint* — which would be an unaligned read on
    // targets that require it (some ARM configurations).
    internal fixed uint memoryTypeWords[32 * 2];
    internal uint memoryHeapCount;
    // 16 * VkMemoryHeap, each 2 u64 words: [i*2] = size, [i*2+1] = flags (u32 + padding).
    // ulong element type also gives the struct the 8-byte alignment VkMemoryHeap carries
    // in the C layout, keeping memoryHeaps at the same offset the loader writes.
    internal fixed ulong memoryHeapWords[16 * 2];
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryRequirements
{
    internal ulong size;
    internal ulong alignment;
    internal uint memoryTypeBits;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkMemoryAllocateInfo
{
    internal int sType;
    internal nint pNext;
    internal ulong allocationSize;
    internal uint memoryTypeIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkBufferCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal ulong size;
    internal VkBufferUsageFlags usage;
    internal int sharingMode;
    internal uint queueFamilyIndexCount;
    internal nint pQueueFamilyIndices;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkShaderModuleCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal nuint codeSize;
    internal nint pCode; // uint32_t array
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetLayoutBinding
{
    internal uint binding;
    internal int descriptorType;
    internal uint descriptorCount;
    internal uint stageFlags;
    internal nint pImmutableSamplers;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetLayoutCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint bindingCount;
    internal nint pBindings;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPushConstantRange
{
    internal uint stageFlags;
    internal uint offset;
    internal uint size;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPipelineLayoutCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint setLayoutCount;
    internal nint pSetLayouts;
    internal uint pushConstantRangeCount;
    internal nint pPushConstantRanges;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkPipelineShaderStageCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint stage;
    internal nint module;
    internal nint pName; // entry-point name, null-terminated UTF-8
    internal nint pSpecializationInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkComputePipelineCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal VkPipelineShaderStageCreateInfo stage;
    internal nint layout;
    internal nint basePipelineHandle;
    internal int basePipelineIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorPoolSize
{
    internal int type;
    internal uint descriptorCount;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorPoolCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint maxSets;
    internal uint poolSizeCount;
    internal nint pPoolSizes;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorSetAllocateInfo
{
    internal int sType;
    internal nint pNext;
    internal nint descriptorPool;
    internal uint descriptorSetCount;
    internal nint pSetLayouts;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkDescriptorBufferInfo
{
    internal nint buffer;
    internal ulong offset;
    internal ulong range;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkWriteDescriptorSet
{
    internal int sType;
    internal nint pNext;
    internal nint dstSet;
    internal uint dstBinding;
    internal uint dstArrayElement;
    internal uint descriptorCount;
    internal int descriptorType;
    internal nint pImageInfo;
    internal nint pBufferInfo;
    internal nint pTexelBufferView;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandPoolCreateInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal uint queueFamilyIndex;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandBufferAllocateInfo
{
    internal int sType;
    internal nint pNext;
    internal nint commandPool;
    internal int level;
    internal uint commandBufferCount;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkCommandBufferBeginInfo
{
    internal int sType;
    internal nint pNext;
    internal uint flags;
    internal nint pInheritanceInfo;
}

[StructLayout(LayoutKind.Sequential)]
internal struct VkSubmitInfo
{
    internal int sType;
    internal nint pNext;
    internal uint waitSemaphoreCount;
    internal nint pWaitSemaphores;
    internal nint pWaitDstStageMask;
    internal uint commandBufferCount;
    internal nint pCommandBuffers;
    internal uint signalSemaphoreCount;
    internal nint pSignalSemaphores;
}
