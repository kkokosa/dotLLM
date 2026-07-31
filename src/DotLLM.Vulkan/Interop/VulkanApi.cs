using System.Runtime.InteropServices;

namespace DotLLM.Vulkan.Interop;

/// <summary>
/// Minimal P/Invoke declarations against the Vulkan loader (libvulkan.so.1 / vulkan-1.dll).
/// All functions return <c>VkResult</c> (int): 0 = VK_SUCCESS, negative = error,
/// positive = non-error status (e.g. VK_INCOMPLETE).
/// </summary>
/// <remarks>
/// The library name "vulkan-1" is rewritten to the correct OS binary by
/// <see cref="VulkanLibraryResolver"/> at runtime.
/// Handles (VkInstance, VkDevice, VkBuffer, etc.) cross the boundary as <c>nint</c>
/// so tensor payloads never traverse P/Invoke — only opaque pointers.
/// </remarks>
internal static partial class VulkanApi
{
    private const string LibName = "vulkan-1";

    // ── Instance ────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int vkCreateInstance(
        in VkInstanceCreateInfo pCreateInfo, nint pAllocator, out nint pInstance);

    [LibraryImport(LibName)]
    internal static partial void vkDestroyInstance(nint instance, nint pAllocator);

    // ── Physical device ─────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int vkEnumeratePhysicalDevices(
        nint instance, ref uint pPhysicalDeviceCount,
        [Out] nint[]? pPhysicalDevices);

    [LibraryImport(LibName)]
    internal static partial void vkGetPhysicalDeviceProperties(
        nint physicalDevice, out VkPhysicalDeviceProperties pProperties);

    [LibraryImport(LibName)]
    internal static partial void vkGetPhysicalDeviceMemoryProperties(
        nint physicalDevice, out VkPhysicalDeviceMemoryProperties pMemoryProperties);

    [LibraryImport(LibName)]
    internal static partial void vkGetPhysicalDeviceQueueFamilyProperties(
        nint physicalDevice, ref uint pQueueFamilyPropertyCount,
        [Out] VkQueueFamilyProperties[]? pQueueFamilyProperties);

    // ── Logical device ──────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int vkCreateDevice(
        nint physicalDevice, in VkDeviceCreateInfo pCreateInfo,
        nint pAllocator, out nint pDevice);

    [LibraryImport(LibName)]
    internal static partial void vkDestroyDevice(nint device, nint pAllocator);

    [LibraryImport(LibName)]
    internal static partial void vkGetDeviceQueue(
        nint device, uint queueFamilyIndex, uint queueIndex, out nint pQueue);

    [LibraryImport(LibName)]
    internal static partial int vkDeviceWaitIdle(nint device);

    // ── Memory ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int vkAllocateMemory(
        nint device, in VkMemoryAllocateInfo pAllocateInfo,
        nint pAllocator, out nint pMemory);

    [LibraryImport(LibName)]
    internal static partial void vkFreeMemory(
        nint device, nint memory, nint pAllocator);

    [LibraryImport(LibName)]
    internal static partial int vkMapMemory(
        nint device, nint memory, ulong offset, ulong size,
        uint flags, out nint ppData);

    [LibraryImport(LibName)]
    internal static partial void vkUnmapMemory(nint device, nint memory);

    // ── Buffers ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int vkCreateBuffer(
        nint device, in VkBufferCreateInfo pCreateInfo,
        nint pAllocator, out nint pBuffer);

    [LibraryImport(LibName)]
    internal static partial void vkDestroyBuffer(
        nint device, nint buffer, nint pAllocator);

    [LibraryImport(LibName)]
    internal static partial int vkBindBufferMemory(
        nint device, nint buffer, nint memory, ulong memoryOffset);

    [LibraryImport(LibName)]
    internal static partial void vkGetBufferMemoryRequirements(
        nint device, nint buffer, out VkMemoryRequirements pMemoryRequirements);

    // ── Shader modules ──────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int vkCreateShaderModule(
        nint device, in VkShaderModuleCreateInfo pCreateInfo,
        nint pAllocator, out nint pShaderModule);

    [LibraryImport(LibName)]
    internal static partial void vkDestroyShaderModule(
        nint device, nint shaderModule, nint pAllocator);

    // ── Pipeline layout & compute pipeline ──────────────────────────

    [LibraryImport(LibName)]
    internal static partial int vkCreatePipelineLayout(
        nint device, in VkPipelineLayoutCreateInfo pCreateInfo,
        nint pAllocator, out nint pPipelineLayout);

    [LibraryImport(LibName)]
    internal static partial void vkDestroyPipelineLayout(
        nint device, nint pipelineLayout, nint pAllocator);

    [LibraryImport(LibName)]
    internal static partial int vkCreateComputePipelines(
        nint device, nint pipelineCache, uint createInfoCount,
        in VkComputePipelineCreateInfo pCreateInfos,
        nint pAllocator, out nint pPipelines);

    [LibraryImport(LibName)]
    internal static partial void vkDestroyPipeline(
        nint device, nint pipeline, nint pAllocator);

    // ── Descriptor sets ─────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int vkCreateDescriptorSetLayout(
        nint device, in VkDescriptorSetLayoutCreateInfo pCreateInfo,
        nint pAllocator, out nint pSetLayout);

    [LibraryImport(LibName)]
    internal static partial void vkDestroyDescriptorSetLayout(
        nint device, nint descriptorSetLayout, nint pAllocator);

    [LibraryImport(LibName)]
    internal static partial int vkCreateDescriptorPool(
        nint device, in VkDescriptorPoolCreateInfo pCreateInfo,
        nint pAllocator, out nint pDescriptorPool);

    [LibraryImport(LibName)]
    internal static partial void vkDestroyDescriptorPool(
        nint device, nint descriptorPool, nint pAllocator);

    [LibraryImport(LibName)]
    internal static partial int vkAllocateDescriptorSets(
        nint device, in VkDescriptorSetAllocateInfo pAllocateInfo,
        out nint pDescriptorSets);

    [LibraryImport(LibName)]
    internal static partial void vkUpdateDescriptorSets(
        nint device, uint descriptorWriteCount,
        nint pDescriptorWrites,
        uint descriptorCopyCount, nint pDescriptorCopies);

    // ── Command pool & command buffers ──────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int vkCreateCommandPool(
        nint device, in VkCommandPoolCreateInfo pCreateInfo,
        nint pAllocator, out nint pCommandPool);

    [LibraryImport(LibName)]
    internal static partial void vkDestroyCommandPool(
        nint device, nint commandPool, nint pAllocator);

    [LibraryImport(LibName)]
    internal static partial int vkAllocateCommandBuffers(
        nint device, in VkCommandBufferAllocateInfo pAllocateInfo,
        out nint pCommandBuffers);

    [LibraryImport(LibName)]
    internal static partial void vkFreeCommandBuffers(
        nint device, nint commandPool, uint commandBufferCount,
        in nint pCommandBuffers);

    [LibraryImport(LibName)]
    internal static partial int vkBeginCommandBuffer(
        nint commandBuffer, in VkCommandBufferBeginInfo pBeginInfo);

    [LibraryImport(LibName)]
    internal static partial int vkEndCommandBuffer(nint commandBuffer);

    [LibraryImport(LibName)]
    internal static partial void vkCmdBindPipeline(
        nint commandBuffer, int pipelineBindPoint, nint pipeline);

    [LibraryImport(LibName)]
    internal static partial void vkCmdBindDescriptorSets(
        nint commandBuffer, int pipelineBindPoint, nint layout,
        uint firstSet, uint descriptorSetCount, in nint pDescriptorSets,
        uint dynamicOffsetCount, nint pDynamicOffsets);

    [LibraryImport(LibName)]
    internal static partial void vkCmdPushConstants(
        nint commandBuffer, nint layout, uint stageFlags,
        uint offset, uint size, nint pValues);

    [LibraryImport(LibName)]
    internal static partial void vkCmdDispatch(
        nint commandBuffer, uint groupCountX, uint groupCountY, uint groupCountZ);

    [LibraryImport(LibName)]
    internal static partial int vkQueueSubmit(
        nint queue, uint submitCount, in VkSubmitInfo pSubmits, nint fence);

    [LibraryImport(LibName)]
    internal static partial int vkQueueWaitIdle(nint queue);
}
