using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Shared host-side helpers every kernel uses: descriptor-pool allocation,
/// descriptor-set allocation + buffer writes, pipeline barrier insertion,
/// and the synchronous submit/wait fallback that wraps the fence-pipelined
/// <c>Record</c> path into the legacy <c>Launch</c> API.
/// </summary>
/// <remarks>
/// Extracting this into one place lets every kernel share the pool-sizing
/// story (1 pool per kernel instance, <see cref="DefaultMaxSetsPerPool"/>
/// descriptor sets per pool) and the barrier shape used on the hot forward
/// path (SHADER_WRITE → SHADER_READ between kernels). The individual
/// <c>.cs</c> files stay focused on push-constants and dispatch shape.
/// </remarks>
internal static class KernelSupport
{
    /// <summary>
    /// Default upper bound on concurrent descriptor sets allocated from a
    /// single kernel pool across one forward pass. Sized to cover the
    /// matmul kernel's ~211 dispatches on SmolLM-135M (7 per layer × 30
    /// layers + 1 lm_head) with 5× headroom for future models.
    /// </summary>
    internal const uint DefaultMaxSetsPerPool = 1024u;

    /// <summary>
    /// Creates a descriptor pool sized for <paramref name="maxSets"/>
    /// concurrent descriptor sets with <paramref name="buffersPerSet"/>
    /// storage-buffer bindings in each. Used by every kernel's
    /// <c>CreateDescriptorPool</c> so they share sizing policy.
    /// </summary>
    internal static unsafe nint CreateDescriptorPool(
        VulkanDevice device, uint buffersPerSet, uint maxSets = DefaultMaxSetsPerPool)
    {
        var poolSize = new VkDescriptorPoolSize
        {
            type = VkDescriptorType.StorageBuffer,
            descriptorCount = buffersPerSet * maxSets,
        };
        VkDescriptorPoolCreateInfo ci = default;
        ci.sType = VkStructureType.DescriptorPoolCreateInfo;
        ci.maxSets = maxSets;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = (nint)(&poolSize);
        VulkanApi.vkCreateDescriptorPool(device.Handle, ci, 0, out nint pool)
            .ThrowOnError("vkCreateDescriptorPool");
        return pool;
    }

    /// <summary>
    /// Allocates one descriptor set from <paramref name="pool"/> using
    /// <paramref name="setLayout"/>. Throws on pool exhaustion — callers
    /// are expected to <see cref="ResetPool"/> at the start of each forward.
    /// </summary>
    internal static unsafe nint AllocateDescriptorSet(VulkanDevice device, nint pool, nint setLayout)
    {
        nint setLayoutLocal = setLayout;
        var dsai = new VkDescriptorSetAllocateInfo
        {
            sType = VkStructureType.DescriptorSetAllocateInfo,
            descriptorPool = pool,
            descriptorSetCount = 1,
            pSetLayouts = (nint)(&setLayoutLocal),
        };
        VulkanApi.vkAllocateDescriptorSets(device.Handle, dsai, out nint descriptorSet)
            .ThrowOnError("vkAllocateDescriptorSets");
        return descriptorSet;
    }

    /// <summary>
    /// Writes a contiguous set of storage-buffer bindings (<paramref name="buffers"/>)
    /// into <paramref name="descriptorSet"/>, starting at binding index 0.
    /// </summary>
    internal static unsafe void WriteBufferBindings(
        VulkanDevice device, nint descriptorSet, ReadOnlySpan<nint> buffers)
    {
        int n = buffers.Length;
        Span<VkDescriptorBufferInfo> bufferInfos = stackalloc VkDescriptorBufferInfo[n];
        for (int i = 0; i < n; i++)
        {
            bufferInfos[i] = new VkDescriptorBufferInfo
            {
                buffer = buffers[i],
                offset = 0,
                range = ulong.MaxValue, // VK_WHOLE_SIZE
            };
        }

        Span<VkWriteDescriptorSet> writes = stackalloc VkWriteDescriptorSet[n];
        fixed (VkDescriptorBufferInfo* bufPtr = bufferInfos)
        {
            for (int i = 0; i < n; i++)
            {
                writes[i] = new VkWriteDescriptorSet
                {
                    sType = VkStructureType.WriteDescriptorSet,
                    dstSet = descriptorSet,
                    dstBinding = (uint)i,
                    descriptorCount = 1,
                    descriptorType = VkDescriptorType.StorageBuffer,
                    pBufferInfo = (nint)(bufPtr + i),
                };
            }
            fixed (VkWriteDescriptorSet* writesPtr = writes)
            {
                VulkanApi.vkUpdateDescriptorSets(device.Handle, (uint)n, (nint)writesPtr, 0, 0);
            }
        }
    }

    /// <summary>Resets all descriptor sets allocated from <paramref name="pool"/>.</summary>
    internal static void ResetPool(VulkanDevice device, nint pool)
        => VulkanApi.vkResetDescriptorPool(device.Handle, pool, 0).ThrowOnError("vkResetDescriptorPool");

    /// <summary>
    /// Inserts a <c>COMPUTE_SHADER → COMPUTE_SHADER</c> pipeline barrier with
    /// a <c>SHADER_WRITE → SHADER_READ</c> memory dependency. Used between
    /// every pair of kernels that share a command buffer to ensure the
    /// second kernel sees the first kernel's writes.
    /// </summary>
    internal static unsafe void ComputeToComputeBarrier(nint cmdBuf)
    {
        var barrier = new VkMemoryBarrier
        {
            sType = VkStructureType.MemoryBarrier,
            srcAccessMask = VkAccessFlags.ShaderWrite,
            dstAccessMask = VkAccessFlags.ShaderRead | VkAccessFlags.ShaderWrite,
        };
        VulkanApi.vkCmdPipelineBarrier(
            cmdBuf,
            srcStageMask: VkPipelineStageFlags.ComputeShader,
            dstStageMask: VkPipelineStageFlags.ComputeShader,
            dependencyFlags: 0,
            memoryBarrierCount: 1, pMemoryBarriers: barrier,
            bufferMemoryBarrierCount: 0, pBufferMemoryBarriers: 0,
            imageMemoryBarrierCount: 0, pImageMemoryBarriers: 0);
    }

    /// <summary>
    /// Inserts a <c>TRANSFER → COMPUTE_SHADER</c> barrier for the
    /// KV-cache-update → attention handoff (the KV rows land via
    /// <c>vkCmdCopyBuffer</c>, which is in the TRANSFER stage, but the
    /// attention kernel reads them in COMPUTE_SHADER).
    /// </summary>
    internal static unsafe void TransferToComputeBarrier(nint cmdBuf)
    {
        var barrier = new VkMemoryBarrier
        {
            sType = VkStructureType.MemoryBarrier,
            srcAccessMask = VkAccessFlags.TransferWrite,
            dstAccessMask = VkAccessFlags.ShaderRead,
        };
        VulkanApi.vkCmdPipelineBarrier(
            cmdBuf,
            srcStageMask: VkPipelineStageFlags.Transfer,
            dstStageMask: VkPipelineStageFlags.ComputeShader,
            dependencyFlags: 0,
            memoryBarrierCount: 1, pMemoryBarriers: barrier,
            bufferMemoryBarrierCount: 0, pBufferMemoryBarriers: 0,
            imageMemoryBarrierCount: 0, pImageMemoryBarriers: 0);
    }

    /// <summary>
    /// Inserts a <c>HOST → COMPUTE_SHADER</c> barrier so compute kernels see
    /// host writes to host-visible host-coherent buffers that were made
    /// before the submit. Vulkan's host-coherent guarantee covers visibility
    /// through vkQueueSubmit, but an explicit HOST_WRITE→SHADER_READ barrier
    /// is the documented way to make the ordering safe across drivers when
    /// we've done the upload right before recording.
    /// </summary>
    internal static unsafe void HostToComputeBarrier(nint cmdBuf)
    {
        var barrier = new VkMemoryBarrier
        {
            sType = VkStructureType.MemoryBarrier,
            srcAccessMask = VkAccessFlags.HostWrite,
            dstAccessMask = VkAccessFlags.ShaderRead | VkAccessFlags.TransferRead,
        };
        VulkanApi.vkCmdPipelineBarrier(
            cmdBuf,
            srcStageMask: VkPipelineStageFlags.Host,
            dstStageMask: VkPipelineStageFlags.ComputeShader | VkPipelineStageFlags.Transfer,
            dependencyFlags: 0,
            memoryBarrierCount: 1, pMemoryBarriers: barrier,
            bufferMemoryBarrierCount: 0, pBufferMemoryBarriers: 0,
            imageMemoryBarrierCount: 0, pImageMemoryBarriers: 0);
    }

    /// <summary>
    /// Inserts a <c>COMPUTE_SHADER → HOST</c> barrier so the host can read
    /// back a compute kernel's output (specifically the final LM-head
    /// logits) after the submit completes.
    /// </summary>
    internal static unsafe void ComputeToHostBarrier(nint cmdBuf)
    {
        var barrier = new VkMemoryBarrier
        {
            sType = VkStructureType.MemoryBarrier,
            srcAccessMask = VkAccessFlags.ShaderWrite,
            dstAccessMask = VkAccessFlags.HostRead,
        };
        VulkanApi.vkCmdPipelineBarrier(
            cmdBuf,
            srcStageMask: VkPipelineStageFlags.ComputeShader,
            dstStageMask: VkPipelineStageFlags.Host,
            dependencyFlags: 0,
            memoryBarrierCount: 1, pMemoryBarriers: barrier,
            bufferMemoryBarrierCount: 0, pBufferMemoryBarriers: 0,
            imageMemoryBarrierCount: 0, pImageMemoryBarriers: 0);
    }
}
