using System.Runtime.InteropServices;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Loads a SPIR-V compute shader into a <c>VkShaderModule</c> and caches compute
/// pipelines keyed by kernel entry-point name. One <see cref="VulkanModule"/>
/// corresponds to one .spv file (one kernel), mirroring how <c>CudaModule</c>
/// wraps a single .ptx file.
/// </summary>
/// <remarks>
/// SPIR-V is architecturally analogous to PTX: a forward-compatible shader IR
/// that the Vulkan driver translates to the vendor-specific ISA at pipeline-creation time.
/// The driver caches compiled pipelines on disk (implementation-dependent:
/// AMDGPU-PRO, Mesa-shader-cache, NVIDIA blob) so first-load cost is amortized.
/// </remarks>
public sealed class VulkanModule : IDisposable
{
    private readonly VulkanDevice _device;
    private nint _shaderModule;
    private bool _disposed;

    private VulkanModule(VulkanDevice device, nint shaderModule)
    {
        _device = device;
        _shaderModule = shaderModule;
    }

    internal nint Handle => _shaderModule;

    /// <summary>Loads a compiled SPIR-V shader from a file.</summary>
    public static VulkanModule LoadFromFile(VulkanDevice device, string spvPath)
    {
        byte[] spv = File.ReadAllBytes(spvPath);
        return LoadFromBytes(device, spv);
    }

    /// <summary>
    /// Loads a compiled SPIR-V shader from raw bytes. The blob must be a
    /// multiple of 4 bytes (SPIR-V is an array of <c>uint32_t</c>).
    /// </summary>
    public static unsafe VulkanModule LoadFromBytes(VulkanDevice device, byte[] spv)
    {
        if (spv.Length == 0 || (spv.Length & 3) != 0)
            throw new ArgumentException("SPIR-V blob must be a non-empty multiple of 4 bytes.", nameof(spv));

        fixed (byte* spvPtr = spv)
        {
            var ci = new VkShaderModuleCreateInfo
            {
                sType = VkStructureType.ShaderModuleCreateInfo,
                codeSize = (nuint)spv.Length,
                pCode = (nint)spvPtr,
            };
            VulkanApi.vkCreateShaderModule(device.Handle, ci, 0, out nint mod)
                .ThrowOnError("vkCreateShaderModule");
            return new VulkanModule(device, mod);
        }
    }

    /// <summary>
    /// Creates a compute pipeline for the given shader entry point, descriptor-set
    /// layout, and optional push-constant range. Caller owns the returned handles
    /// and is responsible for disposing them (via <see cref="DestroyPipeline"/>).
    /// </summary>
    public unsafe ComputePipeline CreateComputePipeline(
        string entryPoint,
        ReadOnlySpan<VkDescriptorBinding> bindings,
        uint pushConstantBytes = 0)
    {
        // 1. Descriptor-set layout — one binding per storage buffer in the shader.
        nint setLayout = 0;
        nint pipelineLayout = 0;
        nint pipeline = 0;
        try
        {
            int n = bindings.Length;
            Span<VkDescriptorSetLayoutBinding> layoutBindings = stackalloc VkDescriptorSetLayoutBinding[Math.Max(1, n)];
            for (int i = 0; i < n; i++)
            {
                layoutBindings[i] = new VkDescriptorSetLayoutBinding
                {
                    binding = bindings[i].Binding,
                    descriptorType = VkDescriptorType.StorageBuffer,
                    descriptorCount = 1,
                    stageFlags = VkShaderStageFlags.Compute,
                };
            }

            fixed (VkDescriptorSetLayoutBinding* bindingsPtr = layoutBindings)
            {
                var dslCi = new VkDescriptorSetLayoutCreateInfo
                {
                    sType = VkStructureType.DescriptorSetLayoutCreateInfo,
                    bindingCount = (uint)n,
                    pBindings = (nint)bindingsPtr,
                };
                VulkanApi.vkCreateDescriptorSetLayout(_device.Handle, dslCi, 0, out setLayout)
                    .ThrowOnError("vkCreateDescriptorSetLayout");
            }

            // 2. Pipeline layout (set layouts + optional push-constant range).
            var pushRange = new VkPushConstantRange
            {
                stageFlags = VkShaderStageFlags.Compute,
                offset = 0,
                size = pushConstantBytes,
            };

            VkPipelineLayoutCreateInfo plCi = default;
            plCi.sType = VkStructureType.PipelineLayoutCreateInfo;
            plCi.setLayoutCount = 1;
            nint setLayoutLocal = setLayout;
            plCi.pSetLayouts = (nint)(&setLayoutLocal);
            if (pushConstantBytes > 0)
            {
                plCi.pushConstantRangeCount = 1;
                plCi.pPushConstantRanges = (nint)(&pushRange);
            }

            VulkanApi.vkCreatePipelineLayout(_device.Handle, plCi, 0, out pipelineLayout)
                .ThrowOnError("vkCreatePipelineLayout");

            // 3. Compute pipeline — shader stage + pipeline layout.
            byte[] entryUtf8 = System.Text.Encoding.UTF8.GetBytes(entryPoint + "\0");
            fixed (byte* entryPtr = entryUtf8)
            {
                var stage = new VkPipelineShaderStageCreateInfo
                {
                    sType = VkStructureType.PipelineShaderStageCreateInfo,
                    stage = VkShaderStageFlags.Compute,
                    module = _shaderModule,
                    pName = (nint)entryPtr,
                };
                var pipeCi = new VkComputePipelineCreateInfo
                {
                    sType = VkStructureType.ComputePipelineCreateInfo,
                    stage = stage,
                    layout = pipelineLayout,
                    basePipelineIndex = -1,
                };

                VulkanApi.vkCreateComputePipelines(_device.Handle, 0, 1, pipeCi, 0, out pipeline)
                    .ThrowOnError("vkCreateComputePipelines");
            }

            var result = new ComputePipeline(_device, setLayout, pipelineLayout, pipeline);
            // Transfer ownership — clear locals so finally{} does not double-free.
            setLayout = 0; pipelineLayout = 0; pipeline = 0;
            return result;
        }
        finally
        {
            if (pipeline != 0) VulkanApi.vkDestroyPipeline(_device.Handle, pipeline, 0);
            if (pipelineLayout != 0) VulkanApi.vkDestroyPipelineLayout(_device.Handle, pipelineLayout, 0);
            if (setLayout != 0) VulkanApi.vkDestroyDescriptorSetLayout(_device.Handle, setLayout, 0);
        }
    }

    /// <summary>Releases the associated pipeline, layout, and descriptor-set layout.</summary>
    public void DestroyPipeline(ComputePipeline pipeline) => pipeline.Dispose();

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_shaderModule != 0)
        {
            VulkanApi.vkDestroyShaderModule(_device.Handle, _shaderModule, 0);
            _shaderModule = 0;
        }
    }
}

/// <summary>
/// Describes one storage-buffer binding slot in a compute shader's descriptor set.
/// </summary>
public readonly record struct VkDescriptorBinding(uint Binding);

/// <summary>
/// A compute pipeline bundle: <c>VkPipeline</c> plus the descriptor-set-layout
/// and pipeline-layout it was created against.
/// </summary>
public sealed class ComputePipeline : IDisposable
{
    private readonly VulkanDevice _device;
    private nint _setLayout;
    private nint _pipelineLayout;
    private nint _pipeline;

    internal ComputePipeline(VulkanDevice device, nint setLayout, nint pipelineLayout, nint pipeline)
    {
        _device = device;
        _setLayout = setLayout;
        _pipelineLayout = pipelineLayout;
        _pipeline = pipeline;
    }

    /// <summary>The <c>VkPipeline</c> handle.</summary>
    public nint Pipeline => _pipeline;

    /// <summary>The <c>VkPipelineLayout</c> handle.</summary>
    public nint Layout => _pipelineLayout;

    /// <summary>The <c>VkDescriptorSetLayout</c> handle.</summary>
    public nint DescriptorSetLayout => _setLayout;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_pipeline != 0)
        {
            VulkanApi.vkDestroyPipeline(_device.Handle, _pipeline, 0);
            _pipeline = 0;
        }
        if (_pipelineLayout != 0)
        {
            VulkanApi.vkDestroyPipelineLayout(_device.Handle, _pipelineLayout, 0);
            _pipelineLayout = 0;
        }
        if (_setLayout != 0)
        {
            VulkanApi.vkDestroyDescriptorSetLayout(_device.Handle, _setLayout, 0);
            _setLayout = 0;
        }
    }
}
