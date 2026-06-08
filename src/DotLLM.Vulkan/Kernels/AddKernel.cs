using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Proof-of-pipeline compute kernel that performs <c>c[i] = a[i] + b[i]</c>
/// over three FP32 storage buffers.
/// </summary>
/// <remarks>
/// This kernel exists to demonstrate and exercise the full Vulkan compute
/// path — SPIR-V load, descriptor set, push constant, command buffer record,
/// queue submit, wait. Real LLM kernels (rmsnorm, rope, attention, swiglu,
/// embedding, dequant) follow the same scaffolding.
/// </remarks>
public sealed class AddKernel : IDisposable
{
    private const int WorkgroupSize = 256;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private AddKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>add.spv</c> from the given directory and creates the pipeline.</summary>
    public static AddKernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "add.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: sizeof(uint)); // just `n`
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new AddKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the add kernel: <c>c[i] = a[i] + b[i]</c> for <paramref name="n"/>
    /// FP32 elements. All three buffers must be at least <c>n * sizeof(float)</c> bytes.
    /// Synchronous — the call returns after <c>vkQueueWaitIdle</c>. Legacy wrapper
    /// around <see cref="Record"/>.
    /// </summary>
    public void Launch(VulkanDevice.Buffer a, VulkanDevice.Buffer b, VulkanDevice.Buffer c, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, a, b, c, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the add kernel into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer a, VulkanDevice.Buffer b, VulkanDevice.Buffer c, int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));

        Span<nint> buffers = stackalloc nint[3] { a.Handle, b.Handle, c.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        uint pushN = (uint)n;
        VulkanApi.vkCmdPushConstants(
            cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
            0, sizeof(uint), (nint)(&pushN));

        uint groups = (uint)((n + WorkgroupSize - 1) / WorkgroupSize);
        VulkanApi.vkCmdDispatch(cmdBuf, groups, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
