using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Fused SwiGLU activation: <c>result[i] = gate[i] * sigmoid(gate[i]) * up[i]</c>.
/// Mirrors <c>DotLLM.Cpu.Kernels.FusedOps.SwiGLU</c> and the CUDA
/// <c>swiglu_f32</c> kernel.
/// </summary>
/// <remarks>
/// Pointwise, no reduction — one thread per output element. Used in the
/// Llama/Mistral/Phi/Qwen MLP block after the gate/up projections.
/// </remarks>
public sealed class SwiGluF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = sizeof(uint); // n

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private SwiGluF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>swiglu_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static SwiGluF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "swiglu_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

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
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new SwiGluF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches SwiGLU over <paramref name="n"/> elements. Synchronous —
    /// returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="gate">FP32 gate buffer (pre-activation).</param>
    /// <param name="up">FP32 up buffer.</param>
    /// <param name="result">FP32 output buffer. May alias <paramref name="up"/> on
    /// the CPU path; Vulkan storage buffers with <c>readonly</c>/<c>writeonly</c>
    /// qualifiers forbid aliasing, so callers must supply a distinct buffer here.</param>
    /// <param name="n">Element count.</param>
    public void Launch(
        VulkanDevice.Buffer gate, VulkanDevice.Buffer up, VulkanDevice.Buffer result, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, gate, up, result, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records SwiGLU into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer gate, VulkanDevice.Buffer up, VulkanDevice.Buffer result, int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));

        long bytes = (long)n * sizeof(float);
        if (gate.Size   < bytes) throw new ArgumentException("Gate buffer too small.",   nameof(gate));
        if (up.Size     < bytes) throw new ArgumentException("Up buffer too small.",     nameof(up));
        if (result.Size < bytes) throw new ArgumentException("Result buffer too small.", nameof(result));

        Span<nint> buffers = stackalloc nint[3] { gate.Handle, up.Handle, result.Handle };
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
