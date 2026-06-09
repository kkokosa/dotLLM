using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Per-feature bias add: <c>output[t, i] += bias[i]</c> for every
/// <c>(t, i)</c> pair. Used downstream by bias-bearing models (Phi-3,
/// Qwen3, DeepSeek-V2) after Q/K/V/O/Gate/Up/Down projections to keep
/// the whole forward in one submit.
/// </summary>
public sealed class BiasAddF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = 2 * sizeof(uint); // seqLen, outputDim

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private BiasAddF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 2);
    }

    /// <summary>Loads <c>bias_add_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static BiasAddF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "bias_add_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[2];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 2);
        return new BiasAddF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the in-place bias add. <paramref name="output"/> is
    /// <c>[seqLen, outputDim]</c> row-major FP32; <paramref name="bias"/>
    /// is <c>[outputDim]</c>. Synchronous — returns after
    /// <c>vkQueueWaitIdle</c>. Legacy wrapper around <see cref="Record"/>.
    /// </summary>
    /// <param name="output">FP32 output buffer, <c>[seqLen, outputDim]</c> row-major.</param>
    /// <param name="bias">FP32 per-feature bias, <c>[outputDim]</c>.</param>
    /// <param name="seqLen">Number of rows (tokens).</param>
    /// <param name="outputDim">Row length (number of features).</param>
    public void Launch(
        VulkanDevice.Buffer output, VulkanDevice.Buffer bias, int seqLen, int outputDim)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, output, bias, seqLen, outputDim);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the in-place bias add into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer output, VulkanDevice.Buffer bias, int seqLen, int outputDim)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));

        long outBytes = (long)seqLen * outputDim * sizeof(float);
        long biasBytes = (long)outputDim * sizeof(float);
        if (output.Size < outBytes) throw new ArgumentException("output buffer too small.", nameof(output));
        if (bias.Size < biasBytes) throw new ArgumentException("bias buffer too small.", nameof(bias));

        Span<nint> buffers = stackalloc nint[2] { output.Handle, bias.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        // Push constants: uint seqLen, uint outputDim (8 bytes total).
        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)seqLen);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)outputDim);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One thread per output element, 256 threads per workgroup.
        long total = (long)seqLen * outputDim;
        uint groupCount = (uint)((total + WorkgroupSize - 1) / WorkgroupSize);
        VulkanApi.vkCmdDispatch(cmdBuf, groupCount, 1, 1);
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
