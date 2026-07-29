using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Full FP32 RMS Normalization: <c>output = (input / rms(input)) * weight</c>
/// with <c>rms = sqrt(mean(x^2) + eps)</c>. Processes a batch of rows in a
/// single launch — one workgroup per row.
/// </summary>
/// <remarks>
/// Mirrors the CUDA kernel <c>rmsnorm_f32</c> in
/// <c>native/kernels/rmsnorm_f32.cu</c> and matches the algorithm used by the
/// CPU path (sum-of-squares, divide by length, add epsilon under the sqrt).
/// </remarks>
public sealed class RmsNormF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;
    private const int PushConstantBytes = sizeof(uint) + sizeof(float); // n, eps

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private RmsNormF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>Loads <c>rmsnorm_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static RmsNormF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "rmsnorm_f32.spv");
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
        return new RmsNormF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches RMS norm over <paramref name="rowCount"/> rows of length
    /// <paramref name="n"/>. Synchronous — returns after <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="input">FP32 input buffer, <c>[rowCount, n]</c> row-major.</param>
    /// <param name="weight">FP32 per-feature scale, <c>[n]</c>.</param>
    /// <param name="output">FP32 output buffer, <c>[rowCount, n]</c> row-major.</param>
    /// <param name="rowCount">Number of rows to normalize.</param>
    /// <param name="n">Row length (number of features).</param>
    /// <param name="eps">Epsilon under the square root. Typical: 1e-5 or 1e-6.</param>
    public void Launch(
        VulkanDevice.Buffer input, VulkanDevice.Buffer weight, VulkanDevice.Buffer output,
        int rowCount, int n, float eps)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, input, weight, output, rowCount, n, eps);
        ctx.SubmitAndWait();
    }

    /// <summary>Records RMSNorm into <paramref name="cmdBuf"/> without submitting.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer input, VulkanDevice.Buffer weight, VulkanDevice.Buffer output,
        int rowCount, int n, float eps)
    {
        if (rowCount <= 0) throw new ArgumentOutOfRangeException(nameof(rowCount));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));

        long rowBytes = (long)n * sizeof(float);
        if (input.Size < rowBytes * rowCount) throw new ArgumentException("Input buffer too small.", nameof(input));
        if (weight.Size < rowBytes) throw new ArgumentException("Weight buffer too small.", nameof(weight));
        if (output.Size < rowBytes * rowCount) throw new ArgumentException("Output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[3] { input.Handle, weight.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        // Push constants: uint n, float eps (8 bytes total).
        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)n);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[4..], eps);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One workgroup per row.
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)rowCount, 1, 1);
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
